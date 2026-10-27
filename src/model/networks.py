"""Compact neural-network families for small tabular simulator datasets."""

from __future__ import annotations

from typing import Sequence

import math
import torch
from torch import nn


def make_activation(name: str) -> nn.Module:
    """Create an activation module from a configuration name."""
    activations = {
        "relu": nn.ReLU,
        "gelu": nn.GELU,
        "silu": nn.SiLU,
        "tanh": nn.Tanh,
    }
    try:
        return activations[name.lower()]()
    except KeyError as error:
        raise ValueError(f"Unsupported activation '{name}'.") from error


class ResidualBlock(nn.Module):
    """A regularised fully connected residual block."""

    def __init__(
        self,
        dimension: int,
        activation: str,
        dropout: float,
        layer_norm: bool,
    ) -> None:
        super().__init__()
        normalization = nn.LayerNorm(dimension) if layer_norm else nn.Identity()
        self.layers = nn.Sequential(
            nn.Linear(dimension, dimension),
            normalization,
            make_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(dimension, dimension),
            nn.Dropout(dropout),
        )
        self.activation = make_activation(activation)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Apply a residual update while preserving the feature dimension."""
        return self.activation(inputs + self.layers(inputs))


def make_mlp(
    input_dimension: int,
    hidden_dimensions: Sequence[int],
    activation: str,
    dropout: float,
    layer_norm: bool,
    residual: bool,
) -> tuple[nn.Sequential, int]:
    """Construct a compact MLP and return it with its output dimension."""
    layers: list[nn.Module] = []
    current_dimension = input_dimension

    for hidden_dimension in hidden_dimensions:
        layers.extend(
            [
                nn.Linear(current_dimension, hidden_dimension),
                (
                    nn.LayerNorm(hidden_dimension)
                    if layer_norm
                    else nn.Identity()
                ),
                make_activation(activation),
                nn.Dropout(dropout),
            ]
        )
        if residual:
            layers.append(
                ResidualBlock(
                    hidden_dimension,
                    activation,
                    dropout,
                    layer_norm,
                )
            )
        current_dimension = hidden_dimension

    return nn.Sequential(*layers), current_dimension


class BaseSurrogate(nn.Module):
    """Base class that separates Gaussian means and constrained log variances."""

    def __init__(self, output_count: int, configuration) -> None:
        super().__init__()
        self.output_count = output_count
        self.probabilistic = bool(configuration.probabilistic)
        self.log_variance_bounds = tuple(configuration.log_var_bounds)

    def split_distribution(
        self,
        raw_output: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Convert raw head values into means and optional log variances."""
        if not self.probabilistic:
            return raw_output, None
        mean, log_variance = raw_output.chunk(2, dim=-1)
        return mean, log_variance.clamp(*self.log_variance_bounds)


class MultiOutputNetwork(BaseSurrogate):
    """One shared network with a joint head for all outputs."""

    def __init__(self, input_count: int, output_count: int, configuration) -> None:
        super().__init__(output_count, configuration)
        self.body, final_dimension = make_mlp(
            input_count,
            configuration.hidden_dims,
            configuration.activation,
            configuration.dropout,
            configuration.layer_norm,
            configuration.residual,
        )
        head_width = output_count * (2 if self.probabilistic else 1)
        self.head = nn.Linear(final_dimension, head_width)

        self.injection_head_enabled = bool(
            getattr(configuration, "injection_head", False)
        )
        self.injection_head = (
            nn.Linear(final_dimension, 1)
            if self.injection_head_enabled
            else None
        )

    def forward(self, inputs: torch.Tensor) -> dict[str, torch.Tensor | None]:
        """Predict all output distributions through a joint head."""
        features = self.body(inputs)
        mean, log_variance = self.split_distribution(
            self.head(features)
        )
        injection_logit = (
            self.injection_head(features)
            if self.injection_head is not None
            else None
        )
        return {
            "mean": mean,
            "log_var": log_variance,
            "injection_logit": injection_logit,
        }


class IndependentNetworks(BaseSurrogate):
    """Use one independent neural network for each output variable."""

    def __init__(self, input_count: int, output_count: int, configuration) -> None:
        super().__init__(output_count, configuration)
        self.networks = nn.ModuleList()

        for _ in range(output_count):
            body, final_dimension = make_mlp(
                input_count,
                configuration.hidden_dims,
                configuration.activation,
                configuration.dropout,
                configuration.layer_norm,
                configuration.residual,
            )
            head_width = 2 if self.probabilistic else 1
            self.networks.append(
                nn.Sequential(body, nn.Linear(final_dimension, head_width))
            )

    def forward(self, inputs: torch.Tensor) -> dict[str, torch.Tensor | None]:
        """Concatenate predictions from the six independent networks."""
        raw_output = torch.cat(
            [network(inputs) for network in self.networks],
            dim=-1,
        )
        if self.probabilistic:
            mean = raw_output[:, 0::2]
            log_variance = raw_output[:, 1::2].clamp(
                *self.log_variance_bounds
            )
        else:
            mean, log_variance = raw_output, None
        return {"mean": mean, "log_var": log_variance}


class SharedBranchNetwork(BaseSurrogate):
    """Use a shared physical representation and one branch per output."""

    def __init__(self, input_count: int, output_count: int, configuration) -> None:
        super().__init__(output_count, configuration)
        self.trunk, trunk_dimension = make_mlp(
            input_count,
            configuration.trunk_dims,
            configuration.activation,
            configuration.dropout,
            configuration.layer_norm,
            configuration.residual,
        )
        self.branches = nn.ModuleList()

        for _ in range(output_count):
            branch, branch_dimension = make_mlp(
                trunk_dimension,
                configuration.branch_dims,
                configuration.activation,
                configuration.dropout,
                configuration.layer_norm,
                configuration.residual,
            )
            head_width = 2 if self.probabilistic else 1
            self.branches.append(
                nn.Sequential(
                    branch,
                    nn.Linear(branch_dimension, head_width),
                )
            )

    def forward(self, inputs: torch.Tensor) -> dict[str, torch.Tensor | None]:
        """Predict each target from a shared trunk and target-specific branch."""
        features = self.trunk(inputs)
        raw_output = torch.cat(
            [branch(features) for branch in self.branches],
            dim=-1,
        )
        if self.probabilistic:
            mean = raw_output[:, 0::2]
            log_variance = raw_output[:, 1::2].clamp(
                *self.log_variance_bounds
            )
        else:
            mean, log_variance = raw_output, None
        return {"mean": mean, "log_var": log_variance}


class GroupedNetwork(BaseSurrogate):
    """Use a common trunk and physically motivated output-group heads."""

    def __init__(
        self,
        input_count: int,
        output_names: list[str],
        configuration,
    ) -> None:
        super().__init__(len(output_names), configuration)
        self.trunk, trunk_dimension = make_mlp(
            input_count,
            configuration.trunk_dims,
            configuration.activation,
            configuration.dropout,
            configuration.layer_norm,
            configuration.residual,
        )
        self.group_indices: list[list[int]] = []
        self.heads = nn.ModuleList()
        assigned_indices: list[int] = []
        
        name_to_index = {
        n.lower(): i
        for i, n in enumerate(output_names)
        }

        for group_names in configuration.groups.values():
            indices = [
                name_to_index[name.lower()] 
                for name in group_names
            ]
            assigned_indices.extend(indices)
            self.group_indices.append(indices)
            branch, final_dimension = make_mlp(
                trunk_dimension,
                configuration.branch_dims,
                configuration.activation,
                configuration.dropout,
                configuration.layer_norm,
                configuration.residual,
            )
            width = len(indices) * (2 if self.probabilistic else 1)
            self.heads.append(
                nn.Sequential(branch, nn.Linear(final_dimension, width))
            )

        expected = list(range(len(output_names)))
        if sorted(assigned_indices) != expected:
            raise ValueError(
                "Output groups must contain every output exactly once."
            )

    def forward(self, inputs: torch.Tensor) -> dict[str, torch.Tensor | None]:
        """Predict output groups and restore the configured output order."""
        features = self.trunk(inputs)
        
        mean = inputs.new_empty((len(inputs), self.output_count))
        log_variance = (
            inputs.new_empty((len(inputs), self.output_count))
            if self.probabilistic
            else None
        )

        for indices, head in zip(self.group_indices, self.heads):
            raw_output = head(features)
            if self.probabilistic:
                
                group_mean, group_log_variance = raw_output.chunk(2, dim=-1)
                
                mean[:, indices] = group_mean.to(mean.dtype)
                
                log_variance[:, indices] = (
                    group_log_variance
                    .clamp(*self.log_variance_bounds)
                    .to(log_variance.dtype)
                )
            else:
                mean[:, indices] = raw_output

        return {"mean": mean, "log_var": log_variance}