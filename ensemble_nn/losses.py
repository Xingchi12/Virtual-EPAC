"""Deterministic and probabilistic regression objectives."""

from __future__ import annotations

from typing import Optional, Sequence, Union

import torch
from torch import nn
from torch.nn import functional as F


class RegressionLoss(nn.Module):
    """
    Calculate MSE, RMSE or diagonal Gaussian negative log-likelihood.

    Supports two ways of combining per-output losses into a scalar:

    - ``"uniform"`` (or an explicit list of weights): a fixed weighted
      average, as before.
    - ``"uncertainty"``: homoscedastic uncertainty weighting (Kendall &
      Gal, 2018). One learnable log-variance parameter per task is
      introduced purely to balance gradient magnitudes across outputs
      with very different scales and units; this is independent of the
      per-sample aleatoric log-variance already predicted by the
      network heads for ``gaussian_nll``.

    An optional auxiliary binary-classification task (for example "did
    injection occur at all") can be combined using the same weighting
    scheme via :meth:`injection_loss`, reserved as the last task slot.
    """

    def __init__(
        self,
        name: str = "gaussian_nll",
        output_weights: Union[str, Sequence[float]] = "uniform",
        variance_regularization: float = 0.0,
        n_outputs: Optional[int] = None,
        n_extra_tasks: int = 0,
    ) -> None:
        super().__init__()
        self.name = name.lower()
        self.variance_regularization = variance_regularization

        supported = {"mse", "rmse", "gaussian_nll"}
        if self.name not in supported:
            raise ValueError(f"Unsupported loss '{name}'. Choose from {supported}.")

        self.weighting_mode = (
            output_weights.lower()
            if isinstance(output_weights, str)
            else "fixed"
        )

        self.task_log_vars = None
        weights = torch.tensor([])

        if self.weighting_mode == "uncertainty":
            if n_outputs is None:
                raise ValueError(
                    "n_outputs must be provided when "
                    "output_weights='uncertainty'."
                )
            self.task_log_vars = nn.Parameter(
                torch.zeros(n_outputs + n_extra_tasks)
            )
        elif self.weighting_mode != "uniform":
            weights = torch.as_tensor(output_weights, dtype=torch.float32)

        self.register_buffer("weights", weights)

    def _per_output_loss(
        self,
        mean: torch.Tensor,
        target: torch.Tensor,
        log_variance: Optional[torch.Tensor],
        mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Return the masked per-output loss, shape ``[n_outputs]``."""
        if mean.shape != target.shape:
            raise ValueError(
                f"Prediction shape {mean.shape} differs from target "
                f"shape {target.shape}."
            )

        if mask is None:
            mask = torch.ones_like(target)

        denominator = mask.sum(dim=0).clamp_min(1.0)

        squared_error = (mean - target).square()

        if self.name == "gaussian_nll":
            if log_variance is None:
                raise ValueError("Gaussian NLL requires predicted log variance.")
            elementwise = 0.5 * (
                log_variance
                + squared_error * torch.exp(-log_variance)
            )
            elementwise = elementwise + (
                self.variance_regularization * log_variance.square()
            )
            return (elementwise * mask).sum(dim=0) / denominator

        if self.name == "rmse":
            masked_squared_error = (squared_error * mask).sum(dim=0) / denominator
            return torch.sqrt(masked_squared_error + 1.0e-8)

        return (squared_error * mask).sum(dim=0) / denominator

    def _combine(self, per_task: torch.Tensor) -> torch.Tensor:
        """Combine per-task scalar losses into one scalar."""
        if self.weighting_mode == "uncertainty":
            task_log_vars = self.task_log_vars[: per_task.shape[0]]
            precision = torch.exp(-task_log_vars)
            return (precision * per_task + task_log_vars).sum()

        if self.weights.numel() == 0:
            weights = torch.ones_like(per_task)
        else:
            weights = self.weights.to(per_task)
        return (per_task * weights).sum() / weights.sum()

    def forward(
        self,
        mean: torch.Tensor,
        target: torch.Tensor,
        log_variance: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Return the combined scalar loss and the masked per-output losses.

        The per-output vector is useful for logging and does not include
        any auxiliary classification task.
        """
        per_output = self._per_output_loss(mean, target, log_variance, mask)
        return self._combine(per_output), per_output

    def injection_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Return the combined and raw binary-cross-entropy injection loss.

        The raw BCE (second element) is unweighted and intended for
        logging; the first element already includes uncertainty
        weighting when configured, using the last reserved task slot.
        """
        bce = F.binary_cross_entropy_with_logits(
            logits.reshape(-1),
            labels.reshape(-1),
        )

        if self.weighting_mode == "uncertainty":
            task_log_var = self.task_log_vars[-1]
            precision = torch.exp(-task_log_var)
            return precision * bce + task_log_var, bce.detach()

        return bce, bce.detach()
