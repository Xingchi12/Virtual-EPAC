
from __future__ import annotations

from typing import Sequence

import math
import torch
from torch import nn

from model.networks import *

class DeepEnsemble(nn.Module):
    """
    Combine independently initialised networks into a predictive mixture.

    Each probabilistic ensemble member represents a diagonal multivariate
    Gaussian distribution. The complete ensemble therefore represents an
    equally weighted mixture of diagonal Gaussian distributions.
    """

    SUPPORTED_SAMPLING_METHODS = (
        "analytical_mean",
        "sample_mean",
        "marginal_median",
        "highest_density",
        "top_k_density_mean",
        "moment_matched_sample_mean",
    )

    def __init__(self, members: list[nn.Module]) -> None:
        super().__init__()

        if not members:
            raise ValueError(
                "A DeepEnsemble must contain at least one member."
            )

        self.members = nn.ModuleList(members)

    def forward_members(
        self,
        inputs: torch.Tensor,
    ) -> list[dict[str, torch.Tensor | None]]:
        """
        Return predictions from every ensemble member.

        This method is useful during training when each ensemble member
        needs to be accessed separately.
        """
        return [member(inputs) for member in self.members]

    def forward(
        self,
        inputs: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Return analytical ensemble mean and predictive uncertainty.

        This preserves the original model interface used during training.
        """
        member_means, member_log_variances, member_injection_logits = (
            self.member_parameters(inputs)
        )

        aggregated = self.aggregate_member_parameters(
            member_means=member_means,
            member_log_variances=member_log_variances,
        )

        injection = self.aggregate_injection_logits(
            member_injection_logits
        )
        if injection is not None:
            aggregated["injection_probability"] = injection["probability"]
            aggregated["injection_epistemic_std"] = injection["epistemic_std"]

        return aggregated

    def predict(
        self,
        inputs: torch.Tensor,
        sampling_method: str = "analytical_mean",
        sample_count: int = 1000,
        top_k: int | None = None,
        return_samples: bool = False,
        covariance_jitter: float = 1.0e-6,
    ) -> dict[str, torch.Tensor | None]:
        """
        Predict using an analytical or sampling-based point estimator.
    
        Parameters
        ----------
        inputs:
            Scaled model inputs with shape:
    
            [batch_size, input_count]
    
        sampling_method:
            One of:
    
            - analytical_mean
            - sample_mean
            - marginal_median
            - highest_density
            - top_k_density_mean
            - moment_matched_sample_mean
    
        sample_count:
            Number of Monte Carlo samples per input. The default is 1,000.
            This value is ignored for analytical_mean.
    
        top_k:
            Number of highest-density mixture samples retained by
            top_k_density_mean.
    
        return_samples:
            If True, include the generated samples in the returned dictionary.
    
        covariance_jitter:
            Initial diagonal regularisation used when constructing the
            moment-matched multivariate Gaussian.
    
        Returns
        -------
        dict[str, torch.Tensor | None]
            Dictionary containing:
    
            - mean: analytical ensemble mean
            - log_var: analytical total marginal log variance
            - epistemic_var: ensemble disagreement variance
            - aleatoric_var: average member variance
            - prediction: selected point prediction
            - covariance: full total covariance
            - samples: optional predictive samples
        """
        if sampling_method not in self.SUPPORTED_SAMPLING_METHODS:
            supported = ", ".join(
                self.SUPPORTED_SAMPLING_METHODS
            )
    
            raise ValueError(
                f"Unsupported sampling method '{sampling_method}'. "
                f"Supported methods are: {supported}."
            )
    
        if sample_count < 1:
            raise ValueError(
                f"sample_count must be at least 1, got {sample_count}."
            )
    
        if sampling_method == "top_k_density_mean":
            if top_k is None:
                raise ValueError(
                    "top_k must be provided when sampling_method is "
                    "'top_k_density_mean'."
                )
    
            if top_k < 1:
                raise ValueError(
                    f"top_k must be at least 1, got {top_k}."
                )
    
            if top_k > sample_count:
                raise ValueError(
                    f"top_k={top_k} cannot exceed "
                    f"sample_count={sample_count}."
                )
    
        # Run every ensemble member once.
        member_means, member_log_variances, member_injection_logits = (
            self.member_parameters(inputs)
        )

        # Calculate the standard analytical ensemble quantities.
        analytical_output = self.aggregate_member_parameters(
            member_means=member_means,
            member_log_variances=member_log_variances,
        )

        injection = self.aggregate_injection_logits(
            member_injection_logits
        )
        if injection is not None:
            analytical_output["injection_probability"] = injection["probability"]
            analytical_output["injection_epistemic_std"] = injection["epistemic_std"]

        # Calculate the full covariance, including off-diagonal epistemic
        # covariance between output variables.
        full_covariance_output = self.aggregate_full_covariance(
            member_means=member_means,
            member_log_variances=member_log_variances,
        )
    
        total_covariance = full_covariance_output["covariance"]
    
        samples: torch.Tensor | None = None
    
        if sampling_method == "analytical_mean":
            prediction = analytical_output["mean"]
    
        elif sampling_method == "moment_matched_sample_mean":
            distribution = self.make_moment_matched_distribution(
                member_means=member_means,
                member_log_variances=member_log_variances,
                jitter=covariance_jitter,
            )
    
            samples = distribution.sample(
                sample_shape=torch.Size([sample_count])
            )
    
            prediction = samples.mean(dim=0)
    
        else:
            # These methods use samples from the original ensemble Gaussian
            # mixture rather than the moment-matched Gaussian.
            samples = self.sample_from_member_parameters(
                member_means=member_means,
                member_log_variances=member_log_variances,
                sample_count=sample_count,
            )
    
            if sampling_method == "sample_mean":
                prediction = self.sample_mean(
                    samples
                )
    
            elif sampling_method == "marginal_median":
                prediction = self.marginal_sample_median(
                    samples
                )
    
            elif sampling_method == "highest_density":
                prediction = self.highest_density_sample(
                    samples=samples,
                    member_means=member_means,
                    member_log_variances=member_log_variances,
                )
    
            elif sampling_method == "top_k_density_mean":
                # top_k has already been checked above.
                if top_k is None:
                    raise RuntimeError(
                        "top_k passed validation but remained None."
                    )
    
                prediction = self.top_k_density_mean(
                    samples=samples,
                    member_means=member_means,
                    member_log_variances=member_log_variances,
                    top_k=top_k,
                )
    
            else:
                raise RuntimeError(
                    "Sampling method passed validation but was not handled."
                )
    
        return {
            **analytical_output,
            "prediction": prediction,
            "covariance": total_covariance,
            "samples": samples if return_samples else None,
        }

    def member_parameters(
        self,
        inputs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """
        Return member-level Gaussian distribution parameters.

        Parameters
        ----------
        inputs:
            Input tensor with shape ``[batch_size, input_count]``.

        Returns
        -------
        member_means:
            Tensor with shape
            ``[member_count, batch_size, output_count]``.

        member_log_variances:
            Tensor with shape
            ``[member_count, batch_size, output_count]`` for
            probabilistic models, otherwise ``None``.

        member_injection_logits:
            Tensor with shape ``[member_count, batch_size, 1]`` if every
            member exposes an injection-classification head, otherwise
            ``None``.
        """
        outputs = self.forward_members(inputs)

        member_means = torch.stack(
            [output["mean"] for output in outputs],
            dim=0,
        )

        probabilistic_flags = [
            output["log_var"] is not None
            for output in outputs
        ]

        if any(probabilistic_flags) and not all(probabilistic_flags):
            raise RuntimeError(
                "All ensemble members must consistently return either "
                "log variances or None."
            )

        member_log_variances = (
            torch.stack(
                [output["log_var"] for output in outputs],
                dim=0,
            )
            if any(probabilistic_flags)
            else None
        )

        injection_flags = [
            output.get("injection_logit") is not None
            for output in outputs
        ]

        if any(injection_flags) and not all(injection_flags):
            raise RuntimeError(
                "All ensemble members must consistently return either "
                "injection logits or None."
            )

        member_injection_logits = (
            torch.stack(
                [output["injection_logit"] for output in outputs],
                dim=0,
            )
            if any(injection_flags)
            else None
        )

        return member_means, member_log_variances, member_injection_logits



    @staticmethod
    def aggregate_member_parameters(
        member_means: torch.Tensor,
        member_log_variances: torch.Tensor | None,
    ) -> dict[str, torch.Tensor]:
        """
        Calculate predictive moments of the ensemble mixture.

        The total predictive variance is the sum of:

        1. Aleatoric variance, representing predicted data noise.
        2. Epistemic variance, representing disagreement between members.
        """
        ensemble_mean = member_means.mean(dim=0)

        epistemic_variance = member_means.var(
            dim=0,
            unbiased=False,
        )

        if member_log_variances is not None:
            aleatoric_variance = (
                member_log_variances.exp().mean(dim=0)
            )

            total_variance = (
                aleatoric_variance
                + epistemic_variance
            )
        else:
            aleatoric_variance = torch.zeros_like(
                epistemic_variance
            )

            total_variance = epistemic_variance

        return {
            "mean": ensemble_mean,
            "log_var": total_variance.clamp_min(1.0e-12).log(),
            "epistemic_var": epistemic_variance,
            "aleatoric_var": aleatoric_variance,
        }


    @staticmethod
    def aggregate_full_covariance(
        member_means: torch.Tensor,
        member_log_variances: torch.Tensor | None,
    ) -> dict[str, torch.Tensor]:
        """
        Calculate the ensemble mean and full total covariance matrix.
    
        Parameters
        ----------
        member_means:
            Ensemble-member means with shape:
    
            [member_count, batch_size, output_count]
    
        member_log_variances:
            Ensemble-member marginal log variances with shape:
    
            [member_count, batch_size, output_count]
    
            If None, only epistemic covariance is included.
    
        Returns
        -------
        dict[str, torch.Tensor]
            Dictionary containing:
    
            mean:
                Ensemble mean with shape [batch_size, output_count].
    
            covariance:
                Total covariance with shape
                [batch_size, output_count, output_count].
    
            aleatoric_covariance:
                Mean within-member covariance.
    
            epistemic_covariance:
                Covariance of ensemble-member means.
        """
        if member_means.ndim != 3:
            raise ValueError(
                "member_means must have shape "
                "[member_count, batch_size, output_count]."
            )
    
        if (
            member_log_variances is not None
            and member_log_variances.shape != member_means.shape
        ):
            raise ValueError(
                "member_log_variances must have the same shape as "
                "member_means."
            )
    
        member_count = member_means.shape[0]
    
        if member_count < 1:
            raise ValueError(
                "At least one ensemble member is required."
            )
    
        ensemble_mean = member_means.mean(dim=0)
    
        centred_means = (
            member_means
            - ensemble_mean.unsqueeze(0)
        )
    
        # [M, B, O] and [M, B, P] -> [B, O, P]
        #
        # Dividing by M, rather than M - 1, gives the population covariance
        # of the equally weighted ensemble distribution.
        epistemic_covariance = torch.einsum(
            "mbo,mbp->bop",
            centred_means,
            centred_means,
        ) / float(member_count)
    
        if member_log_variances is None:
            aleatoric_covariance = torch.zeros_like(
                epistemic_covariance
            )
        else:
            # Each member predicts marginal variances only. Therefore, its
            # within-member covariance matrix is diagonal.
            mean_aleatoric_variance = (
                member_log_variances.exp().mean(dim=0)
            )
    
            aleatoric_covariance = torch.diag_embed(
                mean_aleatoric_variance
            )
    
        total_covariance = (
            aleatoric_covariance
            + epistemic_covariance
        )
    
        return {
            "mean": ensemble_mean,
            "covariance": total_covariance,
            "aleatoric_covariance": aleatoric_covariance,
            "epistemic_covariance": epistemic_covariance,
        }

    @staticmethod
    def aggregate_injection_logits(
        member_injection_logits: torch.Tensor | None,
    ) -> dict[str, torch.Tensor] | None:
        """
        Combine per-member injection logits into a mixture probability.

        The ensemble represents an equally weighted mixture, so the
        predictive probability is the average of each member's
        probability (not the sigmoid of the averaged logit).

        Parameters
        ----------
        member_injection_logits:
            Tensor with shape ``[member_count, batch_size, 1]``, or
            ``None`` if the architecture has no injection head.

        Returns
        -------
        dict[str, torch.Tensor] or None
            ``probability``: mixture probability, shape ``[batch, 1]``.
            ``epistemic_std``: standard deviation of member
            probabilities, shape ``[batch, 1]``.
        """
        if member_injection_logits is None:
            return None

        member_probabilities = torch.sigmoid(member_injection_logits)

        return {
            "probability": member_probabilities.mean(dim=0),
            "epistemic_std": member_probabilities.std(dim=0, unbiased=False),
        }
        
    @staticmethod
    def make_moment_matched_distribution(
        member_means: torch.Tensor,
        member_log_variances: torch.Tensor | None,
        jitter: float = 1.0e-6,
    ) -> torch.distributions.MultivariateNormal:
        """
        Construct a moment-matched multivariate Gaussian.
    
        The distribution has the same mean and full covariance as the
        ensemble mixture, but it does not preserve mixture multimodality.
    
        Parameters
        ----------
        member_means:
            Tensor with shape
            [member_count, batch_size, output_count].
    
        member_log_variances:
            Tensor with shape
            [member_count, batch_size, output_count], or None.
    
        jitter:
            Positive diagonal value added for numerical stability.
    
        Returns
        -------
        torch.distributions.MultivariateNormal
            A batched multivariate Gaussian distribution. Its batch shape is
            [batch_size] and its event shape is [output_count].
        """
        if jitter <= 0.0:
            raise ValueError(
                f"jitter must be positive, got {jitter}."
            )
    
        moments = DeepEnsemble.aggregate_full_covariance(
            member_means=member_means,
            member_log_variances=member_log_variances,
        )
    
        ensemble_mean = moments["mean"]
        total_covariance = moments["covariance"]
    
        output_count = ensemble_mean.shape[-1]
    
        identity = torch.eye(
            output_count,
            device=ensemble_mean.device,
            dtype=ensemble_mean.dtype,
        ).unsqueeze(0)
    
        stable_covariance = (
            total_covariance
            + jitter * identity
        )
    
        return torch.distributions.MultivariateNormal(
            loc=ensemble_mean,
            covariance_matrix=stable_covariance,
        )
    
    def sample_moment_matched_gaussian(
        self,
        inputs: torch.Tensor,
        sample_count: int = 1000,
        jitter: float = 1.0e-6,
    ) -> torch.Tensor:
        """
        Sample from a multivariate Gaussian with ensemble moments.
    
        Returns
        -------
        torch.Tensor
            Samples with shape:
    
            [sample_count, batch_size, output_count]
        """
        if sample_count < 1:
            raise ValueError(
                "sample_count must be at least 1, "
                f"got {sample_count}."
            )
    
        member_means, member_log_variances, _ = (
            self.member_parameters(inputs)
        )

        distribution = self.make_moment_matched_distribution(
            member_means=member_means,
            member_log_variances=member_log_variances,
            jitter=jitter,
        )
    
        return distribution.sample(
            sample_shape=torch.Size([sample_count])
        )
        
    @staticmethod
    def sample_from_member_parameters(
        member_means: torch.Tensor,
        member_log_variances: torch.Tensor | None,
        sample_count: int = 1000,
    ) -> torch.Tensor:
        """
        Sample from the deep-ensemble predictive mixture.

        One ensemble member is selected for each complete output vector.
        This preserves cross-output dependence caused by ensemble-member
        disagreement.

        Parameters
        ----------
        member_means:
            Tensor with shape
            ``[member_count, batch_size, output_count]``.

        member_log_variances:
            Optional tensor with the same shape as ``member_means``.

        sample_count:
            Number of Monte Carlo samples for each input row.

        Returns
        -------
        torch.Tensor
            Samples with shape
            ``[sample_count, batch_size, output_count]``.
        """
        if sample_count < 1:
            raise ValueError(
                f"sample_count must be at least 1, got {sample_count}."
            )

        if member_means.ndim != 3:
            raise ValueError(
                "member_means must have shape "
                "[member_count, batch_size, output_count]."
            )

        member_count, batch_size, output_count = (
            member_means.shape
        )

        member_indices = torch.randint(
            low=0,
            high=member_count,
            size=(sample_count, batch_size),
            device=member_means.device,
        )

        batch_indices = torch.arange(
            batch_size,
            device=member_means.device,
        ).unsqueeze(0)

        selected_means = member_means[
            member_indices,
            batch_indices,
        ]

        if member_log_variances is None:
            return selected_means

        selected_log_variances = member_log_variances[
            member_indices,
            batch_indices,
        ]

        noise = torch.randn(
            sample_count,
            batch_size,
            output_count,
            device=selected_means.device,
            dtype=selected_means.dtype,
        )

        selected_standard_deviations = torch.exp(
            0.5 * selected_log_variances
        )

        return (
            selected_means
            + selected_standard_deviations * noise
        )

    @staticmethod
    def sample_mean(
        samples: torch.Tensor,
    ) -> torch.Tensor:
        """
        Calculate the mean of the predictive samples.

        This is a Monte Carlo approximation of the analytical ensemble mean.
        """
        return samples.mean(dim=0)

    @staticmethod
    def marginal_sample_median(
        samples: torch.Tensor,
    ) -> torch.Tensor:
        """
        Calculate the median independently for each output variable.

        Quantile interpolation is used when the sample count is even. The
        resulting output vector does not necessarily correspond to one
        complete sampled vector.
        """
        return torch.quantile(
            samples,
            q=0.5,
            dim=0,
        )
        
    @staticmethod
    def predictive_log_probability(
        samples: torch.Tensor,
        member_means: torch.Tensor,
        member_log_variances: torch.Tensor | None,
    ) -> torch.Tensor:
        """
        Calculate sample log probabilities under the ensemble mixture.

        Density is calculated jointly across all output dimensions.

        Parameters
        ----------
        samples:
            Samples with shape
            ``[sample_count, batch_size, output_count]``.

        member_means:
            Tensor with shape
            ``[member_count, batch_size, output_count]``.

        member_log_variances:
            Tensor with the same shape as ``member_means``.

        Returns
        -------
        torch.Tensor
            Joint log probabilities with shape
            ``[sample_count, batch_size]``.
        """
        if member_log_variances is None:
            raise RuntimeError(
                "Density-based sampling methods require probabilistic "
                "ensemble members with predicted log variances."
            )

        if samples.ndim != 3:
            raise ValueError(
                "samples must have shape "
                "[sample_count, batch_size, output_count]."
            )

        if member_means.ndim != 3:
            raise ValueError(
                "member_means must have shape "
                "[member_count, batch_size, output_count]."
            )

        if samples.shape[1:] != member_means.shape[1:]:
            raise ValueError(
                "The samples and member parameters must have matching "
                "batch and output dimensions."
            )

        # Resulting shape: [sample_count, member_count, batch, output]
        differences = (
            samples.unsqueeze(1)
            - member_means.unsqueeze(0)
        )

        expanded_log_variances = (
            member_log_variances.unsqueeze(0)
        )

        component_log_probabilities = -0.5 * (
            math.log(2.0 * math.pi)
            + expanded_log_variances
            + differences.square()
            / expanded_log_variances.exp()
        ).sum(dim=-1)

        # Average the member probabilities in log space:
        #
        # log p(y | x)
        # = log[(1 / M) sum_m p_m(y | x)]
        member_count = member_means.shape[0]

        return (
            torch.logsumexp(
                component_log_probabilities,
                dim=1,
            )
            - math.log(member_count)
        )



    @classmethod
    def highest_density_sample(
        cls,
        samples: torch.Tensor,
        member_means: torch.Tensor,
        member_log_variances: torch.Tensor | None,
    ) -> torch.Tensor:
        """
        Select the sampled output vector with the highest mixture density.

        This is a sample-based approximation to the mode of the predictive
        mixture distribution.
        """
        log_probabilities = cls.predictive_log_probability(
            samples=samples,
            member_means=member_means,
            member_log_variances=member_log_variances,
        )

        best_indices = log_probabilities.argmax(dim=0)

        batch_indices = torch.arange(
            samples.shape[1],
            device=samples.device,
        )

        return samples[
            best_indices,
            batch_indices,
        ]

    @classmethod
    def top_k_density_mean(
        cls,
        samples: torch.Tensor,
        member_means: torch.Tensor,
        member_log_variances: torch.Tensor | None,
        top_k: int,
    ) -> torch.Tensor:
        """
        Average the top-k samples with the highest mixture density.

        Complete output vectors are ranked by their joint density before
        being averaged.
        """
        sample_count = samples.shape[0]

        if top_k < 1:
            raise ValueError(
                f"top_k must be at least 1, got {top_k}."
            )

        if top_k > sample_count:
            raise ValueError(
                f"top_k={top_k} cannot exceed "
                f"sample_count={sample_count}."
            )

        log_probabilities = cls.predictive_log_probability(
            samples=samples,
            member_means=member_means,
            member_log_variances=member_log_variances,
        )

        top_indices = torch.topk(
            log_probabilities,
            k=top_k,
            dim=0,
            largest=True,
            sorted=False,
        ).indices

        gather_indices = top_indices.unsqueeze(-1).expand(
            -1,
            -1,
            samples.shape[-1],
        )

        selected_samples = torch.gather(
            samples,
            dim=0,
            index=gather_indices,
        )

        return selected_samples.mean(dim=0)

    @classmethod
    def top_k_density_means(
        cls,
        samples: torch.Tensor,
        member_means: torch.Tensor,
        member_log_variances: torch.Tensor | None,
        top_k_values: Sequence[int],
    ) -> dict[int, torch.Tensor]:
        """
        Calculate density means for several values of k efficiently.

        All values of k use the same predictive samples and density ranking.
        This method is intended for validation experiments.
        """
        if not top_k_values:
            raise ValueError(
                "top_k_values must contain at least one value."
            )

        sample_count = samples.shape[0]
        unique_values = sorted(set(top_k_values))

        for top_k in unique_values:
            if top_k < 1 or top_k > sample_count:
                raise ValueError(
                    "Every top-k value must be between 1 and "
                    f"{sample_count}; got {top_k}."
                )

        log_probabilities = cls.predictive_log_probability(
            samples=samples,
            member_means=member_means,
            member_log_variances=member_log_variances,
        )

        maximum_top_k = max(unique_values)

        ranked_indices = torch.topk(
            log_probabilities,
            k=maximum_top_k,
            dim=0,
            largest=True,
            sorted=True,
        ).indices

        gather_indices = ranked_indices.unsqueeze(-1).expand(
            -1,
            -1,
            samples.shape[-1],
        )

        ranked_samples = torch.gather(
            samples,
            dim=0,
            index=gather_indices,
        )

        cumulative_sum = ranked_samples.cumsum(dim=0)

        return {
            top_k: cumulative_sum[top_k - 1] / float(top_k)
            for top_k in unique_values
        }





def build_single_model(configuration, input_count: int, output_names: list[str]):
    """Build one configured surrogate architecture without Hydra instantiate."""
    architecture = configuration.architecture
    output_count = len(output_names)

    if (
        getattr(configuration, "injection_head", False)
        and architecture != "multi_output"
    ):
        raise NotImplementedError(
            "model.injection_head is only implemented for the "
            "'multi_output' architecture."
        )

    if architecture == "multi_output":
        return MultiOutputNetwork(input_count, output_count, configuration)
    if architecture == "independent":
        return IndependentNetworks(input_count, output_count, configuration)
    if architecture == "shared_branches":
        return SharedBranchNetwork(input_count, output_count, configuration)
    if architecture == "grouped":
        return GroupedNetwork(input_count, output_names, configuration)
    raise ValueError(f"Unsupported architecture '{architecture}'.")


def build_model(configuration, input_count: int, output_names: list[str]):
    """Build a deep ensemble of independently initialised surrogate models."""
    members = [
        build_single_model(configuration, input_count, output_names)
        for _ in range(configuration.ensemble_size)
    ]
    return DeepEnsemble(members)
