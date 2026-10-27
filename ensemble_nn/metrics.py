from __future__ import annotations

from statistics import NormalDist
from typing import Dict, Optional, Sequence

import torch

EPSILON = 1.0e-12



def _validate_pair(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Validate regression targets."""

    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"y_true shape {tuple(y_true.shape)} differs from "
            f"y_pred shape {tuple(y_pred.shape)}."
        )

    if y_true.ndim != 2:
        raise ValueError(
            f"Expected tensors of shape (B, n_outputs), "
            f"got {tuple(y_true.shape)}."
        )

    if y_true.numel() == 0:
        raise ValueError("Empty tensors are not allowed.")

    if not torch.isfinite(y_true).all():
        raise ValueError("y_true contains non-finite values.")

    if not torch.isfinite(y_pred).all():
        raise ValueError("y_pred contains non-finite values.")

    return y_true.float(), y_pred.float()


def _validate_standard_deviation(
    pred_std: torch.Tensor,
    expected_shape: torch.Size,
) -> torch.Tensor:
    """Validate predictive standard deviations."""

    if pred_std.shape != expected_shape:
        raise ValueError(
            f"pred_std shape {tuple(pred_std.shape)} must match "
            f"{tuple(expected_shape)}."
        )

    if not torch.isfinite(pred_std).all():
        raise ValueError("pred_std contains non-finite values.")

    if (pred_std < 0).any():
        raise ValueError("pred_std contains negative values.")

    return pred_std.clamp_min(EPSILON)


# =====================================================================
# Element-wise metrics
# =====================================================================


class MAE:
    """Element-wise absolute error."""

    def __call__(
        self,
        y_true: torch.Tensor,
        y_pred: torch.Tensor,
    ) -> torch.Tensor:
        return torch.abs(y_pred - y_true)


class MSE:
    """Element-wise squared error."""

    def __call__(
        self,
        y_true: torch.Tensor,
        y_pred: torch.Tensor,
    ) -> torch.Tensor:
        return (y_pred - y_true) ** 2


class RMSE:
    """
    Element-wise squared error.

    Note:
        True RMSE only emerges after aggregation: the squared errors
        must be averaged over the batch and then square-rooted. The
        square root is applied in `aggregate_metrics`, not here -
        taking it per-element would make this identical to MAE.
    """

    def __call__(
        self,
        y_true: torch.Tensor,
        y_pred: torch.Tensor,
    ) -> torch.Tensor:
        return (y_pred - y_true) ** 2


class VRMSE:
    """Element-wise variance-normalised RMSE."""

    def __init__(self, epsilon: float = EPSILON):
        self.epsilon = epsilon

    def __call__(
        self,
        y_true: torch.Tensor,
        y_pred: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        if mask is None:
            variance = torch.var(
                y_true,
                dim=0,
                unbiased=False,
                keepdim=True,
            )
        else:
            counts = mask.sum(dim=0, keepdim=True).clamp_min(1.0)
            mean = (y_true * mask).sum(dim=0, keepdim=True) / counts
            variance = (
                ((y_true - mean) ** 2) * mask
            ).sum(dim=0, keepdim=True) / counts

        return torch.sqrt(
            (y_pred - y_true) ** 2
            / variance.clamp_min(self.epsilon)
        )


class NLL:
    """Element-wise Gaussian negative log-likelihood."""

    requires_std = True

    def __init__(self, epsilon: float = EPSILON):
        self.epsilon = epsilon

    def __call__(
        self,
        y_true: torch.Tensor,
        y_pred: torch.Tensor,
        pred_std: torch.Tensor,
    ) -> torch.Tensor:

        pred_std = pred_std.clamp_min(self.epsilon)

        squared_error = (y_pred - y_true) ** 2

        return 0.5 * (
            torch.log(2.0 * torch.pi * pred_std**2)
            + squared_error / pred_std**2
        )


class PredictionIntervalCoverage:
    """Measures the exact percentage of ground truth observations 
    that fall within predicted lower and upper bounds.
    """

    requires_std = True

    def __init__(
        self,
        confidence: float = 0.95,
    ):
        self.confidence = confidence

    def __call__(
        self,
        y_true: torch.Tensor,
        y_pred: torch.Tensor,
        pred_std: torch.Tensor,
    ) -> torch.Tensor:

        z_score = NormalDist().inv_cdf(
            0.5 + self.confidence / 2.0
        )

        return (
            torch.abs(y_pred - y_true)
            <= z_score * pred_std
        ).float()


# =====================================================================
# Batch-level metrics
# =====================================================================


class ECE:
    """
    Per-output expected calibration error (ECE) for predictive intervals.

    For a range of central confidence levels, compares the nominal
    confidence against the empirical coverage of the corresponding
    Gaussian interval, and averages the absolute gap across levels.
    """

    requires_std = True

    def __init__(
        self,
        confidence_levels: Optional[Sequence[float]] = None,
        n_bins=19,
        epsilon: float = EPSILON,
    ):
        if confidence_levels is None:
            confidence_levels = torch.linspace(0.05, 0.95, n_bins).tolist()

        self.confidence_levels = torch.tensor(
            confidence_levels,
            dtype=torch.float64,
        )

        # z value for each central confidence interval.
        self.z_values = torch.tensor(
            [
                NormalDist().inv_cdf((1.0 + float(c)) / 2.0)
                for c in self.confidence_levels
            ],
            dtype=torch.float64,
        )

        self.epsilon = epsilon

    def __call__(
        self,
        y_true: torch.Tensor,
        y_pred: torch.Tensor,
        pred_std: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        y_true = y_true.double()
        y_pred = y_pred.double()
        pred_std = pred_std.double().clamp_min(self.epsilon)

        if mask is None:
            mask = torch.ones_like(y_true)
        else:
            mask = mask.double()

        abs_error = torch.abs(y_true - y_pred)
        counts = mask.sum(dim=0).clamp_min(1.0)

        n_outputs = y_true.shape[1]
        ece = torch.empty(n_outputs, dtype=torch.float64)

        for output_idx in range(n_outputs):

            col_mask = mask[:, output_idx]
            col_error = abs_error[:, output_idx]
            col_std = pred_std[:, output_idx]
            col_count = counts[output_idx]

            empirical_coverage = torch.stack(
                [
                    (
                        (col_error <= z * col_std).double() * col_mask
                    ).sum() / col_count
                    for z in self.z_values
                ]
            )

            ece[output_idx] = torch.mean(
                torch.abs(empirical_coverage - self.confidence_levels)
            )

        return ece.float()


class R2:
    """Per-output R²."""

    def __init__(self, epsilon: float = EPSILON):
        self.epsilon = epsilon

    def __call__(
        self,
        y_true: torch.Tensor,
        y_pred: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        if mask is None:
            mask = torch.ones_like(y_true)

        counts = mask.sum(dim=0)

        residual_sum = torch.sum(
            ((y_true - y_pred) ** 2) * mask,
            dim=0,
        )

        target_mean = torch.sum(
            y_true * mask,
            dim=0,
            keepdim=True,
        ) / counts.clamp_min(1.0).unsqueeze(0)

        total_sum = torch.sum(
            ((y_true - target_mean) ** 2) * mask,
            dim=0,
        )

        regular = total_sum > self.epsilon

        r2 = torch.where(
            regular,
            1.0 - residual_sum / total_sum.clamp_min(self.epsilon),
            torch.where(
                residual_sum <= self.epsilon,
                torch.ones_like(total_sum),
                torch.zeros_like(total_sum),
            ),
        )

        # Columns with fewer than two valid samples have an undefined R².
        return torch.where(
            counts >= 2,
            r2,
            torch.full_like(r2, float("nan")),
        )

def aggregate_metrics(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    output_names: Sequence[str],
    pred_std: Optional[torch.Tensor] = None,
    metrics=None,
    coverage=None,
    mask: Optional[torch.Tensor] = None,
) -> Dict[str, float]:

    if metrics is None:
        metrics = validation_metrics()

    B, C = y_true.shape

    y_true, y_pred = _validate_pair(
        y_true,
        y_pred,
    )

    n_outputs = y_true.shape[1]

    if len(output_names) != n_outputs:
        raise ValueError(
            f"Expected {n_outputs} output names, "
            f"got {len(output_names)}."
        )

    if pred_std is not None:
        pred_std = _validate_standard_deviation(
            pred_std,
            y_true.shape,
        )

    if mask is not None:
        if mask.shape != y_true.shape:
            raise ValueError(
                f"mask shape {tuple(mask.shape)} must match "
                f"{tuple(y_true.shape)}."
            )
        mask = mask.float()
    else:
        mask = torch.ones_like(y_true)

    valid_counts = mask.sum(dim=0)

    results: Dict[str, float] = {}

    for metric in metrics:

        metric_name = metric.__class__.__name__

        if isinstance(metric, (R2, VRMSE)):
            values = metric(y_true, y_pred, mask=mask)

        elif getattr(metric, "requires_std", False):

            if pred_std is None:
                print(f"{metric_name} requires pred_std but is None. Skipping" )
                continue

            if isinstance(metric, ECE):
                values = metric(y_true, y_pred, pred_std, mask=mask)
            else:
                values = metric( y_true, y_pred, pred_std)

        else:
            values = metric(y_true, y_pred)

        # Reduce over batch with masking - not applicable to R2, which
        # already reduces internally using the mask.
        if values.ndim == 2:
            denominator = valid_counts.clamp_min(1.0)
            values = (values * mask).sum(dim=0) / denominator

        if isinstance(metric, RMSE):
            values = torch.sqrt(values)

        for output_name, value in zip(output_names, values):
            results[f"{metric_name}/{output_name}"] = value.item()

        has_valid = valid_counts > 0
        summary = (
            values[has_valid].mean().item()
            if bool(has_valid.any())
            else float("nan")
        )
        results[f"{metric_name}"] = summary
        results[f"{metric_name}/all"] = summary

    return results


def aggregrate_metrics_from_df(
    df, 
    target_columns, 
    prediction_columns, 
    metrics=None
):

    y_true = torch.tensor(
        df[target_columns].values,
        dtype=torch.float32,
    )

    y_pred = torch.tensor(
        df[prediction_columns].values,
        dtype=torch.float32,
    )

    pred_std = None

    std_columns = [f"{x}_std" for x in target_columns]

    if all(col in df.columns for col in std_columns):
        pred_std = torch.tensor(
            df[std_columns].values,
            dtype=torch.float32,
        )

    return aggregate_metrics(
        y_true=y_true,
        y_pred=y_pred,
        pred_std=pred_std,
        output_names=target_columns,
        metrics=metrics,
    )
    
def validation_metrics():
    return [
        MAE(),
        MSE(),
        RMSE(),
        VRMSE(),
        NLL(),
        R2(),
        ECE(),
    ]
    
if __name__ == "__main__":

    B = 8
    n_outputs = 4

    y_true = torch.rand(B, n_outputs)
    y_pred = torch.rand(B, n_outputs)
    pred_std = torch.rand(B, n_outputs)

    metrics = aggregate_metrics(
        y_true=y_true,
        y_pred=y_pred,
        pred_std=pred_std,
        output_names=["x", "y", "z", "w"],
        metrics=validation_metrics,
    )

    for k, v in metrics.items():
        print(f"{k}: {v:.6f}")