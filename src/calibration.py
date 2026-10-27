from __future__ import annotations

from statistics import NormalDist
from typing import Sequence

import numpy as np
import pandas as pd

EPSILON = 1.0e-12


def _masked_columns(
    df: pd.DataFrame,
    name: str,
    *columns: str,
) -> list[np.ndarray]:
    """Extract named columns for one output, dropping invalid rows."""

    arrays = [df[column].values for column in columns]

    valid_col = f"{name}_valid"
    if valid_col in df.columns:
        mask = df[valid_col].values > 0.5
        arrays = [array[mask] for array in arrays]

    return arrays


class StdCalibrator:
    """
    Per-output multiplicative calibration of predictive standard deviations.

    Fits one scalar per output on validation predictions, minimizing the
    Gaussian NLL: s^2 = mean(((y - mean) / std)^2). Applying it leaves point
    predictions (MAE, R2) unchanged; only the predictive std is rescaled.

    Works on any dataframe with columns `{name}`, `{name}_pred` and
    `{name}_std` per output. Rows can be excluded per output via an
    optional `{name}_valid` column.
    """

    def __init__(
        self,
        output_names: Sequence[str],
        epsilon: float = EPSILON,
    ):
        self.output_names = list(output_names)
        self.epsilon = epsilon
        self.scales: np.ndarray | None = None

    def fit(self, df: pd.DataFrame) -> "StdCalibrator":

        scales = []

        for name in self.output_names:

            y, pred, std = _masked_columns(
                df, name, name, f"{name}_pred", f"{name}_std",
            )
            std = np.clip(std, self.epsilon, None)

            z = (y - pred) / std
            scales.append(float(np.sqrt(np.mean(z**2))))

        self.scales = np.asarray(scales, dtype=np.float64)

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:

        if self.scales is None:
            raise RuntimeError(
                "StdCalibrator is not fitted. Call fit() first."
            )

        result = df.copy()

        for name, scale in zip(self.output_names, self.scales):
            result[f"{name}_std"] = result[f"{name}_std"] * scale

        return result

    def fit_transform(
        self,
        df_fit: pd.DataFrame,
        df_apply: pd.DataFrame,
    ) -> pd.DataFrame:
        return self.fit(df_fit).transform(df_apply)


class VarianceScalingCalibrator:
    """
    Per-output power-law calibration of predictive standard deviations.

    Fits `std_calibrated = c * std_raw ** p` per output on validation
    predictions by minimizing the Gaussian NLL. `StdCalibrator` is the
    special case `p = 1`: a single scalar assumes the ensemble is off by
    the same multiplicative factor at every predicted uncertainty level.
    A non-unit `p` corrects for heteroscedastic miscalibration instead,
    where the degree of over/under-confidence itself depends on the
    predicted std - e.g. well-calibrated for confident predictions but
    overconfident for uncertain ones.

    For a fixed `p`, the NLL-minimizing `c` has a closed form
    (`c(p) = sqrt(mean(error^2 / std^(2p)))`), so fitting reduces to a
    1-D grid search over `p` with `c` recovered in closed form at each
    step - no extra optimisation dependency needed.

    `power_bounds` is deliberately tight (default `(0.5, 1.5)`, not the
    full real line): `std_calibrated` is `power`-th-order sensitive to
    `std_raw`, so a fold-level fit with `power` far from 1 is extremely
    sensitive to any test-time std falling outside the validation range
    it was fit on - e.g. for a target whose raw std spans a 10x range
    across ~500 validation rows, an unconstrained fit can land on
    `power~=2` (fitting the few extreme-std rows), which then collapses
    or explodes the calibrated std for any out-of-range test row.
    `power_shrinkage` adds an `L2` penalty on `(power - 1)`, weighted by
    `power_shrinkage / n_samples`, to the per-sample-mean NLL objective -
    i.e. it acts like `power_shrinkage` pseudo-observations pulling
    toward the robust scalar case (`StdCalibrator`). Scaling by
    `1 / n_samples` keeps the penalty's influence roughly fixed in
    absolute (not per-sample) terms: it dominates when a fold's
    validation set is small, and fades out as genuine support for a
    non-unit power accumulates.

    Like `StdCalibrator` (and unlike `IsotonicCalibrator`), the result is
    still a single calibrated std: applying it leaves point predictions
    (MAE, R2) unchanged and keeps `{name}_std` representable as a
    symmetric Gaussian interval.

    Works on any dataframe with columns `{name}`, `{name}_pred` and
    `{name}_std` per output. Rows can be excluded per output via an
    optional `{name}_valid` column.
    """

    def __init__(
        self,
        output_names: Sequence[str],
        epsilon: float = EPSILON,
        power_bounds: tuple[float, float] = (0.5, 1.5),
        power_grid_size: int = 241,
        power_shrinkage: float = 300.0,
    ):
        self.output_names = list(output_names)
        self.epsilon = epsilon
        self.power_bounds = power_bounds
        self.power_grid_size = power_grid_size
        self.power_shrinkage = power_shrinkage
        self.scales: np.ndarray | None = None
        self.powers: np.ndarray | None = None

    def fit(self, df: pd.DataFrame) -> "VarianceScalingCalibrator":

        scales = []
        powers = []

        power_grid = np.linspace(*self.power_bounds, self.power_grid_size)

        for name in self.output_names:

            y, pred, std = _masked_columns(
                df, name, name, f"{name}_pred", f"{name}_std",
            )
            std = np.clip(std, self.epsilon, None)
            error = y - pred
            shrinkage_weight = self.power_shrinkage / len(error)

            best_power = 1.0
            best_scale = float(np.sqrt(np.mean((error / std) ** 2)))
            best_nll = np.inf

            for power in power_grid:
                scale = float(
                    np.sqrt(np.mean((error / np.power(std, power)) ** 2))
                )
                scale = max(scale, self.epsilon)

                calibrated_std = scale * np.power(std, power)
                nll = np.mean(
                    np.log(calibrated_std)
                    + 0.5 * (error / calibrated_std) ** 2
                ) + shrinkage_weight * (power - 1.0) ** 2

                if nll < best_nll:
                    best_nll = nll
                    best_power = float(power)
                    best_scale = scale

            scales.append(best_scale)
            powers.append(best_power)

        self.scales = np.asarray(scales, dtype=np.float64)
        self.powers = np.asarray(powers, dtype=np.float64)

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:

        if self.scales is None:
            raise RuntimeError(
                "VarianceScalingCalibrator is not fitted. Call fit() first."
            )

        result = df.copy()

        for name, scale, power in zip(
            self.output_names, self.scales, self.powers
        ):
            std = np.clip(result[f"{name}_std"].values, self.epsilon, None)
            result[f"{name}_std"] = scale * np.power(std, power)

        return result

    def fit_transform(
        self,
        df_fit: pd.DataFrame,
        df_apply: pd.DataFrame,
    ) -> pd.DataFrame:
        return self.fit(df_fit).transform(df_apply)


class IsotonicCalibrator:
    """
    Nonparametric regression recalibration (Kuleshov et al., 2018).

    Fits the empirical distribution of PIT values
    ``p_i = Phi((y_i - mean_i) / std_i)``
    on validation predictions, per output. A central interval at
    confidence `c` is built from the *empirical* quantiles of the fitted
    `p_i` at `(1-c)/2` and `(1+c)/2`, rather than the Gaussian's own
    quantiles. This is the closed-form inverse of Kuleshov's isotonic
    regression calibrator: fitting isotonic regression on
    `(p_i, empirical_cdf(p_i))` recovers the empirical CDF itself, and
    inverting it to build an interval reduces exactly to the quantile
    function of the fitted `p_i`.

    Unlike `StdCalibrator` (a single multiplicative scale, equivalent to
    the regression analogue of temperature scaling), this corrects for
    skew / heavy tails in the residual distribution at every confidence
    level simultaneously - at the cost of no longer being representable
    as a single calibrated std: intervals must be queried per
    confidence level via `interval()`.

    Works on any dataframe with columns `{name}`, `{name}_pred` and
    `{name}_std` per output. Rows can be excluded per output via an
    optional `{name}_valid` column.
    """

    def __init__(
        self,
        output_names: Sequence[str],
        epsilon: float = EPSILON,
    ):
        self.output_names = list(output_names)
        self.epsilon = epsilon
        self._pit: dict[str, np.ndarray] = {}

    def fit(self, df: pd.DataFrame) -> "IsotonicCalibrator":

        normal = NormalDist()

        for name in self.output_names:

            y, pred, std = _masked_columns(
                df, name, name, f"{name}_pred", f"{name}_std",
            )
            std = np.clip(std, self.epsilon, None)

            z = (y - pred) / std
            pit = np.array([normal.cdf(zi) for zi in z])

            self._pit[name] = np.sort(pit)

        return self

    def interval(
        self,
        name: str,
        mean: np.ndarray,
        std: np.ndarray,
        confidence: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Calibrated central interval at the given confidence level."""

        if name not in self._pit:
            raise RuntimeError(
                f"IsotonicCalibrator has no fitted PIT values for "
                f"'{name}'. Call fit() first."
            )

        normal = NormalDist()
        pit = self._pit[name]

        q_lo = np.quantile(pit, (1.0 - confidence) / 2.0)
        q_hi = np.quantile(pit, (1.0 + confidence) / 2.0)

        z_lo = normal.inv_cdf(np.clip(q_lo, self.epsilon, 1.0 - self.epsilon))
        z_hi = normal.inv_cdf(np.clip(q_hi, self.epsilon, 1.0 - self.epsilon))

        return mean + z_lo * std, mean + z_hi * std

    def coverage(
        self,
        name: str,
        target: np.ndarray,
        mean: np.ndarray,
        std: np.ndarray,
        confidence: float,
    ) -> np.ndarray:

        lower, upper = self.interval(name, mean, std, confidence)

        return (target >= lower) & (target <= upper)

    def ece(
        self,
        df: pd.DataFrame,
        confidence_levels: Sequence[float] | None = None,
    ) -> dict[str, float]:
        """Expected calibration error of the calibrated intervals on `df`."""

        if confidence_levels is None:
            confidence_levels = np.linspace(0.05, 0.95, 19)

        confidence_levels = np.asarray(confidence_levels, dtype=np.float64)

        results = {}

        for name in self.output_names:

            y, mean, std = _masked_columns(
                df, name, name, f"{name}_pred", f"{name}_std",
            )
            std = np.clip(std, self.epsilon, None)

            empirical_coverage = np.array([
                self.coverage(name, y, mean, std, c).mean()
                for c in confidence_levels
            ])

            results[name] = float(
                np.mean(np.abs(empirical_coverage - confidence_levels))
            )

        return results
