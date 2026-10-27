
import sys
from statistics import NormalDist

import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from pathlib import Path

import torch

from calibration import StdCalibrator
from dataset.transforms import TargetTransform
from dataset.scalers import make_scaler
from dataset.datamodule import canonical_name
from model.model import build_model

class Predictor:

    def __init__(
        self,
        model,
        device,
        input_names,
        output_names,
        input_scaler,
        output_scaler,
        target_transform = None,
        cfg = None,
    ):
        self.model = model
        self.device = device

        self.input_names = input_names
        self.output_names = output_names

        self.input_scaler = input_scaler
        self.output_scaler = output_scaler
        self.target_transform = target_transform
        self.cfg = cfg
        
        # Per-output power-law std calibration: std_calibrated = std_scale *
        # std_raw ** std_power. Fitted via fit_std_calibration() (power
        # fixed at 1, i.e. a pure scalar). Identity by default.
        self.std_scale = torch.ones(len(output_names))
        self.std_power = torch.ones(len(output_names))

    def fit_std_calibration(
        self,
        df_val: pd.DataFrame,
    ):
        calibrator = StdCalibrator(self.output_names).fit(df_val)

        self.std_scale = torch.tensor(
            calibrator.scales,
            dtype=torch.float32,
        )

        return self.std_scale

    def inverse_targets(
        self,
        values,
    ):

        values = (
            self.output_scaler
            .inverse_transform(
                values
            )
        )
        if self.target_transform is not None:
            values = self.target_transform.inverse_torch(values)

        return values
            
    @torch.no_grad()
    def predict(
        self,
        x: np.ndarray,
    ):
        self.model.eval()
    
        x = torch.as_tensor(
            x,
            dtype=torch.float32,
            device=self.device,
        )
    
        x = self.input_scaler.transform(x)
    
        output = self.model.predict(
            inputs=x,
        )

        output['pred'] = self.inverse_targets(output['mean'])
        
        # The reported uncertainty remains the analytical ensemble uncertainty.
        std_scaled = torch.sqrt(
            output['log_var'].exp().clamp_min(1.0e-12)
        )
    
        # # The reported uncertainty is the aleatoric (data-noise) component
        # # only, excluding epistemic (ensemble-disagreement) variance.
        # std_scaled = torch.sqrt(
        #     output['epistemic'].clamp_min(1.0e-12)
        # )

        upper_physical = self.inverse_targets(
            output['mean'] + std_scaled
        )
    
        lower_physical = self.inverse_targets(
            output['mean'] - std_scaled
        )
    
        raw_std = (
            (upper_physical - lower_physical)
            .abs()
            .div(2.0)
            .clamp_min(1.0e-12)
        )

        output['std'] = (
            self.std_scale.to(self.device)
            * raw_std.pow(self.std_power.to(self.device))
        )

        return output

    @torch.no_grad()
    def predict_from_df(
        self,
        df: pd.DataFrame,
    ):
        x = df[
            self.input_names
        ].values
    
        output = self.predict(x)
    
        result = df.copy()
    
        for i, name in enumerate(self.output_names):
            result[f"{name}_pred"] = output['pred'][:, i]
            result[f"{name}_std"] = output['std'][:, i]

        return result

    def predict_interval_from_df(
        self,
        df: pd.DataFrame,
        confidence: float = 0.9,
    ):
        """
        Calibrated central interval at the given confidence level.

        A Gaussian interval built from the (possibly calibrated)
        `{name}_std` column.
        """
        result = self.predict_from_df(df)

        z = NormalDist().inv_cdf((1.0 + confidence) / 2.0)

        for name in self.output_names:

            mean = result[f"{name}_pred"].values
            std = result[f"{name}_std"].values

            result[f"{name}_lower"] = mean - z * std
            result[f"{name}_upper"] = mean + z * std

        return result

    def predict_from_csv_file(
        self,
        file_path,
    ):
        df_raw = pd.read_csv(file_path)
        df_raw.rename(
            columns={c: canonical_name(c) for c in df_raw.columns},
            inplace=True
        )

        
        return self.predict_from_df(df_raw)

    @classmethod
    def from_experiment(
        cls,
        exp_dir,
        config_name = 'resolved_config.yaml',
        checkpoint="checkpoints/best.pt",
        device="cpu",
        calibrate_std=True,
        calibration_method="scalar",
        calibration_source="best_val_predictions.csv",
    ):
        exp_dir = Path(exp_dir)

        cfg = OmegaConf.load( exp_dir / config_name)

        checkpoint_path = (
            exp_dir 
            / checkpoint
        )
    
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False,
        )
        
        input_names = checkpoint['input_names']
    
        output_names = checkpoint['output_names']
        
        model = build_model(
            cfg.model,
            len(input_names),
            output_names,
        )

        model.load_state_dict(
            checkpoint["model_state"], 
            strict=True
        )

        input_scaler = checkpoint["input_scaler"]
        output_scaler = checkpoint["output_scaler"]
        target_transform = checkpoint["target_transform"]

        predictor = cls(
            model=model,
            device=device,
            input_names=input_names,
            output_names=output_names,
            input_scaler=input_scaler,
            output_scaler=output_scaler,
            target_transform=target_transform,
            cfg=cfg,
        )

        if calibrate_std:

            if calibration_method != "scalar":
                raise ValueError(
                    f"Unknown calibration_method: {calibration_method!r}. "
                    "Expected 'scalar'."
                )

            try:
                df_val = pd.read_csv(exp_dir / calibration_source)
            except Exception:
                print(f"Unable to read: {exp_dir / calibration_source} for calibration")
                print(f"Make sure file {calibration_source} exists or use correct prediction file for calibration.")
                df_val = None

            if df_val is not None:

                scales = predictor.fit_std_calibration(df_val)
                print(
                    "Fitted std calibration on validation set: "
                    + ", ".join(
                        f"{n}={s:.3f}"
                        for n, s in zip(output_names, scales.tolist())
                    )
                )

        return predictor