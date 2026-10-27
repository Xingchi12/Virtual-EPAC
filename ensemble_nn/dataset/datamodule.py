from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import torch

from torch.utils.data import DataLoader

from dataset.dataset import SurrogateDataset
from dataset.scalers import make_scaler
from dataset.transforms import TargetTransform

import re

def canonical_name(name: str) -> str:
    """
    Convert names to a consistent format:
    - lowercase
    - trim whitespace
    - replace non-alphanumeric characters with '_'
    - collapse multiple underscores
    - remove leading/trailing underscores
    """
    name = str(name).strip().lower()

    # Replace any non-alphanumeric character with underscore
    name = re.sub(r'[^a-z0-9]+', '_', name)

    # Collapse multiple underscores
    name = re.sub(r'_+', '_', name)

    # Remove leading/trailing underscores
    name = name.strip('_')

    return name

#: Laser parameters held fixed across the whole dataset.
FIXED_WAIST_UM = 16.0
FIXED_DURATION_FWHM_FS = 40.0

PHYSICS_FEATURE_NAMES = [
    "pf_plasma_wavelength",
    "pf_dephasing_length",
    "pf_depletion_length",
    "pf_plateau_over_dephasing",
    "pf_plateau_over_depletion",
    "pf_a0_proxy",
    "pf_chirped_duration",
    "pf_energy_gain_scaling",
    "pf_focus_over_plateau",
]


def physics_features(inputs, input_names):
    """
    Derive laser-wakefield scaling features from the raw controls.

    With only ~1100 usable rows for the electron energy, supplying the
    known nonlinear combinations directly is cheaper than expecting the
    network to discover them. Every feature is a function of the five
    existing controls, so no new data is introduced.

    Order of the returned columns matches ``PHYSICS_FEATURE_NAMES``.
    """
    column = {name: inputs[:, i] for i, name in enumerate(input_names)}

    laser_energy = column["l_energy_j_req_sim"]
    focus_z = column["l_focus_z_mm_req_sim"]
    gdd = column["l_gdd_fs2_req_sim"]
    density = np.clip(column["g_density_req_sim"], 1.0e-6, None)
    plateau = column["g_plateau_mm_req_sim"]

    # Plasma wavelength ~ n_e^-1/2; dephasing ~ n_e^-3/2; depletion ~ n_e^-1.
    plasma_wavelength = density ** -0.5
    dephasing_length = density ** -1.5
    depletion_length = density ** -1.0

    # How much of the accelerating structure the beam actually traverses.
    plateau_over_dephasing = plateau / dephasing_length
    plateau_over_depletion = plateau / depletion_length

    # Chirp stretches the pulse, lowering peak intensity. The duration is
    # fixed, so a0 ~ sqrt(E / tau_eff) up to constant waist factors.
    stretch = 1.0 + (
        gdd / (FIXED_DURATION_FWHM_FS ** 2)
    ) ** 2
    chirped_duration = FIXED_DURATION_FWHM_FS * np.sqrt(stretch)
    a0_proxy = np.sqrt(
        laser_energy / chirped_duration
    )

    # Dephasing-limited energy gain scales like a0^2 * L_dephasing.
    energy_gain_scaling = (a0_proxy ** 2) * dephasing_length

    focus_over_plateau = focus_z / np.clip(plateau, 1.0e-6, None)

    return np.column_stack(
        [
            plasma_wavelength,
            dephasing_length,
            depletion_length,
            plateau_over_dephasing,
            plateau_over_depletion,
            a0_proxy,
            chirped_duration,
            energy_gain_scaling,
            focus_over_plateau,
        ]
    )


@dataclass
class DatasetInfo:

    input_names: list[str]

    output_names: list[str]

    output_groups: Dict[str, list[str]]

    positive_outputs: set[str]

    n_inputs: int

    n_outputs: int

    n_train: int

    n_val: int

    n_test: int

    fold: int
    
class DataModule:

    def __init__(self, cfg):

        self.cfg = cfg

        self.raw_data = None

        self.input_names = []
        self.output_names = []

        self.group_indices = {}

        self.input_scaler = None
        self.output_scaler = None

        self.target_transform = None

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        self.train_dataloader = None
        self.val_dataloader = None
        self.test_dataloader = None

        self.info = None

        self.setup()

    @property
    def input_dim(self):
        return len(self.input_names)

    @property
    def output_dim(self):
        return len(self.output_names)

    @property
    def datasets(self):
        return {
            "train": self.train_dataset,
            "val": self.val_dataset,
            "test": self.test_dataset,
        }

    @property
    def dataloaders(self):
        return {
            "train": self.train_dataloader,
            "val": self.val_dataloader,
            "test": self.test_dataloader,
        }

    def setup(self):

        self.base_input_names = [
            canonical_name(c)
            for c in self.cfg.data.inputs
        ]

        self.use_physics_features = bool(
            self.cfg.data.get("physics_features", False)
        )

        self.input_names = (
            self.base_input_names + PHYSICS_FEATURE_NAMES
            if self.use_physics_features
            else list(self.base_input_names)
        )

        self.output_names = [
            canonical_name(c)
            for c in self.cfg.data.outputs
        ]
    
        self.raw_data = self._load_split_dataframes()
    
        self._build_scalers_transforms()
    
        self._build_datasets()
    
        self._build_dataloaders()
    
        self._build_metadata()

    def _load_split_dataframes(self):
    
        fold_dir = (
            Path(self.cfg.data.data_dir)
            / self.cfg.data.split.strategy
            / f"fold{self.cfg.data.fold}"
        )
    
        df_splits = {}
        for split in [
            'train', 
            'val',
            'test'
        ]:
            file_path = fold_dir / f"{split}.csv"
    
            if not file_path.exists():
                raise FileNotFoundError(file_path)
            
            df = pd.read_csv(file_path)
            df.rename(
                columns={c: canonical_name(c) for c in df.columns},
                inplace=True
            )
            df_splits[split] = df 
    
        return df_splits

    
    def _build_scalers_transforms(
        self,
    ):

        train_df = self.raw_data['train']

        train_inputs = self.build_inputs(train_df)

        train_targets = train_df[
            self.output_names
        ].to_numpy(
            dtype=float
        )

        self.input_scaler = make_scaler(
            self.cfg.data.input_scaler
        )

        self.input_scaler.fit(
            train_inputs
        )

        positives = {
            canonical_name(v)
            for v in self.cfg.data.positive_outputs
        }

        self.target_transform = (
            TargetTransform(
                names=self.output_names,
                positive_names=positives,
                mode=self.cfg.data.output_transform,
            )
        )

        transformed_targets = (
            self.target_transform.forward(
                train_targets
            )
        )

        self.output_scaler = make_scaler(
            self.cfg.data.output_scaler
        )

        self.output_scaler.fit(
            transformed_targets
        )

    def _build_datasets(
        self,
    ):

        train_df = self.raw_data['train']
        val_df = self.raw_data['val']
        test_df = self.raw_data['test']
        
        train_x = self.build_inputs(train_df)

        train_y = train_df[
            self.output_names
        ].to_numpy(
            dtype=float
        )

        val_x = self.build_inputs(val_df)

        val_y = val_df[
            self.output_names
        ].to_numpy(
            dtype=float
        )

        test_x = self.build_inputs(test_df)

        test_y = test_df[
            self.output_names
        ].to_numpy(
            dtype=float
        )

        mask_columns = [
            f"{name}_valid"
            for name in self.output_names
        ]
        has_masks = all(
            col in train_df.columns
            for col in mask_columns
        )
        has_injection = "injection_label" in train_df.columns

        def masks_for(df):
            if not has_masks:
                return None
            return df[mask_columns].to_numpy(dtype=float)

        def injection_for(df):
            if not has_injection:
                return None
            return df["injection_label"].to_numpy(dtype=float)

        self.train_dataset = (
            SurrogateDataset(
                train_x,
                train_y,
                train_df['original_index'],
                mask=masks_for(train_df),
                injection=injection_for(train_df),
            )
        )

        self.val_dataset = (
            SurrogateDataset(
                val_x,
                val_y,
                val_df['original_index'],
                mask=masks_for(val_df),
                injection=injection_for(val_df),
            )
        )

        self.test_dataset = (
            SurrogateDataset(
                test_x,
                test_y,
                test_df['original_index'],
                mask=masks_for(test_df),
                injection=injection_for(test_df),
            )
        )


    def _build_dataloaders(self):

        batch_size = (
            self.cfg.data.batch_size
        )

        num_workers = (
            self.cfg.data.num_workers
        )

        pin_memory = (
            self.cfg.data.pin_memory
            and torch.cuda.is_available()
        )

        self.train_dataloader = (
            DataLoader(
                self.train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=pin_memory,
            )
        )

        self.val_dataloader = (
            DataLoader(
                self.val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=pin_memory,
            )
        )

        self.test_dataloader = (
            DataLoader(
                self.test_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=pin_memory,
            )
        )


    def _build_metadata(self):

        groups = {}

        for (
            group_name,
            variables,
        ) in self.cfg.model.groups.items():

            groups[group_name] = [
                self.output_names.index(
                    canonical_name(v)
                )
                for v in variables
            ]

        self.group_indices = groups

        self.info = DatasetInfo(
            input_names=self.input_names,
            output_names=self.output_names,
            output_groups=dict(self.cfg.model.groups),
            positive_outputs=set(self.cfg.data.positive_outputs),
            fold=self.cfg.data.fold,
            n_inputs=len(self.input_names),
            n_outputs=len(self.output_names),
            n_train=len(self.train_dataset),
            n_val=len(self.val_dataset),
            n_test=len(self.test_dataset),
        )


    def build_inputs(self, df):
        """Return the model input matrix, optionally physics-expanded."""
        base = df[self.base_input_names].to_numpy(dtype=float)

        if not self.use_physics_features:
            return base

        return np.column_stack(
            [base, physics_features(base, self.base_input_names)]
        )

    def transform_inputs(
        self,
        x,
    ):
        return self.input_scaler.transform(
            x
        )

    def transform_targets(
        self,
        y,
    ):

        y = (
            self.target_transform
            .forward_torch(y)
        )

        y = (
            self.output_scaler
            .transform(y)
        )

        return y

    def inverse_targets(
        self,
        values,
    ):

        if isinstance(
            values,
            torch.Tensor,
        ):

            values = (
                self.output_scaler
                .inverse_transform(
                    values
                )
            )

            return (
                self.target_transform
                .inverse_torch(values)
            )

        values = (
            self.output_scaler
            .inverse_transform(values)
        )

        return (
            self.target_transform
            .inverse(values)
        )


    def scalers_states(self, ):
        return {
                'input_scaler_type': self.cfg.data.input_scaler, 
                'input_scaler_state': self.input_scaler.state_dict(),

                'output_scaler_type': self.cfg.data.output_scaler, 
                'output_scaler_state': self.output_scaler.state_dict(),

                'target_transform_cfg': self.target_transform.get_config()
                                
            }
        
    def save_scalers(self, save_dir: Path | str):
                
        # Save input/output scalers
        
        torch.save(
            self.scalers_states,
            f"{str(save_dir)}/scalers.pt",
        )


