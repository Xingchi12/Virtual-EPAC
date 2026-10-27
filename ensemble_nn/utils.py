from __future__ import annotations

import json
import os
import sys
import random
import re
import shutil
from pathlib import Path
from typing import Any
from datetime import datetime
from omegaconf import DictConfig, OmegaConf
import logging

import numpy as np
import pandas as pd

# cuBLAS reads this variable when its CUDA handle is created. It must be set
# before the first CUDA matrix multiplication, and preferably before importing
# PyTorch in the process entry point.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

def setup_logger(
    experiment_dir: Path,
    name: str = "vepac_surrogates",
    level: int = logging.INFO,
) -> logging.Logger:
    """
    Configure a logger that writes to both console and
    experiment_dir/train.log.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Prevent duplicate handlers if called multiple times
    if logger.handlers:
        return logger

    logger.propagate = False 
    
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    # Console
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    # File
    log_file = str(experiment_dir)  +  "/train.log"
    file_handler = logging.FileHandler(log_file, mode="a")
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger
    

def canonical_name(name: str) -> str:
    """Convert spreadsheet headings to stable Python-style column names."""
    cleaned = re.sub(r"[^0-9A-Za-z]+", "_", str(name).strip())
    return re.sub(r"_+", "_", cleaned).strip("_")


def seed_everything(seed: int, deterministic: bool = True) -> None:
    """Seed Python, NumPy and PyTorch and configure deterministic execution."""
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)


def resolve_device(requested: str = "auto"):
    """Resolve ``auto``, CPU or an explicit CUDA device to ``torch.device``."""
    import torch

    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(requested)



def copy_src(root_src_dir, root_dst_dir, overwrite=True, ignore='outputs'):
    print(f"Copying source: {root_src_dir}")
    for src_dir, dirs, files in os.walk(root_src_dir):
        if '__pycache__' in src_dir:
            continue
        if '.ipynb_checkpoints' in src_dir:
            continue
        if '.hydra' in src_dir:
            continue

        if ignore in src_dir:
            continue
            
        dst_dir = src_dir.replace(root_src_dir, root_dst_dir, 1)
        print(f"Copying {src_dir} -> {dst_dir}")
        
        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir, exist_ok=True)
        for file in files:
            if 'cpython' in file:
                continue
            src_file = os.path.join(src_dir, file)
            dst_file = os.path.join(dst_dir, file)
            if os.path.exists(dst_file):
                if overwrite:
                    shutil.copy(src_file, dst_file)
            else:
                shutil.copy(src_file, dst_file)


    

def count_parameters(model: Any) -> int:
    """Return the number of trainable model parameters."""
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def aggregate_uniform_fold_ensemble(
    all_fold_predictions_df,
    target_columns,
    input_columns,
):
    """
    Aggregate predictions from all fold models using equal weights.

    The total predictive variance contains:

    1. Mean within-model predictive variance.
    2. Between-fold variance of the predictive means.
    """

    # Use fold 0 only to recover one copy of the original row data.
    result = (
        all_fold_predictions_df
        .sort_values(["row_id", "fold"])
        .groupby("row_id", as_index=False)
        .first()
    )

    # Remove fold-specific prediction columns before adding aggregate
    # prediction columns.
    columns_to_keep = [
        "row_id",
        *input_columns,
        *target_columns,
    ]

    columns_to_keep = [
        column
        for column in columns_to_keep
        if column in result.columns
    ]

    result = result[columns_to_keep].copy()

    for target in target_columns:
        prediction_column = f"{target}_pred"
        std_column = f"{target}_std"

        fold_means = (
            all_fold_predictions_df
            .pivot(
                index="row_id",
                columns="fold",
                values=prediction_column,
            )
            .sort_index()
        )

        fold_stds = (
            all_fold_predictions_df
            .pivot(
                index="row_id",
                columns="fold",
                values=std_column,
            )
            .sort_index()
        )

        if fold_means.isna().any().any():
            raise RuntimeError(
                f"Missing fold means for target '{target}'."
            )

        if fold_stds.isna().any().any():
            raise RuntimeError(
                f"Missing fold standard deviations for target "
                f"'{target}'."
            )

        # Uniform average of fold predictive means.
        aggregate_mean = fold_means.mean(
            axis=1
        )

        # Average uncertainty reported inside each fold model.
        within_fold_variance = (
            fold_stds.pow(2).mean(axis=1)
        )

        # Disagreement between predictions from different folds.
        #
        # ddof=0 is appropriate because these five fold models form
        # the complete deployed ensemble, rather than a statistical
        # sample for estimating a population variance.
        between_fold_variance = fold_means.var(
            axis=1,
            ddof=0,
        )

        total_variance = (
            within_fold_variance
            + between_fold_variance
        ).clip(lower=1.0e-12)

        result[
            f"{target}_pred"
        ] = aggregate_mean.to_numpy()

        result[
            f"{target}_within_fold_std"
        ] = np.sqrt(
            within_fold_variance.to_numpy()
        )

        result[
            f"{target}_between_fold_std"
        ] = np.sqrt(
            between_fold_variance.to_numpy()
        )

        result[
            f"{target}_std"
        ] = np.sqrt(
            total_variance.to_numpy()
        )

    return result

def select_lowest_std_predictions(
    all_fold_predictions_df,
    target_columns,
    input_columns,
):
    """
    Select, independently for each target and row, the fold prediction
    with the lowest reported predictive standard deviation.
    """

    result = (
        all_fold_predictions_df
        .sort_values(["row_id", "fold"])
        .groupby("row_id", as_index=False)
        .first()
    )

    columns_to_keep = [
        "row_id",
        *input_columns,
        *target_columns,
    ]

    columns_to_keep = [
        column
        for column in columns_to_keep
        if column in result.columns
    ]

    result = result[columns_to_keep].copy()

    for target in target_columns:
        prediction_column = f"{target}_pred"
        std_column = f"{target}_std"

        fold_predictions = (
            all_fold_predictions_df
            .pivot(
                index="row_id",
                columns="fold",
                values=prediction_column,
            )
            .sort_index()
        )

        fold_stds = (
            all_fold_predictions_df
            .pivot(
                index="row_id",
                columns="fold",
                values=std_column,
            )
            .sort_index()
        )

        # Position of the minimum-uncertainty fold for every row.
        selected_positions = np.argmin(
            fold_stds.to_numpy(),
            axis=1,
        )

        row_positions = np.arange(
            len(fold_stds)
        )

        selected_predictions = (
            fold_predictions
            .to_numpy()[
                row_positions,
                selected_positions,
            ]
        )

        selected_stds = (
            fold_stds
            .to_numpy()[
                row_positions,
                selected_positions,
            ]
        )

        fold_labels = fold_stds.columns.to_numpy()

        selected_folds = fold_labels[
            selected_positions
        ]

        result[
            f"{target}_pred"
        ] = selected_predictions

        result[
            f"{target}_std"
        ] = selected_stds

        result[
            f"{target}_selected_fold"
        ] = selected_folds

    return result

def aggregate_median_predictions(
    all_fold_predictions_df,
    target_columns,
    input_columns,
):
    """
    Aggregate fold predictions using the median predictive mean.

    The uncertainty is represented using fold disagreement based on
    the median absolute deviation.
    """

    result = (
        all_fold_predictions_df
        .sort_values(["row_id", "fold"])
        .groupby("row_id", as_index=False)
        .first()
    )

    columns_to_keep = [
        "row_id",
        *input_columns,
        *target_columns,
    ]

    columns_to_keep = [
        column
        for column in columns_to_keep
        if column in result.columns
    ]

    result = result[columns_to_keep].copy()

    for target in target_columns:
        prediction_column = f"{target}_pred"
        std_column = f"{target}_std"

        fold_predictions = (
            all_fold_predictions_df
            .pivot(
                index="row_id",
                columns="fold",
                values=prediction_column,
            )
            .sort_index()
        )

        fold_stds = (
            all_fold_predictions_df
            .pivot(
                index="row_id",
                columns="fold",
                values=std_column,
            )
            .sort_index()
        )

        median_prediction = fold_predictions.median(
            axis=1
        )

        absolute_deviations = fold_predictions.sub(
            median_prediction,
            axis=0,
        ).abs()

        median_absolute_deviation = (
            absolute_deviations.median(axis=1)
        )

        # Multiplication by 1.4826 places MAD approximately on the
        # standard-deviation scale under a normal distribution.
        robust_between_fold_std = (
            1.4826
            * median_absolute_deviation
        )

        within_fold_variance = (
            fold_stds.pow(2).mean(axis=1)
        )

        total_variance = (
            within_fold_variance
            # + robust_between_fold_std.pow(2)
        ).clip(lower=1.0e-12)

        result[
            f"{target}_pred"
        ] = median_prediction.to_numpy()

        result[
            f"{target}_between_fold_std"
        ] = robust_between_fold_std.to_numpy()

        result[
            f"{target}_std"
        ] = np.sqrt(
            total_variance.to_numpy()
        )

    return result

def aggregate_weighted_fold_ensemble(
    all_fold_predictions_df,
    target_columns,
    input_columns,
    weights_by_target,
):
    """
    Aggregate fold predictions using fixed target-specific weights.

    weights_by_target example:
    {
        "e_charge_fc": {
            0: 0.20,
            1: 0.25,
            2: 0.15,
            3: 0.20,
            4: 0.20,
        },
        ...
    }
    """

    result = (
        all_fold_predictions_df
        .sort_values(["row_id", "fold"])
        .groupby("row_id", as_index=False)
        .first()
    )

    columns_to_keep = [
        "row_id",
        *input_columns,
        *target_columns,
    ]

    columns_to_keep = [
        column
        for column in columns_to_keep
        if column in result.columns
    ]

    result = result[columns_to_keep].copy()

    for target in target_columns:
        prediction_column = f"{target}_pred"
        std_column = f"{target}_std"

        fold_means = (
            all_fold_predictions_df
            .pivot(
                index="row_id",
                columns="fold",
                values=prediction_column,
            )
            .sort_index()
        )

        fold_stds = (
            all_fold_predictions_df
            .pivot(
                index="row_id",
                columns="fold",
                values=std_column,
            )
            .sort_index()
        )

        target_weights = weights_by_target[
            target
        ]

        weights = np.array(
            [
                target_weights[fold]
                for fold in fold_means.columns
            ],
            dtype=float,
        )

        if np.any(weights < 0):
            raise ValueError(
                f"Negative weights found for target '{target}'."
            )

        weights = weights / weights.sum()

        means_array = fold_means.to_numpy()
        variances_array = fold_stds.to_numpy() ** 2

        aggregate_mean = (
            means_array * weights[None, :]
        ).sum(axis=1)

        aggregate_second_moment = (
            weights[None, :]
            * (
                variances_array
                + means_array ** 2
            )
        ).sum(axis=1)

        total_variance = (
            aggregate_second_moment
            - aggregate_mean ** 2
        )

        total_variance = np.maximum(
            total_variance,
            1.0e-12,
        )

        result[
            f"{target}_pred"
        ] = aggregate_mean

        result[
            f"{target}_std"
        ] = np.sqrt(total_variance)

    return result
