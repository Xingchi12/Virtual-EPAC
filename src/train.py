"""Hydra entry point for vEPAC surrogate-model experiments."""

from __future__ import annotations

# Configure deterministic cuBLAS before importing PyTorch or creating CUDA
# handles. This prevents PyTorch's CUDA >= 10.2 reproducibility warning.
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from pathlib import Path
from typing import Any
import sys
import hydra
import numpy as np
import logging
from omegaconf import DictConfig, OmegaConf

from dataset.datamodule import DataModule
from dataset.split_data import DataPreprocessorSplitter
from losses import RegressionLoss
from model.model import build_model
from trainer import Trainer
from metrics import validation_metrics

import torch

import utils

def make_optimizer_and_scheduler(
    cfg: Any,
    model: torch.nn.Module,
    extra_modules: tuple[torch.nn.Module, ...] = (),
) -> (torch.optim.Optimizer, torch.optim.lr_scheduler):

    """Create the configured optimiser."""
    parameters = list(model.parameters())
    for module in extra_modules:
        parameters.extend(module.parameters())

    keyword_arguments = {
        "lr": cfg.optimizer.learning_rate,
        "weight_decay": cfg.optimizer.weight_decay,
        "betas": tuple(cfg.optimizer.betas),
    }
    name = cfg.optimizer.name.lower()
    if name == "adamw":
        optimizer = torch.optim.AdamW(parameters, **keyword_arguments)
    elif name == "adam":
        optimizer = torch.optim.Adam(parameters, **keyword_arguments)
    else:
        raise ValueError(f"Unsupported optimiser '{name}'.")

    
    ## Create the configured learning-rate scheduler."""
    name = cfg.scheduler.name.lower()
    if name == "none":
        scheduler = None
    elif name == "cosine":
        scheduler =  torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cfg.training.epochs,
            eta_min=cfg.scheduler.min_lr,
        )
    elif name == "reduce_on_plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=cfg.training.monitor_mode,
            factor=cfg.scheduler.factor,
            patience=cfg.scheduler.patience,
            min_lr=cfg.scheduler.min_lr,
        )
    else:
        raise ValueError(f"Unsupported scheduler '{name}'.")
        
    return optimizer, scheduler

def initialise_wandb(
    cfg: Any,
    experiment_dir: str,
    model: torch.nn.Module,
    datamodule: Any,
):
    """Initialise an optional Weights & Biases run and record metadata."""
    if not cfg.wandb.enabled:
        return None

    import wandb
    wandb_logged_cfg = OmegaConf.to_container(cfg, resolve=True)

    run = wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        group=cfg.wandb.group,
        name=os.path.dirname(experiment_dir),
        tags=list(cfg.wandb.tags),
        mode=cfg.wandb.mode,
        config=wandb_logged_cfg,
        dir=str(experiment_dir),
    )
    run.summary["trainable_parameters"] = utils.count_parameters(model)

    return run

                
def configure_paths(experiment_folder):
    """Configure the paths for the experiment with the given experiment folder."""
    # Make checkpoints directory as experiment_folder/checkpoints
    os.makedirs(os.path.join(experiment_folder, "checkpoints"), exist_ok=True)
    checkpoint_folder = os.path.join(experiment_folder, "checkpoints")
    # Store run/validation data as experiment_folder/artifacts
    # os.makedirs(os.path.join(experiment_folder, "artifacts"), exist_ok=True)
    # artifact_folder = os.path.join(experiment_folder, "artifacts")
    # Plot data directory as experiment_folder/viz
    os.makedirs(os.path.join(experiment_folder, "viz"), exist_ok=True)
    viz_folder = os.path.join(experiment_folder, "viz")
    return checkpoint_folder, viz_folder


def get_experiment_name(cfg: DictConfig) -> str:
    exp_name = [
        cfg.experiment.name,
        cfg.model.architecture,
        cfg.loss.name,
        f"scaler_{cfg.data.input_scaler}",
        f"ensembles{cfg.model.ensemble_size}",
        cfg.data.split.strategy,
        # f"fold{cfg.data.fold}",
    ]

    return "-".join(exp_name)


def resolve_experiment_folder(root_dir: str, experiment_name: str, cfg: DictConfig) -> str:
    """
    Return the directory for this run, protecting earlier results.

    With ``experiment.overwrite: false`` (the default) an existing
    directory is never written into. A ``__v2``, ``__v3``, ... suffix is
    appended until an unused name is found, so repeated runs of the same
    configuration accumulate side by side instead of silently destroying
    the previous run's checkpoints and metrics.
    """
    candidate = os.path.join(root_dir, experiment_name)

    if cfg.experiment.overwrite or not os.path.exists(candidate):
        return candidate

    version = 2
    while os.path.exists(f"{candidate}__v{version}"):
        version += 1

    versioned = f"{candidate}__v{version}"
    print(
        f"Experiment folder already exists: {candidate}\n"
        f"experiment.overwrite=false, so writing to: {versioned}"
    )
    return versioned


def resolve_folds(cfg: DictConfig) -> list[int]:
    """Return the folds to train, defaulting to every fold in the split."""
    requested = cfg.data.get("folds", None)

    if requested is None:
        return list(range(cfg.data.split.n_splits))

    if isinstance(requested, int):
        requested = [requested]
    else:
        requested = [int(f) for f in requested]

    invalid = [f for f in requested if not 0 <= f < cfg.data.split.n_splits]
    if invalid:
        raise ValueError(
            f"data.folds contains out-of-range folds {invalid}; valid "
            f"range is 0..{cfg.data.split.n_splits - 1}."
        )

    return requested


def check_data_splits(cfg: DictConfig) -> None:
    """
    Generate the cross-validation splits for ``cfg.data.split.strategy``
    if they are not already on disk.
    """
    split_root = Path(cfg.data.data_dir) / cfg.data.split.strategy

    expected_files = [
        split_root / f"fold{fold}" / f"{split_name}.csv"
        for fold in range(cfg.data.split.n_splits)
        for split_name in ("train", "val", "test")
    ]

    if all(f.exists() for f in expected_files):
        return

    print(
        f"Splits for strategy '{cfg.data.split.strategy}' not found "
        f"under {split_root}; generating them."
    )
    DataPreprocessorSplitter(cfg).run()


@hydra.main(version_base=None,config_path=".",config_name="config")
def main(cfg: DictConfig) -> None:
    """Build the experiment components and train the selected model family."""
    utils.seed_everything(
        cfg.experiment.seed,
        cfg.experiment.deterministic,
    )

    device = utils.resolve_device(cfg.experiment.device)

    check_data_splits(cfg)

    experiment_name = get_experiment_name(cfg)

    base_experiment_folder = resolve_experiment_folder(
        cfg.experiment.root_dir,
        experiment_name,
        cfg,
    )
    os.makedirs(base_experiment_folder, exist_ok=True)
    print(f"{base_experiment_folder=}")


    OmegaConf.save(
        cfg,
        base_experiment_folder + "/resolved_config.yaml",
        resolve=True,
    )

    utils.copy_src("./", f"{base_experiment_folder}/src/")
    with open(base_experiment_folder + "/command.txt", "w") as f:
        f.write(" ".join(sys.argv))

    folds = resolve_folds(cfg)
    print(f"Training folds: {folds}")

    # Insead of running one command per fold - use for loop
    for fold in folds:
        cfg.data.fold = fold
        
        experiment_dir = os.path.join(base_experiment_folder, f"fold{cfg.data.fold}")        
        os.makedirs(experiment_dir, exist_ok=True)

        OmegaConf.save(
            cfg,
            experiment_dir + "/resolved_config.yaml",
            resolve=True,
        )

        checkpoint_dir, viz_dir = configure_paths(experiment_dir)
    
        logger = utils.setup_logger(experiment_dir, name=cfg.experiment.name)
    
        datamodule = DataModule(cfg)
    
        model = build_model(
            cfg.model,
            datamodule.input_dim,
            datamodule.output_names,
        )

        loss_fn = RegressionLoss(
            name=cfg.loss.name,
            output_weights=cfg.loss.output_weights,
            variance_regularization=(
                cfg.loss.variance_regularization
            ),
            n_outputs=datamodule.output_dim,
            n_extra_tasks=(
                1 if cfg.model.get("injection_head", False) else 0
            ),
        )

        optimizer, scheduler = make_optimizer_and_scheduler(
            cfg,
            model,
            extra_modules=(loss_fn,),
        )
    
        logger.info( "Wandb run is initializing ... ")
        wandb_run = initialise_wandb(
            cfg,
            experiment_dir,
            model,
            datamodule,
        )
    
        logger.info(f"Experiment directory: {experiment_dir}")
        logger.info(f"Device: {device}")
        logger.info(f"Trainable parameters: {utils.count_parameters(model):,}")
        logger.info(
            "Split sizes: "
            + ", ".join(
                f"{split}={len(dataset)}"
                for split, dataset in datamodule.datasets.items()
            )
        )
    
        trainer = Trainer(
            cfg=cfg,
            model=model,
            optimizer=optimizer,
            datamodule=datamodule,
            loss_fn=loss_fn,
            device=device,
            experiment_dir=Path(experiment_dir),
            checkpoint_dir = Path(checkpoint_dir),
            viz_dir = Path(viz_dir),
            validation_metrics=validation_metrics(),
            wandb_run=wandb_run,
            scheduler=scheduler,
            logger=logger,
        )
    
        final_metrics = trainer.train()
        logger.info("Final validation-selected test metrics:")
        for name, value in sorted(final_metrics.items()):
            logger.info(f"  {name}: {value:.6g}")
    
        if wandb_run is not None:
            wandb_run.finish()


if __name__ == "__main__":
    main()
