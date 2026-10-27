from __future__ import annotations

import argparse
from pathlib import Path

import sys
import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
    
from dataset.datamodule import DataModule
from dataset.scalers import make_scaler
from model.model import DeepEnsemble, build_model

from losses import RegressionLoss
from trainer import Trainer
from metrics import validation_metrics

import utils


# Example usage
# python evaluate.py \
#     --exp-dir vepac_surrogates-multi_output-gaussian_nll-scaler_standard-ensembles5/0/fold0 \
#     --checkpoint recent.pt \
#     --split test \
#     --output evaluation_result.csv

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a pretrained surrogate model."
    )
    
    parser.add_argument(
        "--input-file",
        type=Path,
        default=None,
        help="CSV file for standalone evaluation."
    )

    parser.add_argument(
        "--exp-dir",
        type=Path,
        required=True,
        help="Experiment directory containing resolved_config.yaml",
    )

    parser.add_argument(
        "--checkpoint",
        default="best.pt",
        help="Checkpoint filename inside checkpoints directory.",
    )

    parser.add_argument(
        "--split",
        default="test",
        choices=["train", "val", "test"],
        help="Dataset split to evaluate.",
    )

    parser.add_argument(
        "--sampling-method",
        default="analytical_mean",
        choices=DeepEnsemble.SUPPORTED_SAMPLING_METHODS,
        help=(
            "Point-prediction strategy. The default is the analytical "
            "ensemble mean."
        ),
    )
    
    parser.add_argument(
        "--sampling-count",
        type=int,
        default=1000,
        help=(
            "Number of predictive samples per input. This is ignored for "
            "the analytical ensemble mean."
        ),
    )
    
    parser.add_argument(
        "--top_k_samples",
        type=int,
        default=50,
        help=(
            "Number of highest-density samples to average when using "
            "top_k_density_mean."
        ),
    )
    return parser.parse_args()





def main():

    args = parse_args()
    print(args)

    if args.sampling_method == "top_k_density_mean":
        if args.top_k_samples is None:
            raise ValueError(
                "top_k_samples must be provided when sampling_method "
                "is 'top_k_density_mean'."
            )
            
    config_path = args.exp_dir / "resolved_config.yaml"

    if not config_path.exists():
        raise FileNotFoundError(
            f"Could not find config file: {config_path}"
        )

    cfg = OmegaConf.load(config_path)

    utils.seed_everything(
        cfg.experiment.seed,
        cfg.experiment.deterministic,
    )

    device = utils.resolve_device(
        cfg.experiment.device
    )

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

    logger = logging.getLogger("evaluation")
    logger.setLevel(logging.INFO)

    trainer = Trainer(
        cfg=cfg,
        model=model,
        optimizer=None,
        datamodule=datamodule,
        loss_fn=loss_fn,
        validation_metrics=validation_metrics(),
        device=device,
        experiment_dir=args.exp_dir,
        checkpoint_dir=args.exp_dir / "checkpoints",
        viz_dir=args.exp_dir / "evaluation",
        wandb_run=None,
        scheduler=None,
        logger=logger,
    )

    checkpoint_path = (
        args.exp_dir
        / "checkpoints"
        / args.checkpoint
    )

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    trainer.model.load_state_dict(
        checkpoint["model_state"], 
        strict=True
    )
    print("Model loaded!")
    
    trainer.model.eval()

    epoch = int(checkpoint.get("epoch", 0))

    metrics, payload = trainer.evaluate(
        split=args.split,
        epoch=epoch,
        sampling_method=args.sampling_method,
        sampling_count=args.sampling_count,
        top_k_samples=args.top_k_samples,
    )

    file_tags = [args.split, args.sampling_method]
    if args.sampling_method != 'analytical_mean':
        file_tags.append(f"samples{args.sampling_count}")
    if args.sampling_method == "top_k_density_mean":
        file_tags.append(f"topk_{args.top_k_samples}")
    file_tags = "_".join([str(t) for t in file_tags])
    
    result_path = args.exp_dir / f"evaluation_results_{file_tags}.csv" 
    
    df_payload = trainer.payload_to_df(payload)

    
    df_payload.to_csv(result_path, index=False,)
    print(f"Predictions saved to: {result_path}")

    metrics_df = pd.DataFrame(
        [
            {
                "metric": key,
                "value": value,
            }
            for key, value in metrics.items()
        ]
    )
    
    metrics_path = args.exp_dir / f"evaluation_metrics_{file_tags}.csv" 
    
    metrics_df.to_csv(
        metrics_path,
        index=False,
    )
    
    print(f"Metrics saved to: {metrics_path}")

    print()
    print("=" * 80)
    print("Evaluation Metrics")
    print("=" * 80)

    for name, value in sorted(metrics.items()):
        print(f"{name:<30} {value:.6f}")

if __name__ == "__main__":
    main()