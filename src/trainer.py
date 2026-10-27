
from __future__ import annotations

import os
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any
import logging

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.amp import GradScaler, autocast
from tqdm.auto import tqdm
import pandas as pd

from losses import RegressionLoss
from metrics import aggregate_metrics
from dataset.datamodule import DataModule
import utils


class Trainer:
    """
    Train and evaluate a deep ensemble for multi-output regression.
    """

    def __init__(
        self,
        cfg: Any,
        model: torch.nn.Module,
        datamodule: DataModule,
        validation_metrics: List,
        device: torch.device,
        experiment_dir: Path,
        optimizer: torch.optim.Optimizer = None,
        loss_fn: RegressionLoss = None,
        checkpoint_dir: Path = None,
        viz_dir: Path = None, 
        wandb_run: Any = None,
        scheduler: Any = None,
        logger: logging.Logger = None
        
    ) -> None:
        self.cfg = cfg
        self.model = model.to(device)
        self.optimizer = optimizer
        self.datamodule = datamodule
        self.loss_fn = loss_fn or RegressionLoss()
        self.validation_metrics = validation_metrics
        self.device = device
        self.experiment_dir = experiment_dir 
        self.checkpoint_dir = checkpoint_dir
        self.viz_dir = viz_dir
        self.wandb_run = wandb_run
        self.scheduler = scheduler
        self.logger = logger
        
        self.history: dict[str, list[tuple[int, float]]] = defaultdict(list)
        self.best_metrics: dict[str, float] = {}
        self.start_epoch = 1
        self.last_validation_metrics: dict[str, float] | None = None
        self.last_test_metrics: dict[str, float] | None = None

        # target_noise_std is either a scalar applied to every output, or
        # one value per output (in datamodule.output_names order) so that
        # jitter can be targeted at the outputs it actually helps.
        noise_std = cfg.training.get("target_noise_std", 0.0)
        if isinstance(noise_std, (int, float)):
            self.target_noise_std = float(noise_std)
            self.target_noise_std_vector = None
        else:
            values = [float(v) for v in noise_std]
            expected = len(datamodule.output_names)
            if len(values) != expected:
                raise ValueError(
                    "training.target_noise_std given as a list must have "
                    f"one entry per output ({expected}: "
                    f"{datamodule.output_names}); got {len(values)}."
                )
            self.target_noise_std = max(values)
            self.target_noise_std_vector = torch.tensor(
                values,
                dtype=torch.float32,
                device=device,
            ).unsqueeze(0)

        # Gaussian jitter on the scaled inputs. Unlike target noise this
        # is a genuine data augmentation: it approximates Tikhonov
        # regularisation (Bishop, 1995), and it is physically motivated
        # here because the inputs are *requested* simulator settings that
        # differ from the realised values.
        self.input_noise_std = float(
            cfg.training.get("input_noise_std", 0.0)
        )

        noise_range = cfg.training.get("target_noise_std_range", None)
        if noise_range is None:
            self.target_noise_std_range = None
        else:
            low, high = (float(v) for v in noise_range)
            if low < 0.0 or high < low:
                raise ValueError(
                    "training.target_noise_std_range must be "
                    f"[low, high] with 0 <= low <= high; got [{low}, {high}]."
                )
            self.target_noise_std_range = (low, high)

        self.amp_enabled = bool(
            cfg.training.mixed_precision and device.type == "cuda"
        )
        self.gradient_scaler = GradScaler(
            device.type,
            enabled=self.amp_enabled,
        )

        self.loss_fn = self.loss_fn.to(device)

    def _ensemble_training_loss(
        self,
        member_outputs: list[dict[str, torch.Tensor | None]],
        targets: torch.Tensor,
        mask: torch.Tensor | None = None,
        injection_labels: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Average the loss across all ensemble members.

        Every member is trained directly. Optimising only the ensemble mean
        would allow poorly performing members to cancel each other and would
        weaken epistemic-uncertainty estimates.

        Returns the combined scalar loss (used for backpropagation), the
        mean per-output regression loss across members (for logging), and
        the mean raw injection BCE across members (``None`` if the
        architecture has no injection head).
        """
        member_totals = []
        member_per_output = []
        member_injection_bce = []

        for output in member_outputs:
            total, per_output = self.loss_fn(
                output["mean"],
                targets,
                output["log_var"],
                mask=mask,
            )

            injection_logit = output.get("injection_logit")
            if injection_logit is not None and injection_labels is not None:
                injection_total, injection_bce = self.loss_fn.injection_loss(
                    injection_logit,
                    injection_labels,
                )
                total = total + injection_total
                member_injection_bce.append(injection_bce)

            member_totals.append(total)
            member_per_output.append(per_output)

        combined = torch.stack(member_totals).mean()
        per_output_mean = torch.stack(member_per_output).mean(dim=0)
        injection_bce_mean = (
            torch.stack(member_injection_bce).mean()
            if member_injection_bce
            else None
        )

        return combined, per_output_mean, injection_bce_mean
        
    @torch.no_grad()
    def evaluate(
        self,
        split: str,
        epoch: int,
        sampling_method: str = "analytical_mean",
        sample_count: int = 1000,
        top_k_samples: int | None = None,
    ) -> tuple[dict[str, float], dict[str, torch.Tensor]]:
        """
        Evaluate one dataset split using a selected point estimator.
    
        Parameters
        ----------
        split:
            Dataset split to evaluate, such as ``train``, ``val``, or ``test``.
    
        epoch:
            Epoch associated with the checkpoint or evaluation.
    
        sampling_method:
            Point-prediction strategy. Supported methods are:
    
            - ``analytical_mean``
            - ``sample_mean``
            - ``marginal_median``
            - ``highest_density``
            - ``top_k_density_mean``
    
            The default is ``analytical_mean``.
    
        sample_count:
            Number of predictive samples generated per input. The default is
            1,000. This is ignored for ``analytical_mean``.
    
        top_k_samples:
            Number of highest-density samples averaged when using
            ``top_k_density_mean``.
    
        Notes
        -----
        The regression loss always uses the analytical ensemble mean and
        analytical predictive variance.
    
        The selected sampling strategy only changes the point prediction used
        for prediction metrics and returned evaluation results.
        """
        self.model.eval()
    
        if sampling_method not in self.model.SUPPORTED_SAMPLING_METHODS:
            supported = ", ".join(
                self.model.SUPPORTED_SAMPLING_METHODS
            )
    
            raise ValueError(
                f"Unsupported sampling method '{sampling_method}'. "
                f"Supported methods are: {supported}."
            )
    
        if sample_count < 1:
            raise ValueError(
                "sample_count must be at least 1, "
                f"got {sample_count}."
            )
    
        if sampling_method == "top_k_density_mean":
            if top_k_samples is None:
                raise ValueError(
                    "top_k_samples must be provided when sampling_method "
                    "is 'top_k_density_mean'."
                )
    
            if top_k_samples < 1:
                raise ValueError(
                    "top_k_samples must be at least 1, "
                    f"got {top_k_samples}."
                )
    
            if top_k_samples > sample_count:
                raise ValueError(
                    f"top_k_samples={top_k_samples} cannot exceed "
                    f"sample_count={sample_count}."
                )
    
        physical_inputs: list[torch.Tensor] = []
        physical_targets: list[torch.Tensor] = []
        physical_masks: list[torch.Tensor] = []

        scaled_predictions: list[torch.Tensor] = []
        scaled_variances: list[torch.Tensor] = []

        original_indices: list[torch.Tensor] = []

        injection_labels_all: list[torch.Tensor] = []
        injection_probabilities_all: list[torch.Tensor] = []

        weighted_loss = 0.0
        evaluated_count = 0
        weighted_injection_bce = 0.0
        injection_batches = 0
    
        description = (
            f"Epoch {epoch:04d} | {split} | {sampling_method}"
        )
    
        if sampling_method == "top_k_density_mean":
            description += f" | k={top_k_samples}"
    
        progress = tqdm(
            self.datamodule.dataloaders[split],
            desc=description,
            unit="batch",
            dynamic_ncols=True,
            leave=False,
        )
    
        for batch in progress:
            inputs = batch["inputs"].to(
                self.device,
                non_blocking=True,
            )
    
            targets = batch["targets"].to(
                self.device,
                non_blocking=True,
            )

            mask = batch["mask"].to(
                self.device,
                non_blocking=True,
            )

            injection_labels = (
                batch["injection"].to(
                    self.device,
                    non_blocking=True,
                )
                if "injection" in batch
                else None
            )

            inputs_scaled = self.datamodule.transform_inputs(
                inputs
            )

            targets_scaled = self.datamodule.transform_targets(
                targets
            )

            targets_scaled = torch.nan_to_num(targets_scaled, nan=0.0)

            indices = batch["index"]
    
            # The predict method calculates the ensemble-member outputs only
            # once and returns both analytical and selected predictions.
            output = self.model.predict(
                inputs=inputs_scaled,
                sampling_method=sampling_method,
                sample_count=sample_count,
                top_k=top_k_samples,
                return_samples=False,
            )
    
            analytical_mean = output["mean"]
            analytical_log_variance = output["log_var"]
            selected_prediction = output["prediction"]
    
            if analytical_mean is None:
                raise RuntimeError(
                    "The model returned no analytical mean."
                )
    
            if analytical_log_variance is None:
                raise RuntimeError(
                    "DeepEnsemble.predict() must return total log variance."
                )
    
            if selected_prediction is None:
                raise RuntimeError(
                    "DeepEnsemble.predict() returned no selected prediction."
                )
    
            # The loss remains associated with the complete analytical
            # predictive distribution, not with a post-selected sample.
            loss, _ = self.loss_fn(
                analytical_mean,
                targets_scaled,
                analytical_log_variance,
                mask=mask,
            )

            injection_probability = output.get("injection_probability")
            if injection_probability is not None and injection_labels is not None:
                injection_logit = torch.logit(
                    injection_probability.clamp(1.0e-6, 1.0 - 1.0e-6)
                )
                injection_total, injection_bce = self.loss_fn.injection_loss(
                    injection_logit,
                    injection_labels,
                )
                loss = loss + injection_total
                weighted_injection_bce += float(injection_bce) * inputs.shape[0]
                injection_batches += inputs.shape[0]

                injection_labels_all.append(injection_labels.detach().cpu())
                injection_probabilities_all.append(
                    injection_probability.detach().cpu()
                )

            batch_size = inputs.shape[0]

            weighted_loss += float(loss) * batch_size
            evaluated_count += batch_size

            # Move accumulated tensors to CPU to avoid retaining the entire
            # evaluation split in GPU memory.
            physical_inputs.append(
                inputs.detach().cpu()
            )

            physical_targets.append(
                targets.detach().cpu()
            )

            physical_masks.append(
                mask.detach().cpu()
            )

            scaled_predictions.append(
                selected_prediction.detach().cpu()
            )

            scaled_variances.append(
                analytical_log_variance.exp().detach().cpu()
            )

            original_indices.append(
                torch.as_tensor(indices)
            )
    
            running_loss = (
                weighted_loss
                / max(evaluated_count, 1)
            )
    
            progress.set_postfix(
                loss=f"{running_loss:.5f}",
            )
    
        progress.close()
    
        if evaluated_count == 0:
            raise RuntimeError(
                f"The '{split}' dataloader produced no samples."
            )
    
        input_physical = torch.cat(
            physical_inputs,
            dim=0,
        )
    
        target_physical = torch.cat(
            physical_targets,
            dim=0,
        )

        mask_physical = torch.cat(
            physical_masks,
            dim=0,
        )

        prediction_scaled = torch.cat(
            scaled_predictions,
            dim=0,
        )
    
        variance_scaled = torch.cat(
            scaled_variances,
            dim=0,
        )
    
        indices = torch.cat(
            original_indices,
            dim=0,
        )
    
        # Convert the selected point estimate back to physical units.
        prediction_physical = (
            self.datamodule.inverse_targets(
                prediction_scaled
            )
        )
    
        # The reported uncertainty remains the analytical ensemble uncertainty.
        std_scaled = torch.sqrt(
            variance_scaled.clamp_min(1.0e-12)
        )
    
        upper_physical = self.datamodule.inverse_targets(
            prediction_scaled + std_scaled
        )
    
        lower_physical = self.datamodule.inverse_targets(
            prediction_scaled - std_scaled
        )
    
        std_physical = (
            (upper_physical - lower_physical)
            .abs()
            .div(2.0)
            .clamp_min(1.0e-12)
        )
    
        # Masked-out entries are physically undefined placeholders; fill
        # them with a finite value so the metric validators do not
        # reject the tensor. `mask_physical` excludes them from every
        # reported statistic regardless of this fill value.
        target_physical_filled = torch.nan_to_num(
            target_physical,
            nan=0.0,
        )

        metrics = aggregate_metrics(
            target_physical_filled,
            prediction_physical,
            self.datamodule.output_names,
            pred_std=std_physical,
            coverage=self.cfg.training.uncertainty_coverage,
            metrics=self.validation_metrics,
            mask=mask_physical,
        )

        metrics["loss"] = (
            weighted_loss
            / max(evaluated_count, 1)
        )

        if injection_batches > 0:
            injection_labels_cat = torch.cat(injection_labels_all, dim=0)
            injection_probabilities_cat = torch.cat(
                injection_probabilities_all, dim=0
            ).reshape(-1)

            predicted_positive = (injection_probabilities_cat >= 0.5).float()
            accuracy = (
                (predicted_positive == injection_labels_cat).float().mean()
            )

            metrics["injection/bce"] = weighted_injection_bce / injection_batches
            metrics["injection/accuracy"] = float(accuracy)

        outputs = {
            "x": input_physical,
            "y": target_physical,
            "mask": mask_physical,
            "pred": prediction_physical,
            "std": std_physical,
            "indices": indices,
        }

        if injection_batches > 0:
            outputs["injection_label"] = torch.cat(injection_labels_all, dim=0)
            outputs["injection_probability"] = torch.cat(
                injection_probabilities_all, dim=0
            ).reshape(-1)

        return metrics, outputs
        
    def train_epoch(self, epoch: int) -> float:
        """Train all ensemble members for one epoch and return mean loss."""
        self.model.train()
        weighted_loss = 0.0
        sample_count = 0

        progress = tqdm(
            self.datamodule.train_dataloader,
            desc=f"Epoch {epoch:04d} | train",
            unit="batch",
            dynamic_ncols=True,
            leave=False,
        )

        for step, batch in enumerate(progress, start=1):
        
            inputs = batch["inputs"].to(
                self.device,
                non_blocking=True,
            )
            
            targets = batch["targets"].to(
                self.device,
                non_blocking=True,
            )

            mask = batch["mask"].to(
                self.device,
                non_blocking=True,
            )

            injection_labels = (
                batch["injection"].to(
                    self.device,
                    non_blocking=True,
                )
                if "injection" in batch
                else None
            )

            inputs = self.datamodule.transform_inputs(
                inputs
            )

            if self.input_noise_std > 0.0:
                inputs = inputs + self.input_noise_std * torch.randn_like(
                    inputs
                )

            targets = self.datamodule.transform_targets(
                targets
            )

            # Missing targets are physically undefined (not merely
            # noisy) for masked-out entries. They are replaced with a
            # finite placeholder purely so that arithmetic does not
            # propagate NaNs; `mask` excludes them from the loss.
            targets = torch.nan_to_num(targets, nan=0.0)

            # Optional target jitter, applied in scaled space so the
            # configured sigma is in units of each output's standard
            # deviation. Training only -- evaluation always uses the
            # clean targets.
            #
            # With target_noise_std_range set, the magnitude is resampled
            # per training example from U(low, high) instead of being
            # fixed, so the model sees a spread of corruption levels
            # rather than one.
            if self.target_noise_std_vector is not None:
                targets = targets + self.target_noise_std_vector.to(
                    targets.dtype
                ) * torch.randn_like(targets)

            elif self.target_noise_std_range is not None:
                low, high = self.target_noise_std_range
                sigma = torch.empty(
                    targets.shape[0],
                    1,
                    device=targets.device,
                    dtype=targets.dtype,
                ).uniform_(low, high)
                targets = targets + sigma * torch.randn_like(targets)

            elif self.target_noise_std > 0.0:
                targets = targets + self.target_noise_std * torch.randn_like(
                    targets
                )

            self.optimizer.zero_grad(set_to_none=True)

            with autocast(
                device_type=self.device.type,
                enabled=self.amp_enabled,
            ):
                member_outputs = self.model.forward_members(inputs)
                loss, _, _ = self._ensemble_training_loss(
                    member_outputs,
                    targets,
                    mask=mask,
                    injection_labels=injection_labels,
                )

            self.gradient_scaler.scale(loss).backward()
            self.gradient_scaler.unscale_(self.optimizer)
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                list(self.model.parameters())
                + list(self.loss_fn.parameters()),
                self.cfg.optimizer.gradient_clip_norm,
            )
            self.gradient_scaler.step(self.optimizer)
            self.gradient_scaler.update()

            batch_size = len(inputs)
            weighted_loss += float(loss.detach()) * batch_size
            sample_count += batch_size
            running_loss = weighted_loss / max(sample_count, 1)
            progress.set_postfix(
                loss=f"{running_loss:.5f}",
                lr=f"{self.optimizer.param_groups[0]['lr']:.2e}",
            )

            if (
                self.wandb_run is not None
                and step % self.cfg.training.logging_interval == 0
            ):
                self.wandb_run.log(
                    {
                        "train/batch_loss": float(loss.detach()),
                        "train/gradient_norm": float(gradient_norm),
                        "epoch": epoch,
                    }
                )

        return weighted_loss / max(sample_count, 1)

    
    def train(self) -> dict[str, float]:
        """Run optimisation, validation-based selection and final testing."""
        epochs_without_improvement = 0
        primary_metric = self.cfg.training.monitor

        if self.cfg.experiment.resume:
            self.restore_training_state()
    
        self.logger.info("Trainer started ... ")

        for epoch in range(
            self.start_epoch,
            self.cfg.training.epochs + 1,
        ):
            train_loss = self.train_epoch(epoch)
            epoch_log: dict[str, float] = {
                "epoch": epoch,
                "train/loss": train_loss,
            }
            validation_performed = (
                epoch % self.cfg.training.validation_interval == 0
            )
            test_performed = (
                epoch % self.cfg.training.test_interval == 0
                or epoch == self.cfg.training.epochs
            )

            if validation_performed:
                metrics, payload = self.evaluate('val', epoch)
                self.logger.info(", ".join([
                    f"{'val'.capitalize():>5}",
                    f"loss={metrics['loss']:.6f}",                      
                    ])
                )

                self._record_metrics(
                    'val',
                    epoch,
                    metrics,
                    epoch_log,
                )

                self.last_validation_metrics = metrics
                if (epoch% self.cfg.training.prediction_plot_interval == 0):
                    self._plot_predictions('val', payload, epoch)

                monitored_value = epoch_log[primary_metric]
                previous_best = self.best_metrics.get(primary_metric)
                if self._is_better(monitored_value, previous_best):
                    self.best_metrics[primary_metric] = monitored_value
                    epochs_without_improvement = 0
                    self._save_checkpoint(epoch, "best.pt", epoch_log)

                    self.payload_to_df(
                        payload,
                        self.experiment_dir / "best_val_predictions.csv"
                    )
                
                else:
                    epochs_without_improvement += 1

                for metric_name in self.cfg.training.save_best_metrics:
                    if metric_name not in epoch_log:
                        continue
                    previous = self.best_metrics.get(metric_name)
                    if self._is_better(epoch_log[metric_name], previous):
                        self.best_metrics[metric_name] = epoch_log[metric_name]
                        safe_name = metric_name.replace("/", "_")
                        self._save_checkpoint(
                            epoch,
                            f"best_{safe_name}.pt",
                            epoch_log,
                        )

                if (
                    self.scheduler is not None
                    and self.cfg.scheduler.name == "reduce_on_plateau"
                ):
                    self.scheduler.step(monitored_value)

            if test_performed:
                metrics, payload = self.evaluate("test", epoch)
                self.last_test_metrics = metrics
                self._record_metrics(
                    "test",
                    epoch,
                    metrics,
                    epoch_log,
                )

                self.payload_to_df(
                    payload,
                    self.experiment_dir / "test_predictions.csv"
                )
                if (epoch % self.cfg.training.prediction_plot_interval == 0  or epoch == self.cfg.training.epochs):
                    self._plot_predictions("test", payload, epoch)

            if (
                self.scheduler is not None
                and self.cfg.scheduler.name == "cosine"
            ):
                self.scheduler.step()

            if self.cfg.training.save_every_epoch:
                self._save_checkpoint(
                    epoch,
                    f"epoch_{epoch:04d}.pt",
                    epoch_log,
                )
            self._save_checkpoint(epoch, "recent.pt", epoch_log)

            if self.wandb_run is not None:
                self.wandb_run.log(epoch_log)

            self._plot_history()
            self.save_json(
                dict(self.history),
                self.experiment_dir / "history.json",
            )
            
            self.print_epoch_summary(
                epoch,
                train_loss,
                validation_performed,
                test_performed,
            )
        
            if (
                epochs_without_improvement
                >= self.cfg.training.early_stopping_patience
            ):
                tqdm.write(
                    "Early stopping: the validation criterion did not "
                    f"improve for {epochs_without_improvement} checks."
                )
                break

        return self._evaluate_best_checkpoint()
        
    def payload_to_df(
        self,
        payload,
        save_path: str = None,
    ):

        for k, v in payload.items():
            if isinstance(v, torch.Tensor):
                payload[k] = v.cpu().numpy()
            else:
                payload[k] = np.asarray(v)
        
        frames = [
            pd.DataFrame(payload["indices"], columns=['original_index']),
            pd.DataFrame(payload["y"], columns=self.datamodule.output_names),
            pd.DataFrame(payload["x"], columns=self.datamodule.input_names),
            pd.DataFrame(
                payload["pred"],
                columns=[f"{c}_pred" for c in self.datamodule.output_names],
            ),
            pd.DataFrame(
                payload["std"],
                columns=[f"{c}_std" for c in self.datamodule.output_names],
            ),
        ]

        if "mask" in payload:
            frames.append(
                pd.DataFrame(
                    payload["mask"],
                    columns=[f"{c}_valid" for c in self.datamodule.output_names],
                )
            )

        if "injection_label" in payload:
            frames.append(
                pd.DataFrame(
                    payload["injection_label"].reshape(-1, 1),
                    columns=["injection_label"],
                )
            )

        if "injection_probability" in payload:
            frames.append(
                pd.DataFrame(
                    payload["injection_probability"].reshape(-1, 1),
                    columns=["injection_probability"],
                )
            )

        df_payload = pd.concat(frames, axis=1)

        if save_path:
            df_payload.to_csv(save_path, index=False,)

        return df_payload

    def _record_metrics(
        self,
        split: str,
        epoch: int,
        metrics: dict[str, float],
        log_data: dict[str, float],
    ) -> None:
        """Store split metrics in the epoch log and persistent history."""
        for name, value in metrics.items():
            key = f"{split}/{name}"
            log_data[key] = value
            self.history[key].append((epoch, value))

    def print_epoch_summary(
        self,
        epoch: int,
        train_loss: float,
        validation_performed: bool,
        test_performed: bool,
    ) -> None:
        """logger.info one concise summary using the most recent validation result."""
        validation = self.last_validation_metrics
        
        test = self.last_test_metrics
        validation_age = "current" if validation_performed else "recent"

        sections = [
            f"Epoch {epoch:04d}/{self.cfg.training.epochs:04d}",
            f"train_loss={train_loss:.6f}",
        ]
        if validation is not None:
            sections.extend(
                [
                    f"val_loss={validation['loss']:.6f}",
                    f"val_MAE({validation_age})={validation.get('MAE', 'nan')}",
                    f"val_RMSE({validation_age})={validation.get('RMSE', 'nan')}",
                    f"val_NLL({validation_age})={validation.get('NLL', 'nan')}",
                    f"val_R2({validation_age})={validation.get('R2', 'nan')}",
                ]
            )
        else:
            sections.append("val_accuracy=not-yet-evaluated")

        if test_performed and test is not None:
            sections.extend(
                [
                    f"test_loss={test['loss']:.6f}",
                    f"test_MAE={test.get('MAE', 'nan'):.4f}",
                    f"test_RMSE={test.get('RMSE', 'nan'):.4f}",
                    f"test_NLL={test.get('NLL', 'nan'):.4f}",
                    f"test_R2={test.get('R2', 'nan'):.4f}",
                ]
            )

        sections.append(
            f"lr={self.optimizer.param_groups[0]['lr']:.3e}"
        )
        self.logger.info(", ".join(sections))

    def _plot_predictions(
        self,
        split: str,
        payload: dict[str, torch.Tensor],
        epoch: int,
    ) -> None:
        """Create publication-ready ground-truth versus prediction panels."""

        for k, v in payload.items():
            if isinstance(v, torch.Tensor):
                payload[k] = v.cpu().numpy()
            else:
                payload[k] = np.asarray(v)
            
        names = self.datamodule.output_names
        column_count = 3
        row_count = math.ceil(len(names) / column_count)
        figure, axes = plt.subplots(
            row_count,
            column_count,
            figsize=(5.2 * column_count, 4.6 * row_count),
        )
        axes = np.ravel(axes)

        for index, name in enumerate(names):
            axis = axes[index]
            truth = payload["y"][:, index]
            prediction = payload["pred"][:, index]
            lower = float(np.nanmin([truth, prediction]))
            upper = float(np.nanmax([truth, prediction]))

            axis.scatter(
                truth,
                prediction,
                s=18,
                alpha=0.65,
                edgecolors="none",
            )
            axis.plot(
                [lower, upper],
                [lower, upper],
                "k--",
                linewidth=1.2,
                label="Ideal",
            )
            axis.set_xlabel(f"Ground truth: {name}")
            axis.set_ylabel(f"Prediction: {name}")
            axis.set_title(name)
            axis.grid(alpha=0.25)
            axis.legend(frameon=False)

        for axis in axes[len(names):]:
            axis.axis("off")

        figure.suptitle(
            f"{split.capitalize()} predictions, epoch {epoch}",
            fontsize=15,
        )
        figure.tight_layout()
        path = (
            self.viz_dir / f"predictions_{split}_epoch_{epoch:04d}.png"
        )
        figure.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(figure)

        if self.wandb_run is not None:
            import wandb

            self.wandb_run.log(
                {
                    f"plots/{split}_predictions": wandb.Image(str(path)),
                    "epoch": epoch,
                }
            )

    def _plot_history(self) -> None:
        """Plot physical-unit RMSE and regression accuracy across epochs."""
        figure, axes = plt.subplots(1, 2, figsize=(13, 5))

        for split in ("train", "val", "test"):
            rmse_key = f"{split}/RMSE_mean"
            accuracy_key = f"{split}/accuracy"
            if rmse_key in self.history:
                points = self.history[rmse_key]
                axes[0].plot(
                    [epoch for epoch, _ in points],
                    [value for _, value in points],
                    label=split.capitalize(),
                    linewidth=2,
                )
            if accuracy_key in self.history:
                points = self.history[accuracy_key]
                axes[1].plot(
                    [epoch for epoch, _ in points],
                    [value for _, value in points],
                    label=split.capitalize(),
                    linewidth=2,
                )

        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Mean per-output RMSE")
        axes[0].set_title("Physical-unit RMSE")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Regression accuracy")
        axes[1].set_title("Accuracy = 1 / (1 + VRMSE)")

        for axis in axes:
            axis.grid(alpha=0.25)
            axis.legend(frameon=False)

        figure.tight_layout()
        figure.savefig(
            self.viz_dir / "training_history.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(figure)


    def restore_training_state(
        self,
        checkpoint_path: str,
    ) -> None:
        """Restore model, optimiser, scheduler and AMP state for resumption."""
        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device,
            weights_only=False,
        )
        self.model.load_state_dict(checkpoint["model_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
    
        scheduler_state = checkpoint.get("scheduler_state")
        if self.scheduler is not None and scheduler_state is not None:
            self.scheduler.load_state_dict(scheduler_state)
    
        grad_scaler_state = checkpoint.get("grad_scaler_state", None)
        if grad_scaler_state is not None:
            self.gradient_scaler.load_state_dict(grad_scaler_state)
        self.start_epoch = int(checkpoint["epoch"]) + 1
        
    def _save_checkpoint(
        self,
        epoch: int,
        filename: str,
        metrics: dict[str, float],
    ) -> None:
        """Save model, optimiser, preprocessing and training state atomically."""
        payload = {
            "epoch": epoch,
            "model_config": self.cfg.model,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": (
                self.scheduler.state_dict()
                if self.scheduler is not None
                else None
            ),
            "grad_scaler_state": self.gradient_scaler.state_dict(),
            "metrics": metrics,
            "input_scaler": self.datamodule.input_scaler,
            "output_scaler": self.datamodule.output_scaler,
            'input_scaler_state': self.datamodule.input_scaler.state_dict(),
            'output_scaler_state': self.datamodule.output_scaler.state_dict(),
            "target_transform": self.datamodule.target_transform,
            "input_names": self.datamodule.input_names,
            "output_names": self.datamodule.output_names,
        }
        self.atomic_torch_save(
            payload,
            self.checkpoint_dir / filename,
        )

    def _is_better(self, value: float, previous: float | None) -> bool:
        """Compare a monitored value according to the cfgured mode."""
        if previous is None:
            return True
        if self.cfg.training.monitor_mode == "min":
            return value < previous
        return value > previous


    def _evaluate_best_checkpoint(self) -> dict[str, float]:
        """Reload the validation-selected checkpoint and evaluate test data."""
        checkpoint_path = (
            self.checkpoint_dir / "best.pt"
        )
        if not checkpoint_path.exists():
            raise RuntimeError(
                "No best checkpoint exists. Ensure validation_interval allows "
                "at least one validation pass."
            )

        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device,
            weights_only=False,
        )
        self.model.load_state_dict(checkpoint["model_state"])
        epoch = int(checkpoint["epoch"])
        metrics, payload = self.evaluate("test", epoch)
        self._plot_predictions("test", payload, epoch)
        self.save_json(
            metrics,
            self.experiment_dir
            / "final_test_metrics.json",
        )

        if self.wandb_run is not None:
            self.wandb_run.log(
                {
                    f"final_test/{name}": value
                    for name, value in metrics.items()
                }
            )
        return metrics

    def atomic_torch_save(self, payload: Any, path: Path) -> None:
        """Write a PyTorch checkpoint atomically to avoid partial checkpoints."""
        import torch
    
        temporary_path = path.with_suffix(path.suffix + ".tmp")
        torch.save(payload, temporary_path)
        os.replace(temporary_path, path)
    
    
    def save_json(self, payload: Any, path: Path) -> None:
        """Write mappings containing NumPy scalar values as formatted JSON."""
        def convert(value: Any) -> Any:
            if np.isscalar(value):
                return float(value)
            return str(value)
    
        with path.open("w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, default=convert)

