# TA2 Neural-Ensemble Simulator Surrogate

This project trains uncertainty-aware neural-network ensembles to replace the TA2 simulators with data-driven surrogate models. The five simulator controls are used as inputs, and the six measured electron and radiation quantities are learned as outputs.

## Input and output variables

### Inputs

1. `pulse_energy`
2. `zfoc`
3. `dazzler_second_order`
4. `density`
5. `plateau`

### Outputs

1. `measured_charge_fC`
2. `measured_energy_MeV`
3. `measured_transverse_density_fC_um2`
4. `rad_yield_total`
5. `rad_mean_energy_keV`
6. `rad_angular_density_ph_mrad2`

The data loader normalises column names automatically. For example, an Excel header named `pulse energy` is interpreted as `pulse_energy`.

## Project structure

Use the following directory layout:

```text
ta2-surrogate/
├── config.yaml
├── train.py
├── trainer.py
├── model.py
├── datasets.py
├── losses.py
├── metrics.py
├── utils.py
├── requirements.txt
├── README.md
├── data/
│   └── TA2_combined_forML.xlsx
└── experiments/
    └── generated automatically during training
```

The `data/` directory is the recommended location for the Excel workbook. Update `config.yaml` accordingly:

```yaml
paths:
  data_file: data/TA2_combined_forML.xlsx
  sheet_name: 0
```

`sheet_name: 0` selects the first worksheet.

## Installation

### 1. Create and activate a virtual environment

Python 3.10 or 3.11 is recommended.

#### Linux or macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

#### Windows PowerShell

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
```

### 2. Install PyTorch with CUDA 11.8 support

Install the requested pinned PyTorch build:

```bash
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu118
```

This package uses PyTorch for the neural networks and mixed-precision training. `torchvision` and `torchaudio` are included to keep the PyTorch package versions mutually compatible, although this tabular project does not directly use image or audio operations.

A compatible NVIDIA driver is required to use the CUDA 11.8 PyTorch wheels. The full CUDA Toolkit does not normally need to be installed separately because the wheel supplies its required CUDA runtime libraries.

If the machine has no compatible NVIDIA GPU, install the CPU build instead:

```bash
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cpu
```

Do not install both variants in the same environment.

### 3. Install the remaining dependencies

```bash
pip install -r requirements.txt
```

The remaining dependencies include Hydra, OmegaConf, Pandas, OpenPyXL, NumPy, Scikit-learn, Matplotlib and Weights & Biases.

### 4. Verify the installation

```bash
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA runtime:', torch.version.cuda); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

For the requested GPU installation, the expected PyTorch version begins with `2.7.1`, and `CUDA available` should be `True` on a correctly configured NVIDIA system.

## Configuration

Hydra reads the default settings from `config.yaml`. Configuration values can be changed either in the file or at the command line.

Important sections are:

- `experiment`: experiment name, output directory, seed, device and resume checkpoint.
- `wandb`: Weights & Biases project, mode, tags and account information.
- `columns`: the five inputs, six outputs and positive-output declarations.
- `data`: batch size, workers, scaling, target transformation and split strategy.
- `model`: architecture, ensemble size, hidden dimensions, branches and predictive variance.
- `loss`: deterministic or Gaussian objective and output weighting.
- `optimizer`: optimiser, learning rate, weight decay and gradient clipping.
- `scheduler`: learning-rate schedule.
- `training`: epoch count, mixed precision, validation frequency, logging, checkpointing and early stopping.

By default, positive outputs are transformed using

```text
y_transformed = log(1 + y)
```

and then standardised using statistics fitted only to the training subset. Predictions are inverse-transformed before physical-unit metrics and figures are produced.

## Implemented model families

Select a model using:

```yaml
model:
  architecture: shared_branches
```

### 1. One multi-output network

Configuration value:

```text
multi_output
```

A single multilayer perceptron learns a shared representation and predicts all six outputs through one output layer.

```text
Five inputs -> shared hidden layers -> six output distributions
```

This design has comparatively few parameters and can exploit common physical structure among the outputs. It can suffer from negative transfer if unrelated targets produce conflicting training gradients.

Training command:

```bash
python train.py model.architecture=multi_output experiment.name=multi_output_network
```

### 2. Six independent single-output networks

Configuration value:

```text
independent
```

A separate neural network is created for every target. The networks do not share hidden features.

```text
Five inputs -> network 1 -> measured_charge_fC
Five inputs -> network 2 -> measured_energy_MeV
Five inputs -> network 3 -> measured_transverse_density_fC_um2
Five inputs -> network 4 -> rad_yield_total
Five inputs -> network 5 -> rad_mean_energy_keV
Five inputs -> network 6 -> rad_angular_density_ph_mrad2
```

This avoids negative transfer and permits each target to learn a different functional representation. It uses more parameters and cannot exploit useful shared information between outputs.

Training command:

```bash
python train.py model.architecture=independent experiment.name=independent_outputs
```

### 3. Shared trunk with output-specific branches

Configuration value:

```text
shared_branches
```

The model first learns common latent features in a shared trunk. Each output then has a dedicated branch.

```text
                       -> charge branch
                       -> energy branch
Five inputs -> trunk   -> transverse-density branch
                       -> radiation-yield branch
                       -> mean-radiation-energy branch
                       -> angular-density branch
```

This architecture balances statistical efficiency with output-specific flexibility and is the default model family.

Training command:

```bash
python train.py model.architecture=shared_branches experiment.name=shared_output_branches
```

### 4. Physically grouped networks

Configuration value:

```text
grouped
```

The default groups are:

```yaml
model:
  groups:
    measured:
      - measured_charge_fC
      - measured_energy_MeV
      - measured_transverse_density_fC_um2
    radiation:
      - rad_yield_total
      - rad_mean_energy_keV
      - rad_angular_density_ph_mrad2
```

The model shares an initial trunk but uses one branch for measured electron quantities and another branch for radiation quantities. This encodes a weak physical inductive bias without forcing all targets to share the same final representation.

Training command:

```bash
python train.py model.architecture=grouped experiment.name=grouped_physics_outputs
```

## Ensemble and uncertainty model

Each selected model family is replicated `ensemble_size` times with independently initialised parameters. The ensemble prediction is the mean of the member predictions:

```text
ensemble_mean = mean(member_predictions)
```

When probabilistic prediction is enabled, every member estimates a mean and a diagonal Gaussian variance for each output. The final predictive variance combines:

- **Aleatoric variance:** uncertainty represented by each probabilistic member.
- **Epistemic variance:** disagreement among independently trained ensemble members.

Enable this formulation with:

```yaml
model:
  probabilistic: true

loss:
  name: gaussian_nll
```

A deterministic comparison can be run with:

```bash
python train.py model.probabilistic=false loss.name=mse experiment.name=deterministic_mse
```

## Training

### Default training run

From the project root:

```bash
python train.py
```

The default configuration uses:

- a five-member deep ensemble;
- a shared trunk with output-specific branches;
- spatially blocked splitting;
- robust input scaling;
- `log1p` transformation and standardisation of positive targets;
- Gaussian negative log-likelihood;
- AdamW optimisation;
- mixed precision on a CUDA device;
- gradient clipping;
- validation-based checkpoint selection;
- early stopping;
- offline Weights & Biases logging.

### Custom batch size, learning rate and weight decay

```bash
python train.py \
  data.batch_size=32 \
  optimizer.learning_rate=1e-4 \
  optimizer.weight_decay=5e-4 \
  experiment.name=small_batch_regularised
```

### Choose CPU or GPU

Automatic device selection:

```bash
python train.py experiment.device=auto
```

Require the first CUDA GPU:

```bash
python train.py experiment.device=cuda:0
```

Use the CPU:

```bash
python train.py experiment.device=cpu training.mixed_precision=false
```

### Online Weights & Biases logging

The default `offline` mode records results locally without requiring a network connection.

To use the hosted dashboard:

```bash
wandb login
python train.py wandb.mode=online wandb.project=ta2-surrogate experiment.name=online_run
```

To disable Weights & Biases:

```bash
python train.py wandb.enabled=false
```

## Cross-validation and split strategies

Three split strategies are available.

### Random K-fold

```bash
python train.py \
  data.split.strategy=random_kfold \
  data.split.n_splits=5 \
  data.split.fold=0 \
  experiment.name=random_fold_0
```

This measures interpolation performance but can be optimistic when nearby points appear in different subsets.

### Spatially blocked cross-validation

```bash
python train.py \
  data.split.strategy=spatial \
  data.split.n_splits=5 \
  data.split.fold=0 \
  data.split.spatial_bins=3 \
  experiment.name=spatial_fold_0
```

Multidimensional quantile cells are assigned intact to folds. This more directly tests prediction in held-out regions of the simulator design space.

### Cluster-based cross-validation

```bash
python train.py \
  data.split.strategy=cluster \
  data.split.n_splits=5 \
  data.split.fold=0 \
  experiment.name=cluster_fold_0
```

K-means clusters are formed in robustly scaled input space, and complete clusters are held out.

### Run all five folds

Linux or macOS:

```bash
for fold in 0 1 2 3 4; do
  python train.py \
    data.split.strategy=spatial \
    data.split.fold=$fold \
    experiment.name=spatial_fold_${fold}
done
```

## Evaluation

Validation and test evaluation are integrated into `train.py` through the `Trainer.evaluate` method.

During training:

- the validation subset is evaluated every `validation_interval` epochs;
- the test subset is evaluated every `test_interval` epochs for diagnostic reporting;
- checkpoint selection and early stopping use validation metrics only;
- after training, `best.pt` is reloaded and evaluated on the test subset;
- the test subset is never used to choose the best checkpoint.

Run training followed by final best-checkpoint evaluation with:

```bash
python train.py experiment.name=ta2_final_model
```

The final evaluation results are written to:

```text
experiments/<experiment-name_timestamp>/tables/final_test_metrics.json
```

The primary validation criterion is configured as:

```yaml
training:
  monitor: val/VRMSE
  monitor_mode: min
```

The reported metrics include:

- MSE for each output;
- RMSE for each output;
- MAE for each output;
- variance-normalised RMSE for each output;
- aggregate metrics across all outputs;
- Gaussian negative log-likelihood for probabilistic models;
- predictive-interval coverage.

### Evaluation frequency

```bash
python train.py \
  training.validation_interval=1 \
  training.test_interval=10 \
  training.prediction_plot_interval=25
```

### Resume training from a checkpoint

```bash
python train.py \
  experiment.resume=/absolute/path/to/experiments/run/checkpoints/last.pt \
  experiment.name=resumed_training
```

The checkpoint restores:

- model parameters;
- optimiser state;
- scheduler state;
- mixed-precision gradient scaler;
- completed epoch number.

## Experiment outputs

Every run creates a timestamped directory:

```text
experiments/
└── ta2_surrogate_YYYYMMDD_HHMMSS/
    ├── resolved_config.yaml
    ├── checkpoints/
    │   ├── best.pt
    │   ├── best_val_VRMSE.pt
    │   ├── best_val_RMSE_mean.pt
    │   └── last.pt
    ├── viz/
    │   ├── predictions_train_epoch_*.png
    │   ├── predictions_val_epoch_*.png
    │   ├── predictions_test_epoch_*.png
    │   └── rmse_history.png
    ├── history.json
    └── final_test_metrics.json
```

The prediction figures place ground truth on the horizontal axis and predicted values on the vertical axis. A dashed identity line represents perfect prediction. Figures are exported at 300 DPI.

## Recommended experiment sequence

Run the same spatial folds for all four architecture families:

```bash
python train.py model.architecture=multi_output data.split.strategy=spatial experiment.name=multi_output
python train.py model.architecture=independent data.split.strategy=spatial experiment.name=independent
python train.py model.architecture=shared_branches data.split.strategy=spatial experiment.name=shared_branches
python train.py model.architecture=grouped data.split.strategy=spatial experiment.name=grouped
```

Do not select an architecture from one random split. Compare models using repeated or complete cross-validation and examine:

- mean and standard deviation across folds;
- every output separately;
- high-output and boundary regimes;
- uncertainty coverage;
- stability across seeds;
- error in spatially or cluster-held-out regions.

## Reproducibility notes

- Random seeds are configured under `experiment.seed`.
- Deterministic PyTorch behaviour is requested by default.
- Data scalers are fitted on training data only.
- Missing rows are dropped before splitting when `drop_missing: true`.
- Checkpoints include the fitted preprocessing objects and output names.
- Results may still differ slightly across GPU models, CUDA kernels and PyTorch builds.

## Common problems

### Excel file not found

Check that the workbook exists at:

```text
data/TA2_combined_forML.xlsx
```

and that `config.yaml` contains:

```yaml
paths:
  data_file: data/TA2_combined_forML.xlsx
```

An absolute path can also be supplied:

```bash
python train.py paths.data_file=/absolute/path/to/TA2_combined_forML.xlsx
```

### CUDA is not detected

Run:

```bash
nvidia-smi
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

If `torch.cuda.is_available()` is `False`, verify the NVIDIA driver and ensure that the CUDA 11.8 PyTorch wheel, rather than the CPU wheel, is installed.

### Out-of-memory error

Reduce the batch size or ensemble size:

```bash
python train.py data.batch_size=16 model.ensemble_size=3
```

### Weights & Biases authentication error

Use offline mode:

```bash
python train.py wandb.mode=offline
```

or disable logging:

```bash
python train.py wandb.enabled=false
```

## Terminal training progress

The trainer uses `tqdm` for every training, validation and test pass.
During training, the progress bar displays the running loss and learning rate.
During validation and testing, it displays the running loss while predictions
are collected, followed by the final regression accuracy.

This is a regression problem, so ordinary classification accuracy is not
defined. The displayed progress score is:

```text
regression_accuracy = 1 / (1 + VRMSE)
```

It has the following interpretation:

- `1.0` means perfect prediction.
- `0.5` means the mean VRMSE is `1.0`.
- Values closer to zero indicate poorer predictions.

After every epoch, the terminal prints training loss and the most recently
computed validation accuracy. If validation is not scheduled for that epoch,
the summary labels the value as `recent`. Test loss and accuracy are printed
only on scheduled test epochs.

Example output:

```text
Epoch 0021/0500 | train_loss=-0.183421 | val_loss=-0.142311 |
val_accuracy(current)=0.7134 | lr=3.000e-04
```

A Gaussian negative log-likelihood can be negative when the predicted
variance is small and the residual is sufficiently small. This is valid and
does not indicate a failure.

## Deterministic CUDA execution

The entry point sets the following variable before importing PyTorch:

```text
CUBLAS_WORKSPACE_CONFIG=:4096:8
```

This addresses the cuBLAS reproducibility warning produced when deterministic
PyTorch algorithms are enabled with CUDA 10.2 or newer. It must be set before
CUDA creates a cuBLAS handle. The script uses `os.environ.setdefault`, so an
explicit value already provided by the shell is preserved.

To set it manually on Linux before launching training:

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python train.py
```

The smaller supported workspace setting can be selected when GPU memory is
particularly constrained:

```bash
export CUBLAS_WORKSPACE_CONFIG=:16:8
python train.py
```
