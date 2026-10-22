# vEPAC Multitask GP: Workflow & Execution Structure

This document outlines the structure, data loading, artifact storage, and execution flow for the two notebooks in `vepac_multitaskgp/`.

---

## 1. Pipeline Overview

```
[ data/merged_all.csv ] ──> 01_training.ipynb ──> Saves Artifacts (.pkl, .pth, .pt)
                                                             │
[ data/test_selected_data.csv ] ──────────────> 02_inference.ipynb ──> Saves Results to results/
```

---

## 2. Notebook 1: Training (`01_training.ipynb`)

* **Purpose**: Ingest data, train the Multitask ExactGP model, and save model artifacts.
* **Input Data**:
  * Source file: `data/merged_all.csv` (1,900 raw runs $\rightarrow$ 1,318 valid non-zero runs).
  * 5 Inputs: `l_energy_J__req_sim`, `l_focus_z_mm__req_sim`, `l_gdd_fs2__req_sim`, `g_density__req_sim`, `g_plateau_mm__req_sim`.
  * 4 Targets: `e_charge_fC`, `x_yield_ph`, `e_energy_mean_MeV`, `x_energy_mean_keV`.
* **Execution Steps**:
  1. Filters valid simulation runs and splits 80% train (1,054 runs) / 20% test (264 runs).
  2. Standardizes inputs and outputs via `StandardScaler`.
  3. Trains `MultitaskExactGP` with Matérn 5/2 kernel + Rank-2 coregionalization for 120 iterations using Adam (lr = 0.05).
* **Saved Artifacts**:
  * `scaler_x.pkl`, `scaler_y.pkl`: Fitted feature and target scalers.
  * `multitask_gp_model.pth`, `multitask_gp_likelihood.pth`: Trained PyTorch state dicts.
  * `train_x.pt`, `train_y.pt`: Training tensors required by ExactGP conditioning.
  * `test_x.pt`, `test_y.pt`: Holdout test tensors.
  * `model_metadata.pkl`: Input/output column names and task metadata.

---

## 3. Notebook 2: Inference (`02_inference.ipynb`)

* **Purpose**: Load saved model artifacts and perform inference and evaluation.
* **Loading Step**:
  1. Loads `scaler_x.pkl`, `scaler_y.pkl`, and `model_metadata.pkl`.
  2. Loads `train_x.pt` and `train_y.pt` to initialize `MultitaskGPModel`.
  3. Loads weights from `multitask_gp_model.pth` and `multitask_gp_likelihood.pth`.
  4. Sets model and likelihood to `eval()` mode.
* **Inference Tasks**:
  1. **Holdout Test Set (264 runs)**: Evaluates predictions against ground truth and computes MAE, RMSE, R², and Interval ECE.
  2. **Unseen 5th Set (`data/test_selected_data.csv`, 594 runs)**: Standardizes inputs with `scaler_x`, generates predictions with uncertainty bounds, and inverse-transforms to physical units.

---

## 4. Results Storage (`results/`)

All outputs are saved directly under `vepac_multitaskgp/results/`:

| Output File | Description | Source Notebook |
| :--- | :--- | :--- |
| `training_loss.png` | Negative MLL loss convergence curve | `01_training.ipynb` |
| `parity_plots.png` | Parity plots (actual vs. predicted) for 264 test runs | `02_inference.ipynb` |
| `test_metrics_summary.csv` | MAE, RMSE, R², and ECE metrics table for test set | `02_inference.ipynb` |
| `set5_parity_plots.png` | Parity plots for the unseen 594 samples (5th set) | `02_inference.ipynb` |
| `set5_metrics_summary.csv` | MAE, RMSE, R², and ECE metrics table for 5th set | `02_inference.ipynb` |
| `test_selected_data_with_predictions.csv` | Augmented 5th set dataset with GP prediction columns | `02_inference.ipynb` |
