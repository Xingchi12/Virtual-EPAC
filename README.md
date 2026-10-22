# vEPAC: Laser Wakefield Accelerator Multitask GP Surrogate Model

This repository provides an end-to-end, self-contained **Multitask Gaussian Process (`MultitaskExactGP`)** surrogate model developed for the vEPAC (cEPAC) project using `GPyTorch`.

It maps 5 laser and target control parameters to 4 key Tier-1 & Tier-2 accelerator and radiation scalars simultaneously, capturing the underlying physical correlations between electron beam properties and betatron X-ray emission.

---

## 📁 Repository Structure

All self-contained scripts, datasets, pre-trained models, and results are located in [`vepac_multitaskgp/`](./vepac_multitaskgp/):

* **[`01_training.ipynb`](./vepac_multitaskgp/01_training.ipynb)**: Data ingestion, feature standardization, training loop (120 iterations via Adam), and model checkpointing.
* **[`02_inference.ipynb`](./vepac_multitaskgp/02_inference.ipynb)**: Checkpoint loading, test set evaluation (264 holdout runs), unseen 5th set prediction (594 runs), and parity plot generation.
* **[`03_examples_loading_and_inference.md`](./vepac_multitaskgp/03_examples_loading_and_inference.md)**: Detailed execution workflow and results file inventory.
* **[`data/`](./vepac_multitaskgp/data/)**: Self-contained datasets (`merged_all.csv` and `test_selected_data.csv`).
* **[`results/`](./vepac_multitaskgp/results/)**: Parity plots, training loss curve, and evaluation metrics summaries.
* **Model Checkpoints**: `multitask_gp_model.pth`, `multitask_gp_likelihood.pth`, `scaler_x.pkl`, `scaler_y.pkl`, `model_metadata.pkl`, and PyTorch conditioning tensors (`train_x.pt`, etc.).

---

## 🔬 Parameters Overview

### Inputs (5 Control Features)
1. `l_energy_J__req_sim`: Laser pulse energy ($J$)
2. `l_focus_z_mm__req_sim`: Laser focal position $z_{\text{foc}}$ relative to gas target ($\text{mm}$)
3. `l_gdd_fs2__req_sim`: Group delay dispersion / chirp ($\text{fs}^2$)
4. `g_density__req_sim`: Target plasma density ($10^{18}\text{ cm}^{-3}$)
5. `g_plateau_mm__req_sim`: Gas target plateau length ($\text{mm}$)

### Outputs (4 Target Observables)
1. `e_charge_fC`: Total electron beam charge ($\text{fC}$) - *Tier 1*
2. `x_yield_ph`: Total betatron X-ray photon yield ($\text{photons}$) - *Tier 1*
3. `e_energy_mean_MeV`: Mean electron beam energy ($\text{MeV}$) - *Tier 2*
4. `x_energy_mean_keV`: Mean X-ray photon energy ($\text{keV}$) - *Tier 2*

---

## ⚡ Quick Start

### 1. Requirements
Ensure you have Python 3.9+ with:
```bash
pip install torch gpytorch numpy pandas scikit-learn matplotlib
```

### 2. Run Inference
Open [`vepac_multitaskgp/02_inference.ipynb`](./vepac_multitaskgp/02_inference.ipynb) in Jupyter. It loads the pre-trained weights and pre-computed scalers directly, producing evaluation metrics and parity plots without requiring re-training.

### 3. Run Training
Open [`vepac_multitaskgp/01_training.ipynb`](./vepac_multitaskgp/01_training.ipynb) to train the model from scratch on `vepac_multitaskgp/data/merged_all.csv`.
