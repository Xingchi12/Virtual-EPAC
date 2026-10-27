# vEPAC Surrogates using Ensemble Neural Network

For details about models architecture, see `ensemble_nn/README.md`

## Setup

```bash
pip install -r ensemble_nn/requirements.txt
```

## Reproducing the results

Training is a Hydra entry point run from inside `ensemble_nn/`:

```bash
cd ensemble_nn
python train.py wandb.enabled=True data.output_transform=none experiment.name=exp_vepac
```

This trains 5-fold cross-validation models and writes checkpoints, configs, and metrics to `experiments/<experiment.name>-<run_tag>/`.

## Predicting from a CSV file

1. Open `ensemble_nn/predict_from_csv.py` and set:
   - `exp_name` to the experiment folder, e.g.
     `../experiments/<experiment.name>-<run_tag>`
   - `n_folds` to the number of folds in that experiment
   - `input_file_path` to the CSV file to predict on

2. Run it from inside `ensemble_nn/`:

```bash
cd ensemble_nn
python predict_from_csv.py
```
The script will save the following files inside
`<exp_name>/predictions/`:

| Files | Point prediction | Reported `_std` |
|------|-------------------|------------------|
| `..._fold{N}.csv` | Fold `N`'s prediction, evaluated on every row | Fold `N`'s own aleatoric std |
| `..._all_folds.csv` (OOF) | Each row predicted once, by the fold that held it out as test | Corresponding fold's aleatoric std |
| `..._uniform_ensemble.csv` | Mean of all `n_folds` predictions | `sqrt(mean of within-fold variances + variance across fold means)` — aleatoric and epistemic combined |
| `..._lowest_std.csv` | Prediction from whichever single fold reports the lowest std for that row | Corresponding fold's std (the minimum across folds) |
| `..._median_ensemble.csv` | Median of all `n_folds` predictions | `sqrt(mean of within-fold variances)` only — aleatoric, robust to any one outlier fold |

- **OOF (out-of-fold) predictions** are the standard way to evaluate the
  surrogate: concatenating each fold's predictions with no ensembling across folds (one prediction per row from a fold model).
- **Uniform ensemble**  averages every fold model's prediction and       reports the full predictive uncertainty.
- **Lowest-std** picks, independently per row and per target, the single
  fold model that claims to be most confident. Used to inspect best-case accuracy.
- **Median ensemble** robust alternative to the uniform ensemble:
  the point prediction is the per-row median across folds rather than
  the mean, so a single outlier fold cannot pull it off.
