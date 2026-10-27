# vEPAC Surrogate Models

Neural-network surrogates for laser-wakefield accelerator outputs
(`e_charge_fC`, `x_yield_ph`, `e_energy_mean_MeV`, `x_energy_mean_keV`).

## Setup

```bash
pip install -r src/requirements.txt
```

## Reproducing the results

Training is a Hydra entry point run from inside `src/`:

```bash
cd src
python train.py wandb.enabled=True data.output_transform=none experiment.name=exp_p3
```

This trains every cross-validation fold and writes checkpoints, configs,
and metrics to `experiments/<experiment.name>-<run_tag>/`.

## Predicting from a CSV file

1. Open `predict_from_csv.py` and set:
   - `exp_name` to the experiment folder under `experiments/`
   - `n_folds` to the number of folds in that experiment
   - `input_file_path` to the CSV file to predict on
2. Run it from the repo root:

```bash
python predict_from_csv.py
```

Per-fold predictions and fold-ensemble aggregates are written to
`experiments/<exp_name>/predictions/`.
