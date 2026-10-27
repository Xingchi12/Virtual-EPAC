import sys
import os
import pandas as pd
from pathlib import Path


import utils
from predictor import Predictor

if __name__ == '__main__':

    input_file_path = '../data/test_selected_data.csv'
    output_file_tag = f"{os.path.basename(input_file_path).split('.')[0]}_enn_predictions"

    exp_name = Path('../experiments/exp_p3-multi_output-gaussian_nll-scaler_standard-ensembles10-random_kfold')

    out_dir = exp_name / 'predictions'
    out_dir.mkdir(parents=True, exist_ok=True)

    all_fold_predictions = []

    n_folds = 5

    for fold in range(n_folds):
        
        # Create a predictor for each fold models
        predictor=Predictor.from_experiment(
        exp_dir=exp_name / f'fold{fold}',
        checkpoint="checkpoints/best.pt",
        calibrate_std=True,
        calibration_method="scalar"
        )

        df_pred_file = predictor.predict_from_csv_file(
            input_file_path
        ).copy()

        # Stable identifier for the input row.
        df_pred_file.insert(
            0,
            "row_id",
            range(len(df_pred_file)),
        )

        # Record which fold model produced the prediction.
        df_pred_file["fold"] = fold

        output_file_path = (
            out_dir
            / f"{output_file_tag}_fold{fold}.csv"
        )

        df_pred_file.to_csv(
            output_file_path,
            index=False,
        )

        print(
            f"Fold predictions saved to {output_file_path}"
        )

        all_fold_predictions.append(df_pred_file)


    all_fold_predictions_df = pd.concat(
        all_fold_predictions,
        axis=0,
        ignore_index=True,
    )

    predictions_output_file = out_dir / f"{output_file_tag}_all_folds.csv"

    all_fold_predictions_df.to_csv(predictions_output_file, index=False)

    print("OOF predictions saved to:", predictions_output_file,)

    ## Aggregrate fold prediction using different strategy
    target_columns = predictor.output_names
    input_columns = predictor.input_names

    uniform_ensemble_df = utils.aggregate_uniform_fold_ensemble(
        all_fold_predictions_df=all_fold_predictions_df,
        target_columns=target_columns,
        input_columns=input_columns,
    )
    uniform_output_file = (out_dir / f"{output_file_tag}_uniform_ensemble.csv")
    uniform_ensemble_df.to_csv(uniform_output_file, index=False,)
    print("Uniform ensemble predictions saved to:", uniform_output_file,)


    lowest_std_df = utils.select_lowest_std_predictions(
        all_fold_predictions_df=all_fold_predictions_df,
        target_columns=target_columns,
        input_columns=input_columns,
    )
    lowest_std_output_file = (out_dir / f"{output_file_tag}_lowest_std.csv")
    lowest_std_df.to_csv(lowest_std_output_file, index=False)
    print("Lowest-std predictions saved to:", lowest_std_output_file)

    median_ensemble_df = utils.aggregate_median_predictions(
        all_fold_predictions_df=all_fold_predictions_df,
        target_columns=target_columns,
        input_columns=input_columns,
    )

    median_ensemble_output_file = (out_dir / f"{output_file_tag}_median_ensemble.csv")
    median_ensemble_df.to_csv(median_ensemble_output_file, index=False)
    print("Median predictions saved to:", median_ensemble_output_file)

