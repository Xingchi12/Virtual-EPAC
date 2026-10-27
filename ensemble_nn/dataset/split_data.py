from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.model_selection import KFold
from sklearn.preprocessing import RobustScaler

from dataset.datamodule import canonical_name

class DataPreprocessorSplitter:

    def __init__(self, cfg):

        self.cfg = cfg

        self.input_columns = [
            canonical_name(c)
            for c in cfg.data.inputs
        ]

        self.output_columns = [
            canonical_name(c)
            for c in cfg.data.outputs
        ]

        self.selected_columns = (
            self.output_columns +
            self.input_columns
        )

        # The output whose positivity defines whether the underlying
        # physical process "succeeded" (for example whether any charge
        # was produced at all). Rows below this threshold leave the
        # dependent quantities (such as a mean particle energy)
        # physically undefined rather than merely noisy, so they must be
        # tracked explicitly instead of silently discarded.
        self.injection_output = canonical_name(
            getattr(
                cfg.data,
                "injection_output",
                self.output_columns[0],
            )
        )

        self.mask_columns = [
            f"{col}_valid"
            for col in self.output_columns
        ]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self):

        df = self.load_data()

        df = self.clean_data(df)

        fold_labels = self.create_fold_labels(
            df[self.input_columns].to_numpy(
                dtype=float
            )
        )

        self.save_all_splits(
            df,
            fold_labels
        )

    # ------------------------------------------------------------------
    # Load Data
    # ------------------------------------------------------------------

    def load_data(self):

        data_file = Path(
            self.cfg.data.raw_file
        )

        if not data_file.exists():

            raise FileNotFoundError(
                data_file.resolve()
            )

        if data_file.suffix.lower() == ".csv":

            df = pd.read_csv(
                data_file
            )

        elif data_file.suffix.lower() in {
            ".xlsx",
            ".xls",
        }:

            df = pd.read_excel(
                data_file,
                engine="openpyxl",
            )

        else:

            raise ValueError(
                f"Unsupported file type: {data_file}"
            )

        df.columns = [
            canonical_name(c)
            for c in df.columns
        ]

        return df

    # ------------------------------------------------------------------
    # Clean Data
    # ------------------------------------------------------------------

    def clean_data(
        self,
        df: pd.DataFrame,
        keep_columns = ['sample_id', 'dataset']
    ):

        missing = [
            col
            for col in self.selected_columns
            if col not in df.columns
        ]

        if missing:

            raise ValueError(
                f"Missing columns: {missing}"
            )

        keep = []
        for col in df.columns:
            if col in keep_columns:
                keep.append(col)

        
        keep.extend(
            self.selected_columns
        
        )

        df = df[
            keep
        ].copy()

        df["original_index"] = np.arange(
            len(df)
        )

        df[self.selected_columns] = (
            df[self.selected_columns]
            .apply(
                pd.to_numeric,
                errors="coerce",
            )
        )

        #
        # Inputs are physical control settings and are expected to be
        # always present. Rows missing an input (or set to exactly zero,
        # which is not a valid control value for this simulator) cannot
        # be used for training or evaluation at all, so they are dropped.
        #
        df = df.dropna(
            subset=self.input_columns
        )

        input_nonzero = (
            df[self.input_columns] != 0
        ).all(axis=1)

        df = df.loc[input_nonzero]

        #
        # Outputs may be legitimately undefined for some rows (for
        # example a mean particle energy when no particles were
        # produced). Rather than dropping those rows -- which would
        # silently discard the physically meaningful "no injection"
        # samples and starve the fully defined outputs of ~27% of their
        # training signal -- keep every row and record, per output,
        # whether its value is valid. Downstream masked losses/metrics
        # use this to ignore undefined entries without discarding the
        # row entirely.
        #
        for output_column, mask_column in zip(
            self.output_columns,
            self.mask_columns,
        ):
            df[mask_column] = (
                df[output_column]
                .notna()
                .astype(float)
            )

        #
        # Binary label recording whether the underlying process
        # "succeeded" (e.g. any charge was produced). This is always
        # well defined -- unlike the dependent quantities it gates --
        # and is used to train an auxiliary classifier that lets the
        # surrogate represent the success/failure threshold explicitly.
        #
        df["injection_label"] = (
            df[self.injection_output] > 0
        ).astype(float)

        df = df.reset_index(
            drop=True
        )

        print(
            f"Rows after cleaning: {len(df)}"
        )

        for output_column, mask_column in zip(
            self.output_columns,
            self.mask_columns,
        ):
            valid_count = int(df[mask_column].sum())
            print(
                f"  {output_column}: "
                f"{valid_count}/{len(df)} rows valid"
            )

        print(
            f"  injection_label positive rate: "
            f"{df['injection_label'].mean():.4f}"
        )

        return df

    # ------------------------------------------------------------------
    # Fold Generation
    # ------------------------------------------------------------------

    def create_fold_labels(
        self,
        inputs: np.ndarray,
    ):

        split_cfg = self.cfg.data.split

        scaled_inputs = (
            RobustScaler()
            .fit_transform(inputs)
        )

        return self.assign_folds(
            scaled_inputs=scaled_inputs,
            strategy=split_cfg.strategy,
            n_splits=split_cfg.n_splits,
            seed=self.cfg.experiment.seed,
            shuffle=split_cfg.shuffle,
            spatial_bins=split_cfg.spatial_bins,
            cluster_n_init=split_cfg.cluster_n_init,
        )

    def assign_folds(
        self,
        scaled_inputs,
        strategy,
        n_splits,
        seed,
        shuffle,
        spatial_bins,
        cluster_n_init,
    ):

        n_samples = len(
            scaled_inputs
        )

        if strategy == "random_kfold":

            labels = np.empty(
                n_samples,
                dtype=int,
            )

            splitter = KFold(
                n_splits=n_splits,
                shuffle=shuffle,
                random_state=(
                    seed if shuffle else None
                ),
            )

            for fold, (_, idx) in enumerate(
                splitter.split(
                    scaled_inputs
                )
            ):
                labels[idx] = fold

            return labels

        if strategy == "cluster_kfold":

            model = KMeans(
                n_clusters=n_splits,
                random_state=seed,
                n_init=cluster_n_init,
            )

            return model.fit_predict(
                scaled_inputs
            )

        if strategy == "spatial_kfold":

            binned = []

            for col in range(
                scaled_inputs.shape[1]
            ):

                bins = pd.qcut(
                    scaled_inputs[:, col],
                    q=spatial_bins,
                    labels=False,
                    duplicates="drop",
                )

                binned.append(
                    np.asarray(
                        bins,
                        dtype=int,
                    )
                )

            cells = np.column_stack(
                binned
            )

            _, cell_ids = np.unique(
                cells,
                axis=0,
                return_inverse=True,
            )

            counts = np.bincount(
                cell_ids
            )

            fold_load = np.zeros(
                n_splits,
                dtype=int,
            )

            cell_fold = np.zeros(
                len(counts),
                dtype=int,
            )

            rng = np.random.default_rng(
                seed
            )

            for cell in np.argsort(
                -counts
            ):

                candidate_folds = np.flatnonzero(
                    fold_load ==
                    fold_load.min()
                )

                fold = int(
                    rng.choice(
                        candidate_folds
                    )
                )

                cell_fold[cell] = fold

                fold_load[fold] += (
                    counts[cell]
                )

            return (
                cell_fold[cell_ids]
            )

        raise ValueError(
            f"Unknown split strategy: {strategy}"
        )

    # ------------------------------------------------------------------
    # Save Splits
    # ------------------------------------------------------------------

    def save_all_splits(
        self,
        df: pd.DataFrame,
        fold_labels: np.ndarray,
    ):

        output_root = Path(
            self.cfg.data.data_dir
        )

        output_root.mkdir(
            parents=True,
            exist_ok=True,
        )

        n_splits = (
            self.cfg.data.split.n_splits
        )

        strategy = (
            self.cfg.data.split.strategy
        )
        
        for val_fold in range(
            n_splits
        ):

            test_fold = (
                val_fold + 1
            ) % n_splits

            train_mask = (
                (fold_labels != val_fold)
                &
                (fold_labels != test_fold)
            )

            train_df = df.loc[
                train_mask
            ]

            val_df = df.loc[
                fold_labels == val_fold
            ]

            test_df = df.loc[
                fold_labels == test_fold
            ]

            fold_dir = (
                output_root / 
                strategy /
                f"fold{val_fold}"
            )

            fold_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            train_df.to_csv(
                fold_dir / "train.csv",
                index=False,
            )

            val_df.to_csv(
                fold_dir / "val.csv",
                index=False,
            )

            test_df.to_csv(
                fold_dir / "test.csv",
                index=False,
            )

            print(
                f"\nfold{val_fold}"
            )

            print(
                f"train={len(train_df)}"
            )

            print(
                f"val={len(val_df)}"
            )

            print(
                f"test={len(test_df)}"
            )
