from sklearn.preprocessing import StandardScaler
import pandas as pd
import os
import numpy as np
import warnings


class DataFetcher:
    """
    assumes default data shape of n x t with n number of samples and t time-steps
    """

    def __init__(self, load_config: dict):
        self.config = load_config
        self._process_config()

    def _process_config(self):
        self.normal_path = self.config["normal_data_path"]
        self.anom_path = self.config["anom_data_path"]
        self.meta_path = self.config.get("meta_data_path", None)

    def load_data(self):
        """
        loads x, y, and meta data (if available else None)
        """
        x = self.read_file(self.normal_path)
        y = self.read_file(self.anom_path)
        meta = None
        if self.meta_path is not None:
            meta = self.read_file(self.meta_path)
        return x, y, meta

    def read_file(self, path):
        """
        Reads a file, deciding whether to use read_csv or read_parquet based on the file extension.
        """
        _, extension = os.path.splitext(path)
        if extension == ".csv":
            return pd.read_csv(path)
        elif extension == ".parquet":
            return pd.read_parquet(path)
        else:
            raise ValueError(f"Unsupported file extension: {extension}")


class DataSplitter:
    """
    Splits data into train, val, and test sets, compatible with metadata
    """

    def __init__(self, split_config: dict):
        self.config = split_config
        self._process_config()

    def _process_config(self):
        self.train_idx = self.config["train_end_idx"]
        self.val_idx = self.config["val_end_idx"]
        self.test_idx = self.config.get("test_end_idx", None)

    def split_data(self, x, y, meta=None):
        train, val, test = self._create_data_splits(x, y, meta)
        return train, val, test

    def _create_data_splits(self, x, y, meta):
        x_train, x_val, x_test = self._split_dataframe(x)
        y_train, y_val, y_test = self._split_dataframe(y)

        if meta is not None:
            meta_train, meta_val, meta_test = self._split_dataframe(meta)
        else:
            meta_train, meta_val, meta_test = None, None, None

        return (
            (x_train, y_train, meta_train),
            (x_val, y_val, meta_val),
            (x_test, y_test, meta_test),
        )

    def _split_dataframe(self, df):
        train_df = df.iloc[: self.train_idx, :].copy()
        val_df = df.iloc[self.train_idx + 1 : self.val_idx, :].copy()
        test_df = df.iloc[self.val_idx : self.test_idx, :].copy() if self.test_idx else df.iloc[self.val_idx:, :].copy()
        return train_df.values, val_df.values, test_df.values


class DataScaler:
    """
    Work in progress: currently *not used* as data is directly scaled in get_data.ipynb
    """

    def __init__(self, scale_config: dict):
        self.config = scale_config
        self._process_config()

    def _process_config(self):
        if self.config is not None:
            self.scaling_type = self.config["scaling_type"]
            self._validate_scaling_type()
            self.scale_meta_idx = self.config["scale_meta_idx"]
        else:
            warnings.warn("No scaling config was provided, no scaling will be applied")

    def _validate_scaling_type(self):
        allowed_scaling_types = ["global", "individual"]
        if self.scaling_type not in allowed_scaling_types:
            raise ValueError(
                f"scaling type should be one of {allowed_scaling_types}, but {self.scaling_type} was given"
            )

    def scale_temporal_data(self, train, val, test):
        if self.scaling_type == "global":
            train_scaled, val_scaled, test_scaled, scaler = self._scale_global(
                train, val, test
            )
        elif self.scaling_type == "individual":
            train_scaled, val_scaled, test_scaled, scaler = self._scale_individual(
                train, val, test
            )
        return (train_scaled, val_scaled, test_scaled), scaler

    def scale_metadata(self, train, val, test):
        if self.scale_meta_idx is not None:
            return self._scale_metadata(
                train.iloc[:, self.scale_meta_idx].values,
                val.iloc[:, self.scale_meta_idx].values,
                test.iloc[:, self.scale_meta_idx].values,
            )
        else:
            return train, val, test, None

    def _scale_metadata(self, metadata_train, metadata_val, metadata_test):
        meta_scaler = StandardScaler()
        metadata_train = meta_scaler.fit_transform(metadata_train)
        metadata_val = meta_scaler.transform(metadata_val)
        metadata_test = meta_scaler.transform(metadata_test)
        return metadata_train, metadata_val, metadata_test, meta_scaler

    def _scale_global_temporal(self, train, val, test):
        global_scaler = StandardScaler()
        flattened_series_train = train.values.flatten().reshape(-1, 1)
        train_scaled = global_scaler.fit_transform(flattened_series_train).reshape(
            -1, train.shape[1]
        )
        flattened_series_val = val.values.flatten().reshape(-1, 1)
        val_scaled = global_scaler.transform(flattened_series_val).reshape(
            -1, val.shape[1]
        )
        flattened_series_test = test.values.flatten().reshape(-1, 1)
        test_scaled = global_scaler.transform(flattened_series_test).reshape(
            -1, test.shape[1]
        )
        return train_scaled, val_scaled, test_scaled, global_scaler

    def _scale_individual_temporal(self, train, val, test):
        train, val, test = train.values, val.values, test.values

        mean_train = np.mean(train, axis=1, keepdims=True)
        std_train = np.std(train, axis=1, keepdims=True)
        train_scaled = (train - mean_train) / std_train

        mean_val = np.mean(val, axis=1, keepdims=True)
        std_val = np.std(val, axis=1, keepdims=True)
        val_scaled = (val - mean_val) / std_val

        mean_test = np.mean(test, axis=1, keepdims=True)
        std_test = np.std(test, axis=1, keepdims=True)
        test_scaled = (test - mean_test) / std_test

        scaler = {
            "train": {"mean": mean_train, "std": std_train},
            "val": {"mean": mean_val, "std": std_val},
            "test": {"mean": mean_test, "std": std_test},
        }
        return train_scaled, val_scaled, test_scaled, scaler
