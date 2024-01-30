from typing import Any
import numpy as np
import os
import pandas as pd
import argparse


def mix_anomalies(cfg, splitted_data):
    """
    Adds anomalies to validation and test input data plus replaces y to binary y-labels.
    By default only adds anomalies to validation and test data, but can also include training data.
    """
    splitted_data = list(splitted_data)
    idx = cfg["mix_idx"]
    for i in idx:
        splitted_data[i] = _add_labeled_anomalies(splitted_data[i], cfg["mix_ratio"])
    return splitted_data


def _add_labeled_anomalies(data, n_samples_frac):
    x, y, meta = data
    n_samples = int(n_samples_frac * y.shape[0])
    selected_rows = np.random.choice(y.shape[0], size=n_samples, replace=False)
    x, y = _replace_rows(x, y, selected_rows)
    return x, y, meta


def _replace_rows(x, y, selected_rows):
    x_size = x.shape[0]
    x[selected_rows, :] = y[selected_rows, :]
    y = np.zeros((x_size, 1))
    y[selected_rows, :] = 1
    return x, y


class AnomFuncs:
    """
    Class that contains functions to inject anomalies into time series data
    """

    def __init__(self, df, save_path="data_store", seed=96) -> None:
        self.seed = seed
        self.rng = np.random.default_rng(self.seed)
        self.df = df
        self.np_data = np.copy(self.df.values)
        self.save_path = save_path

    def _reset_data(self):
        self.np_data = np.copy(self.df.values)

    def _reset_rng(self):
        rng = np.random.default_rng(self.seed)
        # assert rng.random(1) == self.rng.random(1)
        return rng

    @staticmethod
    def inject_platform(ts_row, platform, start, length):
        ts_row[start : start + length] = platform
        return ts_row

    @staticmethod
    def inject_trend(ts_row, slope, start, length):
        ts_row[start : start + length] += +np.arange(0, length) * slope
        return ts_row

    @staticmethod
    def inject_mean(ts_row, mean, start, length):
        ts_row[start : start + length] += mean
        return ts_row

    @staticmethod
    def inject_extremum(ts_row, extremum, pos):
        ts_row[pos] += extremum
        return ts_row

    @staticmethod
    def inject_pattern(ts_row, pattern_param, start, length, pattern="sine"):
        if pattern == "sine":
            ts_row[start : start + length] = AnomFuncs._sine_wave(
                np.arange(0, length), amp=pattern_param
            )
        return ts_row

    @staticmethod
    def _sine_wave(x, amp, freq=0.01, phase=0):
        return amp * np.sin(2 * np.pi * freq * x + phase)

    def generate_anomalies(self, anomaly_type, anomaly_params, name):
        """
        :param anomaly_type: str, one of ['platform', 'trend', 'mean', 'extremum', 'pattern']
        :param anomaly_params: dict or list of dicts, anomaly parameters
        :param name: str, name of the dataset
        :param meta_idx: int, index of the *variable* meta data
        """
        print(f"Generating {anomaly_type} anomalies in {self.save_path}/{name}/")
        os.makedirs(f"{self.save_path}/{name}", exist_ok=True)
        anomaly_params = anomaly_params.copy()
        self._reset_data()
        self.rng = self._reset_rng()
        anomaly_params_processed, meta_keys = self._process_params(
            anomaly_params, self.np_data.shape, self.rng
        )
        for i in range(self.np_data.shape[0]):
            if self.contains_variable_params:
                anomaly_params = {
                    key: value[i] for key, value in anomaly_params_processed.items()
                }
            if anomaly_type == "platform":
                self.np_data[i, :] = self.inject_platform(
                    self.np_data[i, :], **anomaly_params
                )
            elif anomaly_type == "trend":
                self.np_data[i, :] = self.inject_trend(
                    self.np_data[i, :], **anomaly_params
                )
            elif anomaly_type == "mean":
                self.np_data[i, :] = self.inject_mean(
                    self.np_data[i, :], **anomaly_params
                )
            elif anomaly_type == "extremum":
                self.np_data[i, :] = self.inject_extremum(
                    self.np_data[i, :], **anomaly_params
                )
            elif anomaly_type == "pattern":
                self.np_data[i, :] = self.inject_pattern(
                    self.np_data[i, :], **anomaly_params
                )
            else:
                raise ValueError(
                    f"anomaly_type must be one of ['platform', 'trend', 'mean', 'extremum', 'pattern'], but {anomaly_type} was given."
                )
        if meta_keys != []:
            # Save the meta data
            meta_df = pd.DataFrame(
                {key: anomaly_params_processed[key] for key in meta_keys}
            )
            # normalize length and start position
            if "length" in meta_keys:
                print("Normalizing length and/or start position")
                meta_df["length"] = meta_df["length"] / self.np_data.shape[1]
            if "start" in meta_keys:
                meta_df["start"] = meta_df["start"] / self.np_data.shape[1]
            meta_df.to_parquet(
                f"{self.save_path}/{name}/meta_data.parquet", index=False
            )
            print(f"Meta data saved to {self.save_path}/{name}/meta_data.parquet")

        df = pd.DataFrame(
            self.np_data, columns=[f"ecg_{i}" for i in range(self.np_data.shape[1])]
        )
        print(f"Shape of generated data: {df.shape}")
        df.to_parquet(f"{self.save_path}/{name}/generated_tsa.parquet", index=False)
        print(f"Anomalies saved to {self.save_path}/{name}/generated_tsa.parquet")

    def _process_params(self, anomaly_params, data_shape, rng):
        # Check which of the parameters are lists
        check_variable_params = [isinstance(p, list) for p in anomaly_params.values()]
        self.contains_variable_params = any(check_variable_params)
        meta_keys = []
        if self.contains_variable_params:
            # Broadcast or randomly select other parameters
            for key, value in anomaly_params.items():
                if isinstance(value, list):
                    # Randomly select from the list for each entry in ecg_data
                    selected_vals = rng.choice(value, size=data_shape[0])
                    anomaly_params[key] = list(selected_vals)
                    print(
                        f"Randomly selected values for {key}: {selected_vals[:5]}... (first 5 values)"
                    )
                    meta_keys.append(key)
                elif isinstance(value, (int, float)):
                    # Broadcast the scalar value
                    anomaly_params[key] = list(np.full(data_shape[0], value))
                    print(f"Broadcasted value for {key}: {value}")
                else:
                    raise ValueError(
                        "anomaly_params must be a dict of lists, ints or floats."
                    )

        return anomaly_params, meta_keys


class AnomParams:
    def __init__(self):
        self.anom_params = {
            "physionet_a": {
                "platform": 0.2,
                "start": [i for i in np.arange(100, 2001, 1)],
                "length": [i for i in np.arange(400, 600, 1)],
            },
            "physionet_b": {
                "platform": 0.2,
                "start": [i for i in np.arange(100, 2001, 1)],
                "length": 500,
            },
            "physionet_c": {
                "slope": 0.006,
                "start": [i for i in np.arange(100, 2001, 1)],
                "length": [i for i in np.arange(400, 600, 1)],
            },
            "physionet_d": {
                "slope": 0.006,
                "start": [i for i in np.arange(100, 2001, 1)],
                "length": 500,
            },
            "physionet_e": {
                "mean": 0.4,
                "start": [i for i in np.arange(100, 2001, 1)],
                "length": [i for i in np.arange(400, 600, 1)],
            },
            "physionet_f": {
                "mean": 0.4,
                "start": [i for i in np.arange(100, 2001, 1)],
                "length": 500,
            },
            "physionet_g": {
                "spike": 9.0,
                "start": [i for i in np.arange(100, 2601, 100)],
            },
        }


def main():
    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--anom_type", type=str)
    parser.add_argument("--name", type=str)
    args = parser.parse_args()
    # get anomaly params
    anom_params = AnomParams().anom_params
    assert args.name in anom_params.keys(), f"{args.name} not in {anom_params.keys()}"
    # load data
    df = pd.read_parquet(args.data_path)
    # generate anomalies
    anom_funcs = AnomFuncs(df)
    anom_funcs.generate_anomalies(args.anom_type, anom_params[args.name], args.name)


if __name__ == "__main__":
    main()
