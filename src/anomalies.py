from typing import Any
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm
import neurokit2 as nk
import pickle

import warnings
warnings.filterwarnings("ignore")


def mix_anomalies(cfg, splitted_data):
    """
    Adds anomalies to validation and test input data plus replaces y to binary y-labels.
    By default only adds anomalies to validation and test data, but can also include training data.
    """
    splitted_data = list(splitted_data)
    if cfg["mix_train"]:
        idx = [0, 1, 2]
    else:
        idx = [1, 2]
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
    def inject_variance(ts_row, level, start, length):
        var = np.random.normal(0, level, length)
        ts_row[start : start + length] += var
        return ts_row

    @staticmethod
    def inject_extremum(ts_row, level, start):
        # ts_row[start] += level
        ts_row[start] = level
        return ts_row

    @staticmethod
    def inject_pattern(ts_row, pattern_param, start, length, pattern="sine"):
        if pattern == "sine":
            ts_row[start : start + length] = AnomFuncs._sine_wave(
                np.arange(0, length), amp=pattern_param
            )
        return ts_row
    
    @staticmethod
    def inject_amplitude(ts_row, level, start, length):
        amplitude_bell = np.ones(length) * level
        ts_row[start : start + length] *= amplitude_bell
        return ts_row

    @staticmethod
    def inject_frequency_ecg(ts_row, level, startp, lengthp):
        if level == 1:
            return ts_row

        try:
            ecg = nk.ecg_process(ts_row, sampling_rate=300)[0]
        except:
            ### Unexpected error from processing the raw data
            return np.zeros(len(ts_row))
        start, length = int(startp), int(lengthp)

        ### Not enough phases
        if ecg["ECG_T_Offsets"].values.sum() < length or start + length > ecg["ECG_T_Offsets"].values.sum():
            return np.zeros(len(ts_row))
        phase_starts = np.insert(np.where(ecg["ECG_T_Offsets"].values == 1)[0], 0, 0)

        peaks = []
        for peak_name in ["ECG_R_Peaks", "ECG_P_Peaks", "ECG_Q_Peaks", "ECG_S_Peaks", "ECG_T_Peaks"]:
            peaks.extend(np.where(ecg[peak_name].values == 1)[0])
        peaks = np.sort(peaks)

        ### Augment the data with desired frequency
        idx_ind = [[phase_starts[i], phase_starts[i+1]] for i in range(start, start + length)]
        data = []
        for i in range(int(level * length)):
            s, e = idx_ind[int(i % len(idx_ind))]
            data = np.concatenate([data, ts_row[np.arange(s, e)]])

        ### Peaks need to be kept
        injected_idx = np.arange(phase_starts[start], phase_starts[start + length])
        sample_idx = np.sort(np.random.choice([i for i in range(len(data)) if i not in peaks], len(injected_idx), replace=False))
        ts_row[injected_idx] = data[sample_idx]

        return ts_row

    @staticmethod
    def inject_frequency_motion(ts_row, level, startp, lengthp):
        if level == 1:
            return ts_row

        start, length = int(startp), int(lengthp)
        zero_crossings = np.where(np.diff(np.sign(ts_row)) > 0)[0][:-1]

        ### To address noisy cross points
        threshold = 90
        anchor = zero_crossings[0]
        del_idx = []
        for i in range(1, len(zero_crossings)-1):
            if zero_crossings[i] - anchor > threshold:
                anchor = zero_crossings[i]
            else:
                del_idx.append(i)
        zero_crossings = np.delete(zero_crossings, del_idx)

        ### Not enough phases
        if len(zero_crossings) < length or start + length + 1 > len(zero_crossings):
            return np.zeros(len(ts_row))

        idx_ind = [[zero_crossings[i], zero_crossings[i+1]] for i in range(start, start + length)]
        data = []
        for i in range(int(np.ceil(level * length))):
            s, e = idx_ind[int(i % len(idx_ind))]
            data = np.concatenate([data, ts_row[np.arange(s, e)]])

        data = np.interp(
            np.linspace(0, len(data), zero_crossings[start+length] - zero_crossings[start]),
            np.arange(len(data)),
            data,
        )
        ts_row[zero_crossings[start]:zero_crossings[start+length]] = data
        return ts_row

    @staticmethod
    def inject_shift(ts_row, rate, start, length):
        first_half_length = int(length / 2)
        second_half_length = int(length - int(length / 2))
        subsequence = ts_row[start:start+length]
        transition_length = int(length * rate)
        transition_start = np.interp(
            np.linspace(0, first_half_length, transition_length),
            np.arange(first_half_length),
            subsequence[:first_half_length],
        )
        transition_end = np.interp(
            np.linspace(0, second_half_length, length - transition_length),
            np.arange(second_half_length),
            subsequence[-second_half_length:],
        )
        ts_row[start:start+length] = np.concatenate([transition_start, transition_end])
        return ts_row

    @staticmethod
    def _sine_wave(x, amp, freq=0.01, phase=0):
        return amp * np.sin(2 * np.pi * freq * x + phase)

    def generate_anomalies(self, anomaly_type, anomaly_params, truncated_length, aug_num, name):
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
        
        if truncated_length != -1:
            self.np_data = self.np_data[:, :truncated_length]
  
        if aug_num != 1:
            new_aug_data = np.array(self.np_data)
            for _ in range(aug_num-1):
                new_aug_data = np.concatenate([new_aug_data, self.np_data], axis=0)
            self.np_data = new_aug_data
        self.np_data_normal = np.array(self.np_data)

        anomaly_params_processed, meta_keys = self._process_params(
            anomaly_params, self.np_data.shape, self.rng
        )

        keep_idx = []
        for i in tqdm(range(self.np_data.shape[0])):
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
            elif anomaly_type == "variance":
                self.np_data[i, :] = self.inject_variance(
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
            elif anomaly_type == "amplitude":
                self.np_data[i, :] = self.inject_amplitude(
                    self.np_data[i, :], **anomaly_params
                )
            elif anomaly_type == 'frequency':
                if name.split('_')[0] == 'ecg':
                    self.np_data[i, :] = self.inject_frequency_ecg(
                        self.np_data[i, :], **anomaly_params
                    )
                elif name.split('_')[0] == 'mot':
                    self.np_data[i, :] = self.inject_frequency_motion(
                        self.np_data[i, :], **anomaly_params
                    )
            elif anomaly_type == 'shift':
                self.np_data[i, :] = self.inject_shift(
                    self.np_data[i, :], **anomaly_params
                )
            else:
                raise ValueError(
                    f"anomaly_type must be one of ['platform', 'trend', 'mean', 'extremum', 'pattern'], but {anomaly_type} was given."
                )

            if not np.abs(self.np_data[i]).sum() == 0:
                keep_idx.append(i)
        keep_idx = np.array(keep_idx)

        if meta_keys != []:
            # Save the meta data
            meta_df = pd.DataFrame(
                {key: np.array(anomaly_params_processed[key])[keep_idx] for key in meta_keys}
            )
            # normalize length and start position
            if "length" in meta_keys:
                print("Normalizing length and/or start position")
                meta_df["length"] = meta_df["length"] / self.np_data.shape[1]
            if "start" in meta_keys:
                meta_df["start"] = meta_df["start"] / self.np_data.shape[1]
            meta_df.to_parquet(f"{self.save_path}/{name}/meta_data.parquet", index=False)
            print(f"Meta data saved to {self.save_path}/{name}/meta_data.parquet")

        df_normal = pd.DataFrame(self.np_data_normal[keep_idx], columns=[f"ecg_{i}" for i in range(self.np_data_normal.shape[1])])
        df_normal.to_parquet(f"{self.save_path}/{name}/normal.parquet", index=False)
        print(f"Normals saved to {self.save_path}/{name}/normal.parquet")

        df_injected = pd.DataFrame(self.np_data[keep_idx], columns=[f"ecg_{i}" for i in range(self.np_data.shape[1])])
        df_injected.to_parquet(f"{self.save_path}/{name}/generated_tsa.parquet", index=False)
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
    parser.add_argument("--truncated_length", type=int, default=-1)
    parser.add_argument("--aug_num", type=int, default=1)
    parser.add_argument("--name", type=str)
    args = parser.parse_args()
    # get anomaly params
    anom_params = AnomParams().anom_params
    assert args.name in anom_params.keys(), f"{args.name} not in {anom_params.keys()}"
    # load data

    if args.data_path.split('.')[1] == 'parquet':
        df = pd.read_parquet(args.data_path)
    elif args.data_path.split('.')[1] == 'pkl':
        with open(args.data_path, 'rb') as handle:
            df = pickle.load(handle)

    # generate anomalies
    anom_funcs = AnomFuncs(df)
    anom_funcs.generate_anomalies(args.anom_type, anom_params[args.name], args.truncated_length, args.aug_num, args.name)


if __name__ == "__main__":
    main()
