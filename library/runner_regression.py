from __future__ import annotations

import datetime
import inspect
import json
from dataclasses import dataclass

import h5py  # type: ignore
import librosa as lb  # type: ignore
import numpy as np  # type: ignore
import scipy.interpolate as sci  # type: ignore
import torch  # type: ignore
import torch.nn as nn  # type: ignore
from torch.utils.data import random_split  # type: ignore
from torch.utils.data import DataLoader
from tqdm import trange  # type: ignore

from . import bench_models_regression
from .common_preprocessing import (classic_ecog_pipeline,
                                   classic_melspectrogram_pipeline)
from .datasets import EcogDataset
from .runner_common import data_generator

MAX_ITERATIONS_COUNT = 300_0
# MAX_ITERATIONS_COUNT = 400_000
METRIC_ITERATIONS = 1_000
EARLY_STOP_STEPS = 10_000


def corr_multiple(x, y):
    assert x.shape[1] == y.shape[1]
    return [
        np.corrcoef(x[:, i], y[:, i], rowvar=False)[0, 1]
        for i in range(x.shape[1])
    ]


def process_batch(bench_model, generator, is_train, iteration):
    loss_function = nn.MSELoss()

    if is_train:
        bench_model.model.train()
    else:
        bench_model.model.eval()

    x_batch, y_batch = next(generator)
    x_batch = x_batch.detach().numpy()
    y_batch = y_batch.detach().numpy()

    y_batch_speech_indexes = bench_model.detect_voice(y_batch)

    assert x_batch.shape[0] == y_batch.shape[0]
    x_batch = torch.FloatTensor(x_batch)
    y_batch = torch.FloatTensor(y_batch)
    if torch.cuda.is_available():
        x_batch = x_batch.cuda()
        y_batch = y_batch.cuda()

    if is_train:
        bench_model.optimizer.zero_grad()

    y_predicted = bench_model.model(x_batch)
    assert not torch.any(torch.isnan(y_predicted))

    loss = loss_function(y_predicted, y_batch)

    if is_train:
        loss.backward()
        bench_model.optimizer.step()

    assert (
        y_predicted.shape[0] == y_batch.shape[0]
    ), f"{y_predicted.shape[0]} != {y_batch.shape[0]}"
    assert (
        y_predicted.shape[1] == y_batch.shape[1]
    ), f"{y_predicted.shape[1]} != {y_batch.shape[1]}"

    metrics = {}

    y_predicted_numpy = y_predicted.cpu().detach().numpy()
    y_batch_numpy = y_batch.cpu().detach().numpy()

    metrics["loss"] = float(loss.cpu().detach().numpy())

    metrics["correlation"] = float(
        np.nanmean(corr_multiple(y_predicted_numpy, y_batch_numpy))
    )

    if np.any(y_batch_speech_indexes):
        metrics["correlation_speech"] = float(
            np.nanmean(
                corr_multiple(
                    y_predicted_numpy[y_batch_speech_indexes],
                    y_batch_numpy[y_batch_speech_indexes],
                )
            )
        )

    for key, value in metrics.items():
        bench_model.logger.add_value(key, is_train, value, iteration)

    return metrics


def get_random_predictions(model, generator, iterations):
    Y_batch = []
    Y_predicted = []
    for index, (x_batch, y_batch) in enumerate(generator):
        x_batch = torch.FloatTensor(x_batch)
        if torch.cuda.is_available():
            x_batch = x_batch.cuda()
        y_predicted = model(x_batch).cpu().detach().numpy()
        assert x_batch.shape[0] == y_predicted.shape[0]
        Y_predicted.append(y_predicted)
        Y_batch.append(y_batch)
        if index > iterations:
            break

    Y_predicted = np.concatenate(Y_predicted, axis=0)
    Y_batch = np.concatenate(Y_batch, axis=0)
    return Y_batch, Y_predicted


def load_ecog(bench_model, is_debug, patient):
    X = []
    Y = []

    print("Loading data...")
    for filepath in patient["files_list"]:
        with h5py.File(filepath, "r+") as input_file:
            data = input_file["RawData"]["Samples"][()]

        ecog = data[:, patient["ecog_channels"]].astype("double")
        sound = data[:, patient["sound_channel"]].astype("double")

        x = bench_model.preprocess_ecog(ecog, patient["sampling_rate"]).astype(
            "float32"
        )
        y = bench_model.preprocess_sound(
            sound, patient["sampling_rate"], x.shape[0]
        ).astype("float32")

        if len(y.shape) == 1:
            y = y.reshape((-1, 1))

        assert x.shape[0] == y.shape[0]

        X.append(x)
        Y.append(y)

        if is_debug and len(X) >= 2:
            break

    print("Done")

    test_start_file_index = (
        bench_model.TEST_START_FILE_INDEX if not is_debug else 1
    )

    X_train = np.concatenate(X[:test_start_file_index], axis=0)
    Y_train = np.concatenate(Y[:test_start_file_index], axis=0)

    X_val = np.concatenate(X[test_start_file_index:], axis=0)
    Y_val = np.concatenate(Y[test_start_file_index:], axis=0)

    X_test = np.concatenate(X[test_start_file_index:], axis=0)
    Y_test = np.concatenate(Y[test_start_file_index:], axis=0)

    return X_train, Y_train, X_val, Y_val, X_test, Y_test


def load_meg(bench_model, n_mels=40, f_max=2000):
    import sklearn

    filepath = "/home/altukhov/Data/speech/rawdata/meg.h5"
    with h5py.File(filepath, "r+") as input_file:
        meg = input_file["RawData"]["Samples"][()].astype("double")
        sound = input_file["RawData"]["Audio"][()].astype("double")

    sound /= np.max(np.abs(sound))

    hop = 512
    sound_sr = 22050
    melspectrogram = lb.feature.melspectrogram(
        y=sound,
        sr=sound_sr,
        n_mels=n_mels,
        fmax=f_max,
        hop_length=hop,
    )
    melspectrogram = lb.power_to_db(melspectrogram, ref=np.max).T
    melspectrogram = sklearn.preprocessing.scale(melspectrogram)

    x = np.arange(0, len(sound), hop)
    itp = sci.interp1d(
        x, melspectrogram, bounds_error=False, fill_value="extrapolate", axis=0
    )

    meg_nsamp = len(meg)
    meg_sr = 1000
    meg_times = np.arange(0, meg_nsamp / meg_sr, 1 / meg_sr)
    meg_samp = (meg_times * sound_sr).astype(int)
    Y = itp(meg_samp)
    X = meg

    print(f"{X.shape=}", f"{Y.shape=}")

    X = sklearn.preprocessing.scale(X, copy=False)
    # Y = sklearn.preprocessing.scale(Y, copy=False)
    X = X.astype("float32")
    ntr = int(X.shape[0] * 0.50)

    X_test = X[:ntr]
    Y_test = Y[:ntr]

    X_train = X[ntr:]
    Y_train = Y[ntr:]

    X_val = X[:ntr]
    Y_val = Y[:ntr]

    # Y = Y.astype("float32")
    return X_train, Y_train, X_val, Y_val, X_test, Y_test


@dataclass
class EcogConfig:
    files_list: list[str]
    ecog_channels: list[int]
    sound_channel: int
    sampling_rate: float

    @classmethod
    def from_dict(cls, env):
        return cls(
            **{
                k: v
                for k, v in env.items()
                if k in inspect.signature(cls).parameters
            }
        )


def run_regression(bench_model_name, patient, runs_count=1, is_debug=False):
    assert hasattr(
        bench_models_regression, bench_model_name
    ), f"No such model:{bench_model_name}"
    bench_model = getattr(bench_models_regression, bench_model_name)(patient)
    # X_train, Y_train, X_val, Y_val, X_test, Y_test = load_ecog(
    #     bench_model, is_debug, patient
    # )
    # X_train, Y_train, X_val, Y_val, X_test, Y_test = load_meg(bench_model)

    batch_size = bench_model.BATCH_SIZE
    lag_backward = bench_model.LAG_BACKWARD
    lag_forward = bench_model.LAG_FORWARD

    cfg = EcogConfig.from_dict(patient)
    dataset = EcogDataset(
        cfg,
        classic_ecog_pipeline,
        classic_melspectrogram_pipeline,
        lag_backward,
        lag_forward,
        bench_model.SELECTED_CHANNELS,
        bench_model.downsampling_coef,
        bench_model.LOW_PASS_HZ,
        bench_model.HIGH_PASS_HZ,
        bench_model.N_MELS,
        bench_model.F_MAX,
    )

    print(len(dataset))
    train, test = random_split(
        dataset,
        (int(len(dataset) * 0.5), len(dataset) - int(len(dataset) * 0.5)),
    )

    train_generator = iter(DataLoader(train, batch_size=100, shuffle=True))
    test_generator = iter(DataLoader(test, batch_size=100, shuffle=True))
    val_generator = iter(DataLoader(test, batch_size=100, shuffle=True))
    # train_generator = data_generator(
    #     X_train,
    #     Y_train,
    #     batch_size,
    #     lag_backward,
    #     lag_forward,
    #     shuffle=True,
    #     infinite=True,
    # )
    # val_generator = data_generator(
    #     X_val,
    #     Y_val,
    #     batch_size,
    #     lag_backward,
    #     lag_forward,
    #     shuffle=True,
    #     infinite=True,
    # )

    max_iterations_count = MAX_ITERATIONS_COUNT if not is_debug else 1_000

    print("Starting iterations")

    for run_iteration in range(runs_count):
        print("Run iteration:", run_iteration)
        best_iteration = 0
        max_metric = -float("inf")
        max_metric_speech = -float("inf")
        bench_model = getattr(bench_models_regression, bench_model_name)(
            patient
        )
        model_filename = f"regression___{patient['name']}___{bench_model.__class__.__name__}___{str(datetime.datetime.now())}"
        model_path = f"model_dumps/{model_filename}.pth"
        for iteration in trange(MAX_ITERATIONS_COUNT):
            process_batch(bench_model, train_generator, True, iteration)
            with torch.no_grad():
                metrics = process_batch(
                    bench_model, val_generator, False, iteration
                )
                is_last_iteration = iteration == (max_iterations_count - 1)
                if iteration % 250 == 0 or is_last_iteration:
                    smoothed_metric = bench_model.logger.get_smoothed_value(
                        "correlation"
                    )
                    smoothed_metric_speech = (
                        bench_model.logger.get_smoothed_value(
                            "correlation_speech"
                        )
                    )
                    if (
                        smoothed_metric >= max_metric
                        or smoothed_metric_speech >= max_metric_speech
                    ):
                        max_metric = max(smoothed_metric, max_metric)
                        max_metric_speech = max(
                            smoothed_metric_speech, max_metric_speech
                        )
                        best_iteration = iteration
                        torch.save(bench_model.model.state_dict(), model_path)
                    else:
                        assert iteration >= best_iteration
                        if (iteration - best_iteration) > EARLY_STOP_STEPS:
                            print(
                                f"Stopping model. Iteration {iteration} {round(smoothed_metric, 2)} {round(smoothed_metric_speech, 2)}. Best iteration {best_iteration} {round(max_metric, 2)} {round(max_metric_speech, 2)}."
                            )
                            break
        bench_model.model.load_state_dict(torch.load(model_path))
        bench_model.model.eval()

        # test_generator = data_generator(
        #     X_test,
        #     Y_test,
        #     batch_size,
        #     lag_backward,
        #     lag_forward,
        #     shuffle=True,
        #     infinite=True,
        # )

        result = {}
        result["train_corr"] = np.mean(
            corr_multiple(
                *get_random_predictions(
                    bench_model.model, train_generator, METRIC_ITERATIONS
                )
            )
        )
        result["val_corr"] = np.mean(
            corr_multiple(
                *get_random_predictions(
                    bench_model.model, val_generator, METRIC_ITERATIONS
                )
            )
        )
        result["test_corr"] = np.mean(
            corr_multiple(
                *get_random_predictions(
                    bench_model.model, test_generator, METRIC_ITERATIONS
                )
            )
        )
        result["train_logs"] = bench_model.logger.train_logs
        result["val_logs"] = bench_model.logger.test_logs
        result["iterations"] = iteration
        print(f"{result=}")

        ##########################################################
        Y_predicted = []
        Y_batch = []

        BATCH_SIZE = 100
        LAG_BACKWARD = 1000
        LAG_FORWARD = 0
        import matplotlib.pyplot as plt
        from tqdm import tqdm

        for index, (x_batch, y_batch) in tqdm(
            enumerate(DataLoader(test, batch_size=100, shuffle=False))
        ):
            #### Train
            bench_model.model.eval()

            if index < 200:
                continue  # skip first samples

            if index > 240:
                break  # skip all the rest

            x_batch = torch.FloatTensor(x_batch)
            y_predicted = bench_model.model(x_batch).cpu().data.numpy()
            assert x_batch.shape[0] == y_predicted.shape[0]
            Y_predicted.append(y_predicted)
            Y_batch.append(y_batch)

        Y_predicted = np.concatenate(Y_predicted, axis=0)
        Y_batch = np.concatenate(Y_batch, axis=0)

        print(
            "Correlation   val",
            np.corrcoef(Y_predicted[:, 0], Y_batch[:, 0], rowvar=False)[0, 1],
        )

        plt.figure(figsize=(16, 3))
        plt.plot(Y_predicted, label="predicted", alpha=0.7)
        plt.plot(Y_batch, label="true", alpha=0.7)
        plt.legend()
        plt.show()

        save_path = f"results/{model_filename}.json"
        with open(save_path, "w") as f:
            print("Saving results to ")
            json.dump(result, f)
