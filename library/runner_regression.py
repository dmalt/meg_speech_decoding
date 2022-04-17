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
from .common_preprocessing import ClassicEcogPipeline  # noqa
from .common_preprocessing import ClassicMelspectrogramPipeline  # noqa
from .datasets import EcogDataset

MAX_ITERATIONS_COUNT = 100_0
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

    assert x_batch.shape[0] == y_batch.shape[0]
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

    metrics = {}

    y_predicted_numpy = y_predicted.cpu().detach().numpy()
    y_batch_numpy = y_batch.cpu().detach().numpy()

    metrics["loss"] = float(loss.cpu().detach().numpy())

    metrics["correlation"] = float(
        np.nanmean(corr_multiple(y_predicted_numpy, y_batch_numpy))
    )

    y_batch_speech_indexes = bench_model.detect_voice(y_batch_numpy)
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
    assert hasattr(bench_models_regression, bench_model_name), f"No such model:{bench_model_name}"
    bench_model = getattr(bench_models_regression, bench_model_name)(patient)

    cfg = EcogConfig.from_dict(patient)
    transform = ClassicEcogPipeline(
        bench_model.downsampling_coef,
        bench_model.LOW_PASS_HZ,
        bench_model.HIGH_PASS_HZ,
        bench_model.SELECTED_CHANNELS,
    )
    transform_target = ClassicMelspectrogramPipeline(
        bench_model.downsampling_coef, bench_model.N_MELS, bench_model.F_MAX
    )
    lb = bench_model.LAG_BACKWARD
    lf = bench_model.LAG_FORWARD
    dataset = EcogDataset(cfg, transform, transform_target, lb, lf)

    print(len(dataset))
    n = len(dataset)
    train, test = random_split(dataset, [int(n * 0.5), n - int(n * 0.5)])

    bs = bench_model.BATCH_SIZE
    train_generator = iter(DataLoader(train, batch_size=bs, shuffle=True))
    test_generator = iter(DataLoader(test, batch_size=bs, shuffle=True))
    val_generator = iter(DataLoader(test, batch_size=bs, shuffle=True))

    max_iterations_count = MAX_ITERATIONS_COUNT if not is_debug else 1_000

    best_iteration = 0
    max_metric = -float("inf")
    max_metric_speech = -float("inf")
    bench_model = getattr(bench_models_regression, bench_model_name)(patient)
    model_filename = f"regression___{patient['name']}___{bench_model.__class__.__name__}___{str(datetime.datetime.now())}"
    model_path = f"model_dumps/{model_filename}.pth"

    for iteration in trange(MAX_ITERATIONS_COUNT):
        process_batch(bench_model, train_generator, True, iteration)

        with torch.no_grad():
            process_batch(bench_model, val_generator, False, iteration)
        if iteration % 250 and iteration != max_iterations_count - 1:
            continue
        b, max_metric, max_metric_speech, best_iteration = should_stop(
            bench_model,
            iteration,
            max_metric,
            max_metric_speech,
            model_path,
            best_iteration,
        )
        if b:
            break

    bench_model.model.load_state_dict(torch.load(model_path))
    bench_model.model.eval()

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
        enumerate(DataLoader(test, batch_size=bs, shuffle=False))
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


def should_stop(
    bench_model,
    iteration,
    max_metric,
    max_metric_speech,
    model_path,
    best_iteration,
):
    m_smooth = bench_model.logger.get_smoothed_value("correlation")
    m_smooth_speech = bench_model.logger.get_smoothed_value(
        "correlation_speech"
    )
    if m_smooth >= max_metric or m_smooth_speech >= max_metric_speech:
        max_metric = max(m_smooth, max_metric)
        max_metric_speech = max(m_smooth_speech, max_metric_speech)
        best_iteration = iteration
        torch.save(bench_model.model.state_dict(), model_path)
        return False, max_metric, max_metric_speech, best_iteration
    else:
        assert iteration >= best_iteration
        if (iteration - best_iteration) > EARLY_STOP_STEPS:
            print(
                f"Stopping model. Iteration {iteration} {round(m_smooth, 2)} {round(m_smooth_speech, 2)}. Best iteration {best_iteration} {round(max_metric, 2)} {round(max_metric_speech, 2)}."
            )
            return True, max_metric, max_metric_speech, best_iteration
