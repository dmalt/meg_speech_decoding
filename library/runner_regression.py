from __future__ import annotations

import json
import logging
from collections import deque
from dataclasses import astuple, dataclass
from typing import Any, Deque, Protocol

import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import trange  # type: ignore

from .bench_models_regression import BenchModelRegressionBase
from .runner_common import get_random_predictions, infinite
from .torch_datasets import Continuous

log = logging.getLogger(__name__)


def detect_voice(
    y_batch: npt.NDArray[np.floating[npt.NBitBase]], thresh: float = 1
) -> npt.NDArray[np.bool_]:
    n_channels = y_batch.shape[1]
    return np.sum(y_batch > thresh, axis=1) > int(n_channels * 0.25)


def corr_multiple(x, y):
    assert x.shape[1] == y.shape[1], f"{x.shape=}, {y.shape=}"
    return [np.corrcoef(x[:, i], y[:, i], rowvar=False)[0, 1] for i in range(x.shape[1])]


def train_batch(bench_model, x_batch, y_batch):
    bench_model.model.train()
    # loss_function = nn.MSELoss()
    loss_function = nn.MSELoss()
    bench_model.optimizer.zero_grad()
    y_predicted = bench_model.model(x_batch)
    loss = loss_function(y_predicted, y_batch)
    loss.backward()
    bench_model.optimizer.step()
    return y_predicted.cpu().detach().numpy(), loss.cpu().detach().numpy()


def test_batch(bench_model, x_batch, y_batch):
    bench_model.model.eval()
    with torch.no_grad():
        loss_function = nn.MSELoss()
        y_predicted = bench_model.model(x_batch)
        loss = loss_function(y_predicted, y_batch)
    return y_predicted.cpu().detach().numpy(), loss.cpu().detach().numpy()


def compute_metrics(y_predicted, y_batch, loss, speech_idx=None):
    y_batch = y_batch.cpu().detach().numpy()

    metrics = {}
    metrics["loss"] = float(loss)
    metrics["correlation"] = np.nanmean(corr_multiple(y_predicted, y_batch))

    if speech_idx is not None:
        y_predicted, y_batch = y_predicted[speech_idx], y_batch[speech_idx]
        metrics["correlation_speech"] = float(np.nanmean(corr_multiple(y_predicted, y_batch)))
    else:
        metrics["correlation_speech"] = 0

    return metrics


class ScalarLogger(Protocol):
    def add_scalar(self, tag: str, scalar_value: float, global_step: int | None) -> None:
        ...


# TODO: change dataset type
def run_regression(
    bench_model, dataset: Continuous, cfg, logger: ScalarLogger, debug: bool = False
) -> dict[str, float]:
    train, test = dataset.train_test_split(cfg.train_test_ratio)
    bs = cfg.batch_size
    train_generator = infinite(DataLoader(train, batch_size=bs, shuffle=True))
    test_generator = infinite(DataLoader(test, batch_size=bs, shuffle=True))
    val_generator = infinite(DataLoader(test, batch_size=bs, shuffle=True))

    max_steps = cfg.max_iterations_count if not debug else 1_000

    model_filename = f"{bench_model.__class__.__name__}"
    model_path = f"model_dumps/{model_filename}.pth"
    tracker = BestMetricsTracker()
    model_saver = ModelSaver(bench_model, model_path, tracker, cfg.early_stop_steps)

    i = 0
    for i in trange(max_steps):
        x_train, y_train = next(train_generator)
        y_predicted, loss = train_batch(bench_model, x_train, y_train)
        speech_idx = detect_voice(y_train.detach().numpy())
        train_metrics = compute_metrics(y_predicted, y_train, loss, speech_idx=speech_idx)
        for tag, value in train_metrics.items():
            logger.add_scalar(f"train/{tag}", value, i)

        x_test, y_test = next(test_generator)
        speech_idx = detect_voice(y_test.detach().numpy())
        y_predicted, loss = test_batch(bench_model, x_test, y_test)
        test_metrics = compute_metrics(y_predicted, y_test, loss, speech_idx=speech_idx)
        for tag, value in test_metrics.items():
            logger.add_scalar(f"test/{tag}", value, i)
        try:
            model_saver.update(i)
        except ModelIsStuckException as e:
            log.error(e)
            break

    save_path = f"results/{model_filename}.json"
    generators = {"train": train_generator, "test": test_generator, "val": val_generator}
    model_saver.save_results(save_path, generators, i)
    assert model_saver.tracker.max_test_metrics is not None
    return dict(
        correlation_raw=model_saver.tracker.max_test_metrics[0],
        correlation_speech=model_saver.tracker.max_test_metrics[1],
    )


class ModelIsStuckException(Exception):
    pass


@dataclass
class ModelSaver:
    model: BenchModelRegressionBase
    model_path: str
    tracker: BestMetricsTracker
    early_stop_steps: int

    def __post_init__(self) -> None:
        self.best_iteration = 0

    def update(self, iteration: int) -> None:
        if self.tracker.is_test_metrics_improved():
            self.tracker.update_max_test_metrics(iteration)
            torch.save(self.model.model.state_dict(), self.model_path)
        elif (iteration - self.best_iteration) > self.early_stop_steps:
            msg = self.tracker.stop_message(iteration)
            raise ModelIsStuckException(msg)

    def save_results(self, save_path: str, generators, iteration: int) -> None:
        self.model.model.load_state_dict(torch.load(self.model_path))
        self.model.model.eval()
        result = self.tracker.get_final_metrics(generators, iteration)
        log.info(
            f'train correlation={result["train_corr"]:.4f}, '
            + f'test correlation={result["test_corr"]:.4f}, '
            + f'val correlation={result["val_corr"]=:.4f}'  # TODO
        )
        log.info(f"Saving results to {save_path}")
        with open(save_path, "w") as f:
            json.dump(result, f)


class BestMetricsTracker:
    def __init__(self, metrics_buflen: int = 100):
        self.best_metrics: np.ndarray | None = None
        self.metrics_buffer: Deque[np.ndarray] = deque()
        self.buflen = metrics_buflen

    def update_buffer(self, new_metrics: np.ndarray) -> None:
        if len(self.metrics_buffer) >= self.buflen:
            self.metrics_buffer.popleft()
        self.metrics_buffer.append(new_metrics)

    def is_improved(self) -> bool:
        if self.best_metrics is None:
            return True
        smoothed_metrics = self.get_smoothed_metrics()
        return bool(np.all(smoothed_metrics >= self.best_metrics))

    def update_best(self) -> None:
        self.best_metrics = self.get_smoothed_metrics()

    def get_smoothed_metrics(self) -> np.ndarray:
        assert self.metrics_buffer, "buffer is empty"
        return np.asarray(self.metrics_buffer).mean(axis=0)

    # def get_final_metrics(self, generators: dict[str, Any], iteration: int) -> dict[str, Any]:
    #     result = {}
    #     for gen_name, gen in generators.items():
    #         p = get_random_predictions(self.model.model, gen, self.metric_iter)
    #         result[gen_name + "_corr"] = np.mean(corr_multiple(*p))
    #     result["train_logs"] = self.model.logger.train_logs
    #     result["val_logs"] = self.model.logger.test_logs
    #     result["iterations"] = iteration
    #     return result

    # def stop_message(self, iteration: int) -> str:
    #     raw = self.model.logger.get_smoothed_value("correlation")
    #     speech = self.model.logger.get_smoothed_value("correlation_speech")
    #     return (
    #         "Stopping model training due to no progress."
    #         + f"\n{iteration=} "
    #         + f"metric raw = {round(raw, 2)}, "
    #         + f"metric speech = {round(speech, 2)}."
    #         + f"\n{self.best_iteration=}, "
    #         + f"metric raw = {round(self.max_raw, 2)} "
    #         + f"metric speech = {round(self.max_speech, 2)}."
    #     )
