from __future__ import annotations

import logging
from collections import deque
from typing import Deque, Generator, Protocol

import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
from tqdm import trange  # type: ignore

log = logging.getLogger(__name__)


def detect_voice(
    y_batch: npt.NDArray[np.floating[npt.NBitBase]], thresh: float = 1
) -> npt.NDArray[np.bool_]:
    n_channels = y_batch.shape[1]
    return np.sum(y_batch > thresh, axis=1) > int(n_channels * 0.25)


def corr_multiple(x, y):
    assert x.shape[1] == y.shape[1], f"{x.shape=}, {y.shape=}"
    return [np.corrcoef(x[:, i], y[:, i], rowvar=False)[0, 1] for i in range(x.shape[1])]


def train_batch(model: nn.Module, optimizer: torch.optim.Optimizer, x_batch, y_batch):
    model.train()
    loss_function = nn.MSELoss()
    optimizer.zero_grad()
    y_predicted = model(x_batch)
    loss = loss_function(y_predicted, y_batch)
    loss.backward()
    optimizer.step()
    return y_predicted.cpu().detach().numpy(), loss.cpu().detach().numpy()


def test_batch(model: nn.Module, x_batch, y_batch):
    model.eval()
    with torch.no_grad():
        loss_function = nn.MSELoss()
        y_predicted = model(x_batch)
        loss = loss_function(y_predicted, y_batch)
    return y_predicted.cpu().detach().numpy(), loss.cpu().detach().numpy()


def compute_metrics(y_predicted, y_batch):
    y_batch = y_batch.cpu().detach().numpy()
    speech_idx = detect_voice(y_batch)

    metrics = {}
    metrics["correlation"] = np.nanmean(corr_multiple(y_predicted, y_batch))

    if speech_idx is not None:
        y_predicted, y_batch = y_predicted[speech_idx], y_batch[speech_idx]
        metrics["correlation_speech"] = float(np.nanmean(corr_multiple(y_predicted, y_batch)))
    else:
        metrics["correlation_speech"] = 0

    return metrics


class ScalarTracker(Protocol):
    def add_scalar(self, tag: str, scalar_value: float, global_step: int | None) -> None:
        ...


def loop_generator(model: nn.Module, data_generator, optimizer=None) -> Generator:
    for x, y in data_generator:
        if optimizer is not None:
            y_predicted, loss = train_batch(model, optimizer, x, y)
        else:
            y_predicted, loss = test_batch(model, x, y)
        metrics = compute_metrics(y_predicted, y)
        metrics["loss"] = loss
        yield metrics


# TODO: change dataset type
def run_experiment(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    train_gen,
    test_gen,
    cfg,
    experiment_tracker: ScalarTracker,
) -> None:

    n_steps = cfg.max_iterations_count
    # n_steps = 1000

    model_filename = f"{model.__class__.__name__}"
    model_path = f"model_dumps/{model_filename}.pth"

    metrics_tracker = BestMetricsTracker()

    train_loop = loop_generator(model, train_gen, optimizer)
    test_loop = loop_generator(model, test_gen)
    for i, train_metrics, test_metrics in zip(trange(n_steps), train_loop, test_loop):
        for tag, value in train_metrics.items():
            experiment_tracker.add_scalar(f"train/{tag}", value, i)
        for tag, value in test_metrics.items():
            experiment_tracker.add_scalar(f"test/{tag}", value, i)

        metrics_tracker.update_buffer(np.array([v for k, v in test_metrics.items() if k != "loss"]))
        if (not i % cfg.upd_evry_n_steps):
            log.debug(f"{metrics_tracker.best_metrics=}, {metrics_tracker.get_smoothed_metrics()=}")
            if metrics_tracker.is_improved():
                metrics_tracker.update_best()
                log.info(f"Dumping model for iteration = {i}")
                torch.save(model.state_dict(), model_path)

    if metrics_tracker.is_improved():
        torch.save(model.state_dict(), model_path)

    model.load_state_dict(torch.load(model_path))  # type: ignore


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
