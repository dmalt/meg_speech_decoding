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

from .runner_common import infinite
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


class ScalarLogger(Protocol):
    def add_scalar(self, tag: str, scalar_value: float, global_step: int | None) -> None:
        ...


# TODO: change dataset type
def train_loop(
    model, train_generator, test_generator, cfg, logger: ScalarLogger, debug: bool = False
) -> None:

    max_steps = cfg.max_iterations_count if not debug else 1_000

    model_filename = f"{model.__class__.__name__}"
    model_path = f"model_dumps/{model_filename}.pth"
    metrics_tracker = BestMetricsTracker()

    i = 0
    for i in trange(max_steps):
        x_train, y_train = next(train_generator)
        y_predicted, loss = train_batch(model, x_train, y_train)
        logger.add_scalar("train/loss", loss, i)
        train_metrics = compute_metrics(y_predicted, y_train)
        for tag, value in train_metrics.items():
            logger.add_scalar(f"train/{tag}", value, i)

        x_test, y_test = next(test_generator)
        y_predicted, loss = test_batch(model, x_test, y_test)
        logger.add_scalar("test/loss", loss, i)
        test_metrics = compute_metrics(y_predicted, y_test)
        for tag, value in test_metrics.items():
            logger.add_scalar(f"test/{tag}", value, i)

        metrics_tracker.update_buffer(np.array([v for v in test_metrics.values()]))
        if not i % cfg.upd_evry_n_steps and metrics_tracker.is_improved():
            metrics_tracker.update_best()
            torch.save(model.model.state_dict(), model_path)

    if metrics_tracker.is_improved():
        torch.save(model.model.state_dict(), model_path)

    model.model.load_state_dict(torch.load(model_path))  # type: ignore


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
