from __future__ import annotations

from collections import deque
from typing import Any, Deque

import numpy as np
import numpy.typing as npt
import torch
from sklearn.metrics import f1_score, precision_score, recall_score  # type: ignore

from library.type_aliases import ChanBatch, SignalBatch


def corr_multiple(x: ChanBatch, y: ChanBatch) -> list[Any]:
    assert x.shape == y.shape, f"{x.shape=}, {y.shape=}"
    res = [np.corrcoef(x[:, i], y[:, i], rowvar=False)[0, 1] for i in range(x.shape[1])]
    # log.debug(f"corr_multiple_res={res}, {x.shape=}, {y.shape=}")
    return res


def detect_voice(y_batch: SignalBatch, thresh: float = 1) -> npt.NDArray[np.bool_]:
    n_channels = y_batch.shape[1]
    return np.sum(y_batch > thresh, axis=1) > int(n_channels * 0.25)


def compute_regression_metrics(y_predicted: ChanBatch, y_true: ChanBatch) -> dict[str, float]:
    speech_idx = detect_voice(y_true)

    metrics = {}
    metrics["correlation"] = float(np.nanmean(corr_multiple(y_predicted, y_true)))

    if speech_idx is not None:
        y_predicted, y_true = y_predicted[speech_idx], y_true[speech_idx]
        corr_speech = corr_multiple(y_predicted, y_true)
        # log.debug(f"compute_regression_metrics: {corr_speech=}, {len(corr_speech)=}")
        metrics["correlation_speech"] = float(np.nanmean(corr_speech))
    else:
        metrics["correlation_speech"] = 0

    return metrics


def compute_classification_metrics(y_predicted: ChanBatch, y_true: ChanBatch) -> dict[str, float]:
    y_pred_tag = torch.round(torch.sigmoid(torch.from_numpy(y_predicted))).numpy()
    res = dict(
        accuracy=float(np.sum(y_pred_tag == y_true) / y_true.size),
        f1=float(f1_score(np.squeeze(y_true), np.squeeze(y_pred_tag))),
        precision=float(precision_score(np.squeeze(y_true), np.squeeze(y_pred_tag))),
        recall=float(recall_score(np.squeeze(y_true), np.squeeze(y_pred_tag))),
    )
    return res


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
        return bool(np.sum(smoothed_metrics) >= np.sum(self.best_metrics))

    def update_best(self) -> None:
        self.best_metrics = self.get_smoothed_metrics()

    def get_smoothed_metrics(self) -> np.ndarray:
        assert self.metrics_buffer, "buffer is empty"
        return np.asarray(self.metrics_buffer).mean(axis=0)
