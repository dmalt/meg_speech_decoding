from __future__ import annotations

import logging
import operator
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import asdict, astuple, dataclass
from functools import reduce, total_ordering
from typing import Any, Deque, Generic, TypeVar

import numpy as np
import torch
from sklearn.metrics import f1_score, precision_score, recall_score  # type: ignore

from library.type_aliases import ChanBatch

log = logging.getLogger(__name__)

TMetrics = TypeVar("TMetrics", bound="Metrics")


class Metrics(ABC):
    """Abstract class implementing metrics conversion to numpy array"""

    @classmethod
    @abstractmethod
    def calc(
        cls: type[TMetrics], y_predicted: ChanBatch, y_true: ChanBatch, bce_with_logit: float
    ) -> TMetrics:
        ...

    def __len__(self) -> int:
        return len(astuple(self))

    def __getitem__(self, ind: int | str) -> float:
        if isinstance(ind, int):
            return astuple(self)[ind]
        elif isinstance(ind, str):
            return asdict(self)[ind]
        else:
            raise IndexError(f"Index must be int or str; got {type(ind)}")

    def __add__(self: TMetrics, other: TMetrics) -> TMetrics:
        return self.__class__(*[x + y for x, y in zip(self, other)])  # type: ignore

    def __truediv__(self: TMetrics, number: float) -> TMetrics:
        return self.__class__(*[x / number for x in astuple(self)])  # type: ignore

    @abstractmethod
    def __lt__(self: TMetrics, other: TMetrics) -> bool:
        """Which model is better by the set of metrics"""


@total_ordering
@dataclass
class RegressionMetrics(Metrics):
    correlation: float
    mse: float

    @classmethod
    def calc(cls, y_predicted: ChanBatch, y_true: ChanBatch, mse: float) -> RegressionMetrics:
        correlation = float(np.nanmean(corr_multiple(y_predicted, y_true)))
        return cls(correlation, mse)

    def __lt__(self, other: RegressionMetrics) -> bool:
        return (self.correlation, -self.mse) < (other.correlation, -other.mse)


def corr_multiple(y1: ChanBatch, y2: ChanBatch) -> list[Any]:
    assert y1.shape == y2.shape, f"{y1.shape=}, {y2.shape=}"
    res = [np.corrcoef(y1[:, i], y2[:, i], rowvar=False)[0, 1] for i in range(y1.shape[1])]
    return res


@total_ordering
@dataclass
class BinaryClassificationMetrics(Metrics):
    f1_score: float
    accuracy: float
    precision: float
    recall: float
    bce_with_logit: float

    @classmethod
    def calc(
        cls, y_predicted: ChanBatch, y_true: ChanBatch, bce_with_logit: float
    ) -> BinaryClassificationMetrics:
        bce_with_logit = bce_with_logit
        y_pred_tag = torch.round(torch.sigmoid(torch.from_numpy(y_predicted))).numpy()
        accuracy = float(np.sum(y_pred_tag == y_true) / y_true.size)
        f1 = float(f1_score(np.squeeze(y_true), np.squeeze(y_pred_tag)))
        precision = float(precision_score(np.squeeze(y_true), np.squeeze(y_pred_tag)))
        recall = float(recall_score(np.squeeze(y_true), np.squeeze(y_pred_tag)))
        return cls(f1, accuracy, precision, recall, bce_with_logit)

    def __lt__(self, other: BinaryClassificationMetrics) -> bool:
        return (self.f1_score, self.accuracy) < (other.f1_score, other.accuracy)


class MetricsTracker(Generic[TMetrics]):
    def __init__(self, metrics_buflen: int = 100):
        self.best_metrics: TMetrics | None = None
        self.metrics_buffer: Deque[TMetrics] = deque()
        self.buflen = metrics_buflen

    def update_buffer(self, new_metrics: TMetrics) -> None:
        if len(self.metrics_buffer) >= self.buflen:
            self.metrics_buffer.popleft()
        self.metrics_buffer.append(new_metrics)

    def is_improved(self) -> bool:
        if not self.metrics_buffer:
            return False
        smoothed_metrics = self.get_smoothed_metrics()
        if self.best_metrics is None:
            self.best_metrics = smoothed_metrics
            return True
        if self.best_metrics < smoothed_metrics:
            self.best_metrics = smoothed_metrics
            return True
        return False

    def get_smoothed_metrics(self) -> TMetrics:
        assert self.metrics_buffer, "buffer is empty"
        return reduce(operator.add, self.metrics_buffer) / len(self.metrics_buffer)
