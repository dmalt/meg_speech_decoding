from __future__ import annotations

import logging
import operator as op
from dataclasses import asdict, dataclass
from functools import reduce
from typing import Callable, Generator, Iterable, Protocol

import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader

from library.func_utils import log_execution_time
from library.metrics import MetricsTracker, TMetrics
from library.type_aliases import ChanBatch, ChanBatchTensor, SigBatchTensor

log = logging.getLogger(__name__)


LossFunction = Callable[[ChanBatchTensor, ChanBatchTensor], torch.Tensor]


@dataclass
class BaseIter:
    model: nn.Module
    data_loader: DataLoader
    loss: LossFunction

    def __iter__(self) -> Generator[tuple[ChanBatch, ChanBatch, float], None, None]:
        for x_batch, y_batch in self.data_loader:
            if torch.cuda.is_available():
                x_batch, y_batch = x_batch.cuda(), y_batch.cuda()
            y_predicted, loss = self.process_batch(x_batch, y_batch)
            yield y_predicted, y_batch.cpu().detach().numpy(), loss

    def process_batch(self, x: SigBatchTensor, y: ChanBatchTensor) -> tuple[ChanBatch, float]:
        raise NotImplementedError


@dataclass
class TrainIter(BaseIter):
    optimizer: Optimizer

    def process_batch(self, x: SigBatchTensor, y: ChanBatchTensor) -> tuple[ChanBatch, float]:
        self.model.train()
        self.optimizer.zero_grad()
        y_predicted = self.model(x)
        loss = self.loss(y_predicted, y)
        loss.backward()
        self.optimizer.step()
        return y_predicted.cpu().detach().numpy(), float(loss.cpu().detach().numpy())


@dataclass
class TestIter(BaseIter):
    def __iter__(self) -> Generator[tuple[ChanBatch, ChanBatch, float], None, None]:
        for x_batch, y_batch in self.data_loader:
            if torch.cuda.is_available():
                x_batch, y_batch = x_batch.cuda(), y_batch.cuda()
            y_predicted, loss = self.process_batch(x_batch, y_batch)
            yield y_predicted, y_batch.cpu().detach().numpy(), loss

    def process_batch(self, x: SigBatchTensor, y: ChanBatchTensor) -> tuple[ChanBatch, float]:
        self.model.eval()
        with torch.no_grad():
            y_predicted = self.model(x)
            loss = self.loss(y_predicted, y)
        return y_predicted.cpu().detach().numpy(), float(loss.cpu().detach().numpy())


@log_execution_time("collecting best model metrics")
def eval_model(metric_iter: Iterable, metrics_cls: type[TMetrics], nsteps: int) -> TMetrics:
    return reduce(op.add, map(lambda x: metrics_cls.calc(*x), metric_iter)) / nsteps


class ScalarTracker(Protocol):
    def add_scalar(self, tag: str, scalar_value: float, global_step: int | None) -> None:
        ...


MetricsComputer = Callable[[ChanBatch, ChanBatch], dict[str, float]]


@log_execution_time(desc="the experiment")
def train_model(
    train_iter: Iterable[TMetrics],
    test_iter: Iterable[TMetrics],
    tr: Iterable,
    model: nn.Module,
    upd_steps_freq: int,
    experiment_tracker: ScalarTracker,
) -> None:

    model_filename = f"{model.__class__.__name__}"
    model_path = f"model_dumps/{model_filename}.pth"

    metrics_tracker: MetricsTracker[TMetrics] = MetricsTracker()

    for i, m_train, m_test in zip(tr, train_iter, test_iter):
        for tag, value in asdict(m_train).items():
            experiment_tracker.add_scalar(f"ongoing_train/{tag}", value, i)
        for tag, value in asdict(m_test).items():
            experiment_tracker.add_scalar(f"ongoing_test/{tag}", value, i)

        metrics_tracker.update_buffer(m_test)
        if not i % upd_steps_freq:
            if metrics_tracker.is_improved():
                log.info(f"Dumping model for iteration = {i}")
                torch.save(model.state_dict(), model_path)

    if metrics_tracker.is_improved():
        torch.save(model.state_dict(), model_path)

    model.load_state_dict(torch.load(model_path))  # type: ignore
