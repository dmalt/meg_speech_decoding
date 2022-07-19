from __future__ import annotations

import logging
from collections import defaultdict
from typing import Callable, DefaultDict, Generator, Protocol

import numpy as np
import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from tqdm import trange  # type: ignore

from library.func_utils import infinite, limited, log_execution_time
from library.metrics import BestMetricsTracker
from library.type_aliases import ChanBatch, ChanBatchTensor, SigBatchTensor

log = logging.getLogger(__name__)


LossFunction = Callable[[ChanBatchTensor, ChanBatchTensor], torch.Tensor]


class TrainTestLoop:
    def __init__(
        self,
        model: nn.Module,
        loader: DataLoader,
        loss_function: LossFunction,
        optimizer: Optimizer | None = None,
    ):
        self.model = model
        self.data_loader = loader
        self.loss_function = loss_function
        self.optimizer = optimizer

    def __iter__(self) -> Generator[tuple[ChanBatch, ChanBatch, float], None, None]:
        x_batch: SigBatchTensor
        y_batch: ChanBatchTensor
        for x_batch, y_batch in self.data_loader:
            if torch.cuda.is_available():
                x_batch, y_batch = x_batch.cuda(), y_batch.cuda()
            if self.optimizer is not None:
                y_predicted, loss = self.train_batch(x_batch, y_batch)
            else:
                y_predicted, loss = self.test_batch(x_batch, y_batch)
            yield y_predicted, y_batch.cpu().detach().numpy(), loss

    def train_batch(self, x: SigBatchTensor, y: ChanBatchTensor) -> tuple[ChanBatch, float]:
        self.model.train()
        assert self.optimizer is not None
        self.optimizer.zero_grad()
        y_predicted = self.model(x)
        loss = self.loss_function(y_predicted, y)
        loss.backward()
        self.optimizer.step()
        return y_predicted.cpu().detach().numpy(), float(loss.cpu().detach().numpy())

    def test_batch(self, x: SigBatchTensor, y: ChanBatchTensor) -> tuple[ChanBatch, float]:
        self.model.eval()
        with torch.no_grad():
            y_predicted = self.model(x)
            loss = self.loss_function(y_predicted, y)
        return y_predicted.cpu().detach().numpy(), float(loss.cpu().detach().numpy())


@log_execution_time("collecting best model metrics")
def eval_model(
    model: torch.nn.Module,
    ldr: DataLoader,
    loss: LossFunction,
    metrics_func: Callable,
    nsteps: int,
    tqdm_desc: str = "",
) -> dict[str, float]:
    metrics: DefaultDict[str, float] = defaultdict(lambda: 0.0)
    tr = trange(nsteps, desc=tqdm_desc)
    for y_pred, y_true, loss_step in limited(TrainTestLoop(model, ldr, loss)).by(tr):
        metrics["loss"] += loss_step / nsteps
        for k, v in metrics_func(y_pred, y_true).items():
            metrics[f"{k}"] += v / nsteps
    return dict(metrics)


class ScalarTracker(Protocol):
    def add_scalar(self, tag: str, scalar_value: float, global_step: int | None) -> None:
        ...


MetricsComputer = Callable[[ChanBatch, ChanBatch], dict[str, float]]


@log_execution_time(desc="the experiment")
def run_experiment(
    model: nn.Module,
    optimizer: Optimizer,
    train_loader: DataLoader,
    test_loader: DataLoader,
    n_steps: int,
    upd_steps_freq: int,
    experiment_tracker: ScalarTracker,
    compute_metrics: MetricsComputer,
    loss: LossFunction,
) -> None:

    model_filename = f"{model.__class__.__name__}"
    model_path = f"model_dumps/{model_filename}.pth"

    metrics_tracker = BestMetricsTracker()

    def to_metrics(args: tuple[ChanBatch, ChanBatch, float]) -> tuple[dict[str, float], float]:
        y_predicted, y_true, loss = args
        return compute_metrics(y_predicted, y_true), loss

    train_loop = map(
        to_metrics, infinite(TrainTestLoop(model, train_loader, loss, optimizer))
    )
    test_loop = map(to_metrics, infinite(TrainTestLoop(model, test_loader, loss)))
    tr = trange(n_steps, desc="Experiment main loop")
    for i, (m_train, l_train), (m_test, l_test) in zip(tr, train_loop, test_loop):
        for tag, value in m_train.items():
            experiment_tracker.add_scalar(f"ongoing_train/{tag}", value, i)
        experiment_tracker.add_scalar("ongoing_train/loss", l_train, i)
        for tag, value in m_test.items():
            experiment_tracker.add_scalar(f"ongoing_test/{tag}", value, i)
        experiment_tracker.add_scalar("ongoing_test/loss", l_test, i)

        metrics_tracker.update_buffer(np.array([v for v in m_test.values()]))
        if not i % upd_steps_freq:
            if metrics_tracker.is_improved():
                metrics_tracker.update_best()
                log.info(f"Dumping model for iteration = {i}")
                torch.save(model.state_dict(), model_path)

    if metrics_tracker.is_improved():
        torch.save(model.state_dict(), model_path)

    model.load_state_dict(torch.load(model_path))  # type: ignore
