from __future__ import annotations

import json
import logging
from dataclasses import dataclass

import numpy as np  # type: ignore
import torch  # type: ignore
import torch.nn as nn  # type: ignore
from torch.utils.data import DataLoader  # type: ignore
from tqdm import trange  # type: ignore

from .bench_models_regression import BenchModelRegressionBase
from .datasets import SpeechDataset
from .runner_common import get_random_predictions, infinite

log = logging.getLogger(__name__)


def corr_multiple(x, y):
    assert x.shape[1] == y.shape[1], f"{x.shape=}, {y.shape=}"
    return [
        np.corrcoef(x[:, i], y[:, i], rowvar=False)[0, 1]
        for i in range(x.shape[1])
    ]


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


def update_metrics(
    bench_model,
    y_predicted,
    y_batch,
    loss,
    iteration,
    is_train,
    speech_idx=None,
):
    y_batch = y_batch.cpu().detach().numpy()

    metrics = {}
    metrics["loss"] = float(loss)
    metrics["correlation"] = np.nanmean(corr_multiple(y_predicted, y_batch))

    if speech_idx is not None:
        y_predicted, y_batch = y_predicted[speech_idx], y_batch[speech_idx]
        metrics["correlation_speech"] = float(
            np.nanmean(corr_multiple(y_predicted, y_batch))
        )

    for key, value in metrics.items():
        bench_model.logger.add_value(key, is_train, value, iteration)

    return metrics


def run_regression(bench_model, dataset: SpeechDataset, cfg):
    train, test = dataset.train_test_split(cfg.train_test_ratio)
    bs = cfg.batch_size
    train_generator = infinite(DataLoader(train, batch_size=bs, shuffle=True))
    test_generator = infinite(DataLoader(test, batch_size=bs, shuffle=True))
    val_generator = infinite(DataLoader(test, batch_size=bs, shuffle=True))

    max_steps = cfg.max_iterations_count if not cfg.debug else 1_000

    model_filename = f"{bench_model.__class__.__name__}"
    model_path = f"model_dumps/{model_filename}.pth"
    tracker = BestModelTracker(
        bench_model, max_steps, cfg.upd_evry_n_steps, cfg.metric_iter
    )
    model_saver = BestModelSaver(
        bench_model, model_path, tracker, cfg.early_stop_steps
    )

    for i in trange(max_steps):
        x_train, y_train = next(train_generator)
        log.debug(f"{x_train.shape=}, {y_train.shape=}")
        y_predicted, loss = train_batch(bench_model, x_train, y_train)
        speech_idx = dataset.detect_voice(y_train.detach().numpy())
        update_metrics(
            bench_model,
            y_predicted,
            y_train,
            loss,
            i,
            is_train=True,
            speech_idx=speech_idx,
        )

        x_test, y_test = next(test_generator)
        speech_idx = dataset.detect_voice(y_test.detach().numpy())
        y_predicted, loss = test_batch(bench_model, x_test, y_test)
        update_metrics(
            bench_model,
            y_predicted,
            y_test,
            loss,
            i,
            is_train=False,
            speech_idx=speech_idx,
        )
        try:
            model_saver.update(i)
        except ModelIsStuckException as e:
            log.error(e)
            break

    save_path = f"results/{model_filename}.json"
    generators = {
        "train": train_generator,
        "test": test_generator,
        "val": val_generator,
    }
    model_saver.save_results(save_path, generators, i)


class ModelIsStuckException(Exception):
    pass


@dataclass
class BestModelSaver:
    model: BenchModelRegressionBase
    model_path: str
    tracker: BestModelTracker
    early_stop_steps: int

    def __post_init__(self):
        self.best_iteration = 0

    def update(self, iteration):
        if not self.tracker.should_update(iteration):
            return
        if self.tracker.metrics_improved():
            self.tracker.update_max_metrics(iteration)
            torch.save(self.model.model.state_dict(), self.model_path)
        elif (iteration - self.best_iteration) > self.early_stop_steps:
            msg = self.tracker.stop_message(iteration)
            raise ModelIsStuckException(msg)

    def save_results(self, save_path, generators, iteration):
        self.model.model.load_state_dict(torch.load(self.model_path))
        self.model.model.eval()
        result = self.tracker.get_final_metrics(generators, iteration)
        log.info(
            f'train correlation={result["train_corr"]:.4f}, '
            + f'test correlation={result["test_corr"]:.4f}, '
            + f'val correlation={result["val_corr"]=:.4f}'
        )
        log.info(f"Saving results to {save_path}")
        with open(save_path, "w") as f:
            json.dump(result, f)


class BestModelTracker:
    def __init__(self, model, max_steps, update_every_n_iter, metric_iter):
        self.best_iteration: int = 0
        self.max_raw: float = -float("inf")
        self.max_speech: float = -float("inf")
        self.model = model
        self.max_steps = max_steps
        self.update_every_n_iter = update_every_n_iter
        self.metric_iter = metric_iter

    def metrics_improved(self):
        raw = self.model.logger.get_smoothed_value("correlation")
        speech = self.model.logger.get_smoothed_value("correlation_speech")
        return raw >= self.max_raw or speech >= self.max_speech

    def get_final_metrics(self, generators, iteration):
        result = {}
        for gen_name, gen in generators.items():
            p = get_random_predictions(self.model.model, gen, self.metric_iter)
            result[gen_name + "_corr"] = np.mean(corr_multiple(*p))
        result["train_logs"] = self.model.logger.train_logs
        result["val_logs"] = self.model.logger.test_logs
        result["iterations"] = iteration
        return result

    def update_max_metrics(self, iteration):
        raw = self.model.logger.get_smoothed_value("correlation")
        speech = self.model.logger.get_smoothed_value("correlation_speech")
        self.max_raw = max(raw, self.max_raw)
        self.max_speech = max(speech, self.max_speech)
        self.best_iteration = iteration

    def should_update(self, iteration):
        if iteration == self.max_steps - 1:
            return True
        if iteration % self.update_every_n_iter:
            return False
        return True

    def stop_message(self, iteration, m_smooth_raw, m_smooth_speech):
        raw = self.model.logger.get_smoothed_value("correlation")
        speech = self.model.logger.get_smoothed_value("correlation_speech")
        return (
            "Stopping model training due to no progress."
            + f"\n{iteration=} "
            + f"metric raw = {round(raw, 2)}, "
            + f"metric speech = {round(speech, 2)}."
            + f"\n{self.best_iteration=}, "
            + f"metric raw = {round(self.max_raw, 2)} "
            + f"metric speech = {round(self.max_speech, 2)}."
        )
