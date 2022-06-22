import logging
import os
import os.path
from typing import Any

import hydra
import numpy as np
import numpy.typing as npt
import torch
from hydra.utils import call, instantiate
from ndp.signal import Signal
from ndp.signal.pipelines import Signal1DProcessor, SignalProcessor, align_samples
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import trange  # type: ignore

from library import git_utils, main_utils
from library.config_schema import MainConfig, get_selected_params
from library.func_utils import log_execution_time
from library.models_regression import SimpleNet
from library.runner import TrainTestLoopRunner, compute_regression_metrics, run_experiment
from library.torch_datasets import Continuous

log = logging.getLogger(__name__)
main_utils.setup_hydra()


@log_execution_time(desc="reading and transforming data")
def read_data(cfg: MainConfig) -> tuple[Signal[npt._32Bit], Signal[npt._32Bit], Any]:
    X, Y_, info = call(cfg.dataset.read)
    transform_x: SignalProcessor[npt._32Bit] = instantiate(cfg.dataset.transform_x)
    transform_y: Signal1DProcessor[npt._32Bit] = instantiate(cfg.dataset.transform_y)
    X = transform_x(X)
    Y = transform_y(Y_)
    Y = align_samples(Y, X)
    assert X.dtype == np.float32 and Y.dtype == np.float32
    return X, Y, info


def create_data_loaders(
    X: Signal[npt._32Bit], Y: Signal[npt._32Bit], cfg: MainConfig
) -> tuple[DataLoader, DataLoader]:
    dataset = Continuous(np.asarray(X), np.asarray(Y), cfg.lag_backward, cfg.lag_forward)
    train, test = dataset.train_test_split(cfg.train_test_ratio)
    train_ldr = DataLoader(train, batch_size=cfg.batch_size, shuffle=True)
    test_ldr = DataLoader(test, batch_size=cfg.batch_size, shuffle=True)
    return train_ldr, test_ldr


def get_final_metrics(model, train_ldr, test_ldr, nsteps):  # type: ignore
    metrics = dict(
        train_corr=0.0,
        train_corr_speech=0.0,
        train_loss=0.0,
        test_corr=0.0,
        test_corr_speech=0.0,
        test_loss=0.0,
    )
    tr = trange(nsteps, desc="Best model evaluation loop: train")
    for _, (y_pred, y_true, l) in zip(tr, TrainTestLoopRunner(model, train_ldr)):
        m = compute_regression_metrics(y_pred, y_true)
        metrics["train_corr"] += m["correlation"] / nsteps
        metrics["train_corr_speech"] += m["correlation_speech"] / nsteps
        metrics["train_loss"] += l / nsteps
    tr = trange(nsteps, desc="Best model evaluation loop: test")
    for _, (y_pred, y_true, l) in zip(tr, TrainTestLoopRunner(model, test_ldr)):
        m = compute_regression_metrics(y_pred, y_true)
        metrics["test_corr"] += m["correlation"] / nsteps
        metrics["test_corr_speech"] += m["correlation_speech"] / nsteps
        metrics["test_loss"] += l / nsteps
    return metrics


@log_execution_time()
@hydra.main(config_path="./configs", config_name="config")
def main(cfg: MainConfig) -> None:
    log.debug(f"Current working directory is {os.getcwd()}")
    if cfg.debug:
        main_utils.print_config(cfg)
    main_utils.create_dirs()
    git_utils.dump_commit_hash(cfg.debug)

    X, Y, _ = read_data(cfg)
    log.info(f"Loaded X: {str(X)}\nLoaded Y: {str(Y)}")
    train_ldr, test_ldr = create_data_loaders(X, Y, cfg)

    model = SimpleNet(cfg.model)
    if torch.cuda.is_available():
        model = model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    tracker = SummaryWriter("TB")  # type: ignore
    run_experiment(model, optimizer, train_ldr, test_ldr, cfg.n_steps, cfg.model_upd_freq, tracker)
    metrics = get_final_metrics(model, train_ldr, test_ldr, cfg.metric_iter)

    tracker.add_hparams(get_selected_params(cfg), metrics)


if __name__ == "__main__":
    main()
