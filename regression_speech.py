import logging
import os
import os.path
from functools import partial
from typing import Any

import hydra
import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate
from ndp.signal import Signal
from ndp.signal.pipelines import Signal1DProcessor, SignalProcessor, align_samples
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter

import speech_meg  # type: ignore
from library import git_utils, main_utils
from library.config_schema import MainConfig, flatten_dict, get_selected_params
from library.func_utils import log_execution_time
from library.metrics import compute_regression_metrics
from library.models_regression import SimpleNet
from library.runner import eval_model, run_experiment
from library.torch_datasets import Continuous
from library.visualize import get_model_weights_figure

log = logging.getLogger(__name__)
main_utils.setup_hydra()


@log_execution_time(desc="reading and transforming data")
def read_data(cfg: MainConfig) -> tuple[Signal[npt._32Bit], Signal[npt._32Bit], Any]:
    X, Y_, info = speech_meg.read_subject(subject="01")
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
    train_ldr = DataLoader(train, batch_size=cfg.batch_size, shuffle=True, drop_last=True)
    test_ldr = DataLoader(test, batch_size=cfg.batch_size, shuffle=True, drop_last=True)
    return train_ldr, test_ldr


@log_execution_time()
@hydra.main(config_path="./configs", config_name="regression_speech_config")
def main(cfg: MainConfig) -> None:
    GlobalHydra.instance().clear()
    log.debug(f"Current working directory is {os.getcwd()}")
    if cfg.debug:
        main_utils.set_debug_level()
        main_utils.print_config(cfg)
    main_utils.create_dirs()
    main_utils.dump_environment()
    git_utils.dump_commit_hash(cfg.debug)

    X, Y, info = read_data(cfg)
    log.info(f"Loaded X: {str(X)}\nLoaded Y: {str(Y)}")
    train_ldr, test_ldr = create_data_loaders(X, Y, cfg)

    model = SimpleNet(cfg.model)
    if torch.cuda.is_available():
        model = model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)

    eval_func = partial(
        eval_model,
        model=model,
        loss=nn.MSELoss(),
        metrics_func=compute_regression_metrics,  # -> config
        nsteps=cfg.metric_iter,
    )
    with SummaryWriter("TB") as sw:
        run_experiment(
            model,
            optimizer,
            train_ldr,
            test_ldr,
            cfg.n_steps,
            cfg.model_upd_freq,
            sw,
            compute_metrics=compute_regression_metrics,  # -> config
            loss=nn.MSELoss(),  # -> config
        )
        hparams = get_selected_params(cfg)
        train_metrics = eval_func(ldr=train_ldr, tqdm_desc="Evaluating model on train")
        test_metrics = eval_func(ldr=test_ldr, tqdm_desc="Evaluating model on test")
        metrics = flatten_dict({"train": train_metrics, "test": test_metrics}, sep="/")

        log.info("Final metrics: " + ", ".join(f"{k}={v:.3f}" for k, v in metrics.items()))
        options = {"debug": [True, False]}
        sw.add_hparams(hparams, metrics, hparam_domain_discrete=options, run_name="hparams")
        fig = get_model_weights_figure(model, X, info.mne_info, cfg.model.hidden_channels)
        sw.add_figure(tag=f"nsteps = {cfg.n_steps}", figure=fig)


if __name__ == "__main__":
    main()
