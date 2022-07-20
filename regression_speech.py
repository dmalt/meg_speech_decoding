import logging
from dataclasses import asdict
from typing import Any

import hydra
import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
from hydra.utils import instantiate
from ndp.signal import Signal
from ndp.signal.pipelines import Signal1DProcessor, SignalProcessor, align_samples
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import trange  # type: ignore

import speech_meg
from library import main_utils
from library.config_schema import MainConfig, ParamsDict, flatten_dict, get_selected_params
from library.func_utils import infinite, limited, log_execution_time
from library.metrics import RegressionMetrics as metrics_cls
from library.models_regression import SimpleNet
from library.runner import LossFunction, TestIter, TrainIter, eval_model, train_model
from library.torch_datasets import Continuous
from library.visualize import get_model_weights_figure

log = logging.getLogger(__name__)
main_utils.setup_hydra()


@log_execution_time(desc="reading and transforming data")
def prepare_data(cfg: MainConfig) -> tuple[Signal[npt._32Bit], Signal[npt._32Bit], Any]:
    X, Y_, info = speech_meg.read_subject(subject="01")
    transform_x: SignalProcessor[npt._32Bit] = instantiate(cfg.dataset.transform_x)
    transform_y: Signal1DProcessor[npt._32Bit] = instantiate(cfg.dataset.transform_y)
    X = transform_x(X)
    Y = transform_y(Y_)
    Y = align_samples(Y, X)
    assert X.dtype == np.float32 and Y.dtype == np.float32
    return X, Y, info


Loaders = dict[str, DataLoader]


def create_data_loaders(X: Signal[npt._32Bit], Y: Signal[npt._32Bit], cfg: MainConfig) -> Loaders:
    dataset = Continuous(np.asarray(X), np.asarray(Y), cfg.lag_backward, cfg.lag_forward)
    train, test = dataset.train_test_split(cfg.train_test_ratio)
    train_ldr = DataLoader(train, batch_size=cfg.batch_size, shuffle=True, drop_last=True)
    test_ldr = DataLoader(test, batch_size=cfg.batch_size, shuffle=True, drop_last=True)
    return dict(train=train_ldr, test=test_ldr)


def get_metrics(model: nn.Module, loss: LossFunction, ldrs: Loaders, n: int) -> ParamsDict:
    metrics = {}
    for stage, ldr in ldrs.items():
        tr = trange(n, desc=f"Evaluating model on {stage}")
        eval_iter = limited(TestIter(model, ldr, loss)).by(tr)
        metrics[stage] = asdict(eval_model(eval_iter, metrics_cls, n))

    log.debug(f"{metrics=}")
    return flatten_dict(metrics, sep="/")


@log_execution_time()
@hydra.main(config_path="./configs", config_name="regression_speech_config")
def main(cfg: MainConfig) -> None:
    main_utils.prepare_script(log, cfg)

    X, Y, info = prepare_data(cfg)
    log.info(f"Loaded X: {str(X)}\nLoaded Y: {str(Y)}")
    ldrs = create_data_loaders(X, Y, cfg)

    model = SimpleNet(cfg.model)
    if torch.cuda.is_available():
        model = model.cuda()

    loss = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    train_iter = map(
        lambda x: metrics_cls.calc(*x), infinite(TrainIter(model, ldrs["train"], loss, optimizer))
    )
    test_iter = map(lambda x: metrics_cls.calc(*x), infinite(TestIter(model, ldrs["test"], loss)))
    tr = trange(cfg.n_steps, desc="Experiment main loop")
    with SummaryWriter("TB") as sw:
        train_model(train_iter, test_iter, tr, model, cfg.model_upd_freq, sw)

        metrics = get_metrics(model, loss, ldrs, cfg.metric_iter)
        log.info("Final metrics: " + ", ".join(f"{k}={v:.3f}" for k, v in metrics.items()))

        options = {"debug": [True, False]}
        hparams = get_selected_params(cfg)
        sw.add_hparams(hparams, metrics, hparam_domain_discrete=options, run_name="hparams")

        fig = get_model_weights_figure(model, X, info.mne_info, cfg.model.hidden_channels)
        sw.add_figure(tag=f"nsteps = {cfg.n_steps}", figure=fig)


if __name__ == "__main__":
    main()
