import logging
import os
import os.path
from collections import defaultdict
from typing import Any

import hydra
import matplotlib.pyplot as plt  # type: ignore
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
from library.interpreter import ModelInterpreter
from library.models_regression import SimpleNet
from library.runner import TrainTestLoopRunner, compute_regression_metrics, run_experiment
from library.torch_datasets import Continuous
from library.visualize import InterpretPlotLayout, TopoVisualizer, plot_temporal_as_line

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


@log_execution_time("collecting best model metrics")
def get_final_metrics(model, train_ldr, test_ldr, nsteps: int) -> dict[str, float]:  # type: ignore
    metrics = defaultdict(lambda: 0.0)  # type: ignore
    for stage, ldr in (("train", train_ldr), ("test", test_ldr)):
        tr = trange(nsteps, desc=f"Best model evaluation loop: {stage}")
        for _, (y_pred, y_true, l) in zip(tr, TrainTestLoopRunner(model, ldr)):
            m = compute_regression_metrics(y_pred, y_true)
            metrics[f"{stage}/corr"] += m["correlation"] / nsteps
            metrics[f"{stage}/corr_speech"] += m["correlation_speech"] / nsteps
            metrics[f"{stage}/loss"] += l / nsteps
    return dict(metrics)


@log_execution_time()
def add_model_weights_figure(model, X, mne_info, tracker, cfg: MainConfig) -> None:  # type: ignore
    mi = ModelInterpreter(model, X)
    freqs, weights, patterns = mi.get_temporal(nperseg=1000)
    sp = mi.get_spatial_patterns()
    sp_naive = mi.get_naive()
    plot_topo = TopoVisualizer(mne_info)
    pp = InterpretPlotLayout(cfg.model.hidden_channels, plot_topo, plot_temporal_as_line)

    pp.FREQ_XLIM = 150
    pp.add_temporal(freqs, weights, "weights")
    pp.add_temporal(freqs, patterns, "patterns")
    pp.add_spatial(sp, "patterns")
    pp.add_spatial(sp_naive, "naive")
    pp.finalize()
    plt.switch_backend("agg")
    tracker.add_figure(tag=f"nsteps = {cfg.n_steps}", figure=pp.fig)


@log_execution_time()
@hydra.main(config_path="./configs", config_name="config")
def main(cfg: MainConfig) -> None:
    log.debug(f"Current working directory is {os.getcwd()}")
    if cfg.debug:
        main_utils.print_config(cfg)
        main_utils.set_debug_level()
    main_utils.create_dirs()
    git_utils.dump_commit_hash(cfg.debug)

    X, Y, info = read_data(cfg)
    log.info(f"Loaded X: {str(X)}\nLoaded Y: {str(Y)}")
    train_ldr, test_ldr = create_data_loaders(X, Y, cfg)

    model = SimpleNet(cfg.model)
    if torch.cuda.is_available():
        model = model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    with SummaryWriter("TB") as sw:
        run_experiment(model, optimizer, train_ldr, test_ldr, cfg.n_steps, cfg.model_upd_freq, sw)
        hparams = get_selected_params(cfg)
        metrics = get_final_metrics(model, train_ldr, test_ldr, cfg.metric_iter)
        options = {"debug": [True, False]}
        sw.add_hparams(hparams, metrics, hparam_domain_discrete=options, run_name="hparams")
        add_model_weights_figure(model, X, info.mne_info, sw, cfg)


if __name__ == "__main__":
    main()
