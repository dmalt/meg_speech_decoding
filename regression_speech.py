import logging
from dataclasses import asdict
from functools import reduce
from typing import Any

import hydra
import matplotlib
import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
from hydra.utils import instantiate
from ndp.signal import Signal
from ndp.signal.annotations import Annotations
from ndp.signal.pipelines import Signal1DProcessor, SignalProcessor, align_samples
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm, trange  # type: ignore

import speech_meg
from library import main_utils
from library.config_schema import MainConfig, ParamsDict, flatten_dict, get_selected_params
from library.func_utils import infinite, limited, log_execution_time
from library.metrics import RegressionMetrics as metrics_cls
from library.models_regression import SimpleNet
from library.runner import LossFunction, TestIter, TrainIter, eval_model, train_model
from library.torch_datasets import Continuous
from library.visualize import ContinuousDatasetPlotter, get_model_weights_figure

matplotlib.use("TkAgg")

log = logging.getLogger(__name__)
main_utils.setup_hydra()


@log_execution_time(desc="reading and transforming data")
def prepare_data(cfg: MainConfig) -> tuple[Signal[npt._32Bit], Signal[npt._32Bit], Any]:
    X, Y_, info = speech_meg.read_subject(subject="01")
    log.debug(f"X before transform: {str(X)}")
    log.debug(f"Y before transform: {str(Y_)}")
    transform_x: SignalProcessor[npt._32Bit] = instantiate(cfg.dataset.transform_x)
    transform_y: Signal1DProcessor[npt._32Bit] = instantiate(cfg.dataset.transform_y)
    X = transform_x(X)
    Y = transform_y(Y_)
    Y = align_samples(Y, X)
    assert X.dtype == np.float32 and Y.dtype == np.float32
    return X, Y, info


Loaders = dict[str, DataLoader]


def get_annot_mask(
    annots: Annotations, annot_type: str, sr: float, n_samp: int
) -> npt.NDArray[np.bool_]:
    return reduce(
        lambda x, y: np.logical_or(x, y),
        (a.as_mask(sr, n_samp) for a in annots if a.type == annot_type),
    )


def create_data_loaders(X: Signal[npt._32Bit], Y: Signal[npt._32Bit], cfg: MainConfig) -> Loaders:
    dataset = Continuous(np.asarray(X), np.asarray(Y), cfg.lag_backward, cfg.lag_forward)
    train, test = dataset.train_test_split(cfg.train_test_ratio)

    train_speech_mask = get_annot_mask(X.annotations, "speech", X.sr, len(train.X))
    test_speech_mask = get_annot_mask(X.annotations, "speech", X.sr, len(test.X))

    train_speech_ind = [i for i in range(len(train)) if train_speech_mask[i + cfg.lag_backward]]
    test_speech_ind = [i for i in range(len(test)) if test_speech_mask[i + cfg.lag_backward]]

    train_speech = Subset(train, train_speech_ind)
    test_speech = Subset(train, test_speech_ind)

    dl_params = dict(batch_size=cfg.batch_size, shuffle=True, drop_last=True)
    return dict(
        train=DataLoader(train, **dl_params),  # type: ignore
        test=DataLoader(test, **dl_params),  # type: ignore
        train_speech=DataLoader(train_speech, **dl_params),  # type: ignore
        test_speech=DataLoader(test_speech, **dl_params),  # type: ignore
    )


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
    if cfg.plot_loaded:
        ContinuousDatasetPlotter(X, Y).plot()
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

    dataset = Continuous(np.asarray(X), np.asarray(Y), cfg.lag_backward, cfg.lag_forward)
    timeseries_ldr = DataLoader(dataset, shuffle=False, batch_size=cfg.batch_size)
    Y_predicted_data = np.zeros_like(Y.data, dtype=np.float32)
    off, stride = cfg.lag_backward, cfg.batch_size
    for i, (y_pred, _, _) in tqdm(
        enumerate(TestIter(model, timeseries_ldr, loss)), total=len(dataset) // stride
    ):
        Y_predicted_data[off + i * stride : off + (i + 1) * stride, :] = y_pred

    matplotlib.use("TkAgg")
    plotter = ContinuousDatasetPlotter(Signal(Y_predicted_data, Y.sr, Y.annotations), Y)
    plotter.plot()


if __name__ == "__main__":
    main()
