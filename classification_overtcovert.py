import logging
from dataclasses import asdict
from functools import reduce
from typing import Any

import hydra
import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
from hydra.utils import instantiate
from ndp.signal import Signal
from ndp.signal.pipelines import SignalProcessor
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import trange

import speech_meg  # type: ignore
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
def read_data(transform_x_cfg: Any, subject: str) -> tuple[Signal[npt._32Bit], Any]:
    X, _, info = speech_meg.read_subject(subject=subject)
    transform_x: SignalProcessor[npt._32Bit] = instantiate(transform_x_cfg)
    X = transform_x(X)
    assert X.dtype == np.float32
    return X, info


def get_joint_mask(X: Signal, annot_type: str) -> npt.NDArray[np.bool_]:
    masks = (a.as_mask(X.sr, len(X)) for a in X.annotations if a.type == annot_type)
    return reduce(lambda x, y: np.logical_or(x, y), masks)


Loaders = dict[str, DataLoader]


def create_data_loaders(X: Signal[npt._32Bit], cfg: MainConfig) -> Loaders:
    speech_mask = get_joint_mask(X, "speech")
    covert_mask = get_joint_mask(X, "covert")
    log.info(f"True class ratio: {np.sum(speech_mask)/len(speech_mask)}")
    log.debug(f"{speech_mask.shape=}, {speech_mask=}")

    Y_joint = np.logical_or(speech_mask, covert_mask)[:, np.newaxis].astype("float32")
    dataset = Continuous(np.asarray(X), Y_joint, cfg.lag_backward, cfg.lag_forward)

    X_no_overt = np.asarray(X)[np.logical_not(speech_mask), :]
    Y_no_overt = covert_mask[np.logical_not(speech_mask), np.newaxis].astype("float32")
    dataset_covert = Continuous(X_no_overt, Y_no_overt, cfg.lag_backward, cfg.lag_forward)

    train, test = dataset.train_test_split(cfg.train_test_ratio)
    _, test_covert = dataset_covert.train_test_split(cfg.train_test_ratio)

    dl_params = dict(batch_size=cfg.batch_size, shuffle=True, drop_last=True)
    train_ldr = DataLoader(train, **dl_params)  # type: ignore
    test_ldr = DataLoader(test, **dl_params)  # type: ignore
    covert_test_ldr = DataLoader(test_covert, **dl_params)  # type: ignore
    return dict(train=train_ldr, test=test_ldr, covert=covert_test_ldr)


def get_metrics(model: nn.Module, loss: LossFunction, ldrs: Loaders, n: int) -> ParamsDict:
    metrics = {}
    for stage, ldr in ldrs.items():
        tr = trange(n, desc=f"Evaluating model on {stage}")
        eval_iter = limited(TestIter(model, ldr, loss)).by(tr)
        metrics[stage] = asdict(eval_model(eval_iter, metrics_cls, n))

    log.debug(f"{metrics=}")
    return flatten_dict(metrics, sep="/")


@log_execution_time()
@hydra.main(config_path="./configs", config_name="classification_overtcovert_config")
def main(cfg: MainConfig) -> None:
    main_utils.prepare_script(log, cfg)

    X, info = read_data(cfg.dataset.transform_x, cfg.subject)
    log.info(f"Loaded X: {str(X)}")
    ldrs = create_data_loaders(X, cfg)

    model = SimpleNet(cfg.model)
    if torch.cuda.is_available():
        model = model.cuda()

    # model_dt = "2022-07-08_00-36-48"
    # model_path = (
    #     Path(get_original_cwd())
    #     / "outputs"
    #     / "classif"
    #     / "debug:True"
    #     / "MEG"
    #     / model_dt
    #     / "model_dumps"
    #     / "SimpleNet.pth"
    # )
    # model.load_state_dict(torch.load(model_path))  # type: ignore
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    loss = nn.MSELoss()

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
