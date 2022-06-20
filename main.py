import logging
import os
import os.path
import sys
from collections import Counter
from time import perf_counter

import hydra
import numpy as np
import numpy.typing as npt
import torch
from hydra.utils import call, instantiate
from ndp.signal import Signal, Signal1D
from ndp.signal.pipelines import Signal1DProcessor, SignalProcessor, align_samples
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import trange

from library import git_utils, hydra_utils
from library.models_regression import SimpleNet
from library.runner_common import get_random_predictions, infinite
from library.runner_regression import corr_multiple, loop_generator, run_experiment
from library.torch_datasets import Continuous

log = logging.getLogger(__name__)

hydra_utils.setup_hydra()


def prompt_proceeding_with_dirty_repo() -> None:
    log.warning("Git repository is not clean. Continue? (y/n)")
    while (ans := input("-> ")).lower() not in ("y", "n"):
        print("Please input 'y' or 'n'")
    log.info(f"Answer: {ans}")
    if ans == "n":
        sys.exit(0)


def handle_debug(debug: bool, cfg: hydra_utils.Config) -> None:
    if debug:
        hydra_utils.set_debug_level()
        OmegaConf.resolve(cfg)  # type: ignore
        log.debug(OmegaConf.to_yaml(cfg))
    elif git_utils.is_repo_clean():
        with open("commit_hash", "w") as f:
            f.write(git_utils.get_latest_commit_hash())
    else:
        prompt_proceeding_with_dirty_repo()


def create_dirs() -> None:
    for dir_name in ["results", "model_dumps"]:
        if not os.path.isdir(dir_name):
            os.makedirs(dir_name)
            log.debug(f"{dir_name} dir created")


def create_dataset(cfg: hydra_utils.Config) -> tuple[Continuous, float, float]:
    log.info("Loading data...")
    t1 = perf_counter()
    X: Signal[npt._32Bit]
    Y_: Signal1D[npt._32Bit]
    X, Y_, _ = call(cfg.dataset.read)
    log.info(f"Loaded X: {str(X)}")
    log.info(f"Loaded Y: {str(Y_)}")
    t2 = perf_counter()
    log.info(f"Loading data finished in {t2 - t1:.2f} sec.")
    log.debug(f"{os.getcwd()=}")

    log.info("Transforming data")
    transform_x: SignalProcessor[npt._32Bit] = instantiate(cfg.dataset.transform_x)
    transform_y: Signal1DProcessor[npt._32Bit] = instantiate(cfg.dataset.transform_y)
    X = transform_x(X)
    Y = transform_y(Y_)
    Y = align_samples(Y, X)
    assert X.dtype == np.float32
    assert Y.dtype == np.float32
    t3 = perf_counter()
    log.info(f"Transforming data finished in {t3 - t2:.2f} sec.")
    log.debug(f"{X.data.shape=}, {Y.data.shape=}")
    return Continuous(np.asarray(X), np.asarray(Y), cfg.lag_backward, cfg.lag_forward), t1, t3


@hydra.main(config_path="./configs", config_name="config")
def main(cfg: hydra_utils.Config) -> None:
    log.debug(f"Current working directory is {os.getcwd()}")
    handle_debug(cfg.debug, cfg)
    create_dirs()

    dataset, t1, t3 = create_dataset(cfg)
    model = SimpleNet(cfg.model)
    if torch.cuda.is_available():
        model = model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train_runner.learning_rate)

    tracker = SummaryWriter("tensorboard_events")  # type: ignore

    train, test = dataset.train_test_split(cfg.train_runner.train_test_ratio)
    bs = cfg.train_runner.batch_size
    train_generator = infinite(DataLoader(train, batch_size=bs, shuffle=True))
    test_generator = infinite(DataLoader(test, batch_size=bs, shuffle=True))
    run_experiment(model, optimizer, train_generator, test_generator, cfg.train_runner, tracker)

    metrics = Counter()
    for _, m in zip(trange(cfg.train_runner.metric_iter), loop_generator(model, train_generator)):
        metrics["train_corr"] += m["correlation"] / cfg.train_runner.metric_iter
        metrics["train_corr_speech"] += m["correlation_speech"] / cfg.train_runner.metric_iter
        metrics["train_loss"] += m["loss"] / cfg.train_runner.metric_iter
    for _, m in zip(trange(cfg.train_runner.metric_iter), loop_generator(model, test_generator)):
        metrics["test_corr"] += m["correlation"] / cfg.train_runner.metric_iter
        metrics["test_corr_speech"] += m["correlation_speech"] / cfg.train_runner.metric_iter
        metrics["test_loss"] += m["loss"] / cfg.train_runner.metric_iter

    tracker.add_hparams(
        dict(
            lag_backward=cfg.lag_backward,
            lag_forward=cfg.lag_forward,
            target_features_cnt=cfg.target_features_cnt,
        ),
        dict(metrics),
    )
    t4 = perf_counter()
    log.info(f"Model training finished in {t4 - t3:.2f} sec")
    log.info(f"Overall session time: {t4 - t1:.2f} sec")


if __name__ == "__main__":
    main()
