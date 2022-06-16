import logging
import os
import os.path
import sys
from time import perf_counter

import hydra
import numpy as np
import numpy.typing as npt
from hydra.utils import call, instantiate
from ndp.signal import Signal, Signal1D
from ndp.signal.pipelines import Signal1DProcessor, SignalProcessor, align_samples
from omegaconf import OmegaConf

from library import git_utils, hydra_utils
from library.bench_models_regression import BenchModelRegressionBase
from library.models_regression import SimpleNet
from library.runner_regression import run_regression
from library.torch_datasets import Continuous

log = logging.getLogger(__name__)

hydra_utils.setup_hydra()


@hydra.main(config_path="./configs", config_name="config")
def main(cfg: hydra_utils.Config) -> None:
    log.debug(f"{os.getcwd()=}")
    if cfg.debug:
        hydra_utils.set_debug_level()
        OmegaConf.resolve(cfg)  # type: ignore
        log.debug(OmegaConf.to_yaml(cfg))
    elif git_utils.is_repo_clean():
        with open("commit_hash", "w") as f:
            f.write(git_utils.get_latest_commit_hash())
    else:
        log.warning("Git repository is not clean. Continue? (y/n)")
        while (ans := input("-> ")).lower() not in ("y", "n"):
            print("Please input 'y' or 'n'")
        log.info(f"Answer: {ans}")
        if ans == "n":
            sys.exit(0)

    for dir_name in ["results", "model_dumps"]:
        if not os.path.isdir(dir_name):
            os.makedirs(dir_name)
            log.debug(f"{dir_name} dir created")
    log.debug(f"{os.getcwd()=}")
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
    dataset = Continuous(np.asarray(X), np.asarray(Y), cfg.lag_backward, cfg.lag_forward)

    model = SimpleNet(cfg.model)
    bench_model = BenchModelRegressionBase(model, cfg.train_runner.learning_rate)
    run_regression(bench_model, dataset, cfg.train_runner, cfg.debug)

    t4 = perf_counter()
    log.info(f"Model training finished in {t4 - t3:.2f} sec")
    log.info(f"Overall session time: {t4 - t1:.2f} sec")


if __name__ == "__main__":
    main()
