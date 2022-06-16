import logging
import os
import os.path
from time import perf_counter

import hydra
import numpy as np
import numpy.typing as npt
from hydra.utils import call, instantiate
from joblib import Memory  # type: ignore
from ndp.signal import Signal, Signal1D
from ndp.signal.pipelines import Signal1DProcessor, SignalProcessor, align_samples
from omegaconf import OmegaConf

import setup_utils
from library.bench_models_regression import BenchModelRegressionBase
from library.models_regression import SimpleNet
from library.runner_regression import run_regression
from library.torch_datasets import Continuous

log = logging.getLogger(__name__)

setup_utils.setup_hydra()
memory = Memory("/home/altukhov/Data/speech/cachedir", verbose=0)


@hydra.main(config_path="./configs", config_name="config")
def main(cfg: setup_utils.Config) -> None:
    log.debug(f"{os.getcwd()=}")
    if cfg.debug:
        setup_utils.set_debug_level()
        OmegaConf.resolve(cfg)  # type: ignore
        log.debug(OmegaConf.to_yaml(cfg))
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
