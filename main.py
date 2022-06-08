import logging
import os
import os.path
from time import perf_counter

import hydra
from hydra.utils import instantiate
from omegaconf import OmegaConf

import setup_utils
from library.bench_models_regression import BenchModelRegressionBase
from library.runner_regression import run_regression

log = logging.getLogger(__name__)

setup_utils.setup_hydra()


@hydra.main(config_path="./configs", config_name="config")
def main(cfg):
    if cfg.run_tensorboard:
        setup_utils.run_tensorboard("runs")
    if cfg.runner.debug:
        setup_utils.set_debug_level()
        OmegaConf.resolve(cfg)
        log.debug(OmegaConf.to_yaml(cfg))
    for dir_name in ["results", "model_dumps"]:
        if not os.path.isdir(dir_name):
            os.makedirs(dir_name)
            log.debug(f"{dir_name} dir created")
    log.info("Loading data...")
    t1 = perf_counter()
    dataset, detect_voice, info = instantiate(cfg.dataset)
    t2 = perf_counter()
    log.info(f"Loading data finished in {t2 - t1:.2f} sec.")

    model = instantiate(cfg.model.regression)
    bench_model = BenchModelRegressionBase(model, cfg.model.learning_rate)
    run_regression(bench_model, dataset, cfg.runner, detect_voice)

    t3 = perf_counter()
    log.info(f"Model training finished in {t3 - t2:.2f} sec")
    log.info(f"Overall session time: {t3 - t1:.2f} sec")


if __name__ == "__main__":
    main()
