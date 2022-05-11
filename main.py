import logging
import os
import os.path
from time import perf_counter

import hydra  # type: ignore
from hydra.utils import instantiate  # type: ignore
from omegaconf import ListConfig, OmegaConf  # type: ignore

from library.bench_models_regression import BenchModelRegressionBase
from library.runner_classification import run_classification
from library.runner_regression import run_regression

log = logging.getLogger(__name__)

OmegaConf.register_new_resolver("len", lambda x: len(x))
OmegaConf.register_new_resolver(
    "python_range", lambda x: ListConfig(list(eval(x)))
)


def set_debug_level():
    for logger_name in logging.root.manager.loggerDict:
        logging.getLogger(logger_name).setLevel(logging.DEBUG)


def run_tensorboard(logdir):
    """Modified from https://stackoverflow.com/a/61960273"""
    import os
    import sys
    import threading

    venv_dir = os.path.dirname(sys.executable)
    tb_thread = threading.Thread(
        target=lambda: os.system(f"{venv_dir}/tensorboard --logdir={logdir}"),
        daemon=True,
    )
    tb_thread.start()


@hydra.main(config_path="./configs", config_name="config")
def main(cfg):
    if cfg.runner.debug:
        run_tensorboard("runs")
        set_debug_level()
        OmegaConf.resolve(cfg)
        log.debug(OmegaConf.to_yaml(cfg))
    for dir_name in ["results", "model_dumps"]:
        if not os.path.isdir(dir_name):
            os.makedirs(dir_name)
            log.debug(f"{dir_name} dir created")
    log.info("Loading data...")
    t1 = perf_counter()
    dataset = instantiate(cfg.dataset)
    t2 = perf_counter()
    log.info(f"Loading data finished in {t2 - t1:.2f} sec.")

    if "regression" in cfg.model:
        model = instantiate(cfg.model.regression)
        bench_model = BenchModelRegressionBase(model, cfg.model.learning_rate)
        run_regression(bench_model, dataset, cfg.runner)
    elif cfg.model == "classification":
        run_classification(cfg.pipeline.model, cfg.patient, cfg.debug)

    t3 = perf_counter()
    log.info(f"Model training finished in {t3 - t2:.2f} sec")
    log.info(f"Overall session time: {t3 - t1:.2f} sec")


if __name__ == "__main__":
    main()
