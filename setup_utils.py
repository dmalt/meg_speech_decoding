import logging

from hydra.core.config_store import ConfigStore  # type: ignore
from omegaconf import ListConfig, OmegaConf  # type: ignore

from library import datasets


def setup_hydra():
    OmegaConf.register_new_resolver("len", lambda x: len(x))
    OmegaConf.register_new_resolver("python_range", lambda x: ListConfig(list(eval(x))))

    cs = ConfigStore.instance()
    cs.store(group="dataset/patient", name="meg_patient_config", node=datasets.MegPatientConfig)


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
