import logging
from dataclasses import dataclass
from typing import Any

from hydra.core.config_store import ConfigStore
from omegaconf import ListConfig, OmegaConf

from library.datasets.speech_meg import Subject
from library.models_regression import SimpleNetConfig


@dataclass
class ReadConfig:
    _target_: str
    subject: Any


@dataclass
class DatasetConfig:
    subject: Any
    read: ReadConfig
    transform_x: Any
    transform_y: Any


@dataclass
class TrainRunnerConfig:
    runs_count: int  # not used
    batch_size: int
    max_iterations_count: int
    metric_iter: int
    early_stop_steps: int
    upd_evry_n_steps: int
    train_test_ratio: float
    learning_rate: float


@dataclass
class Config:
    debug: bool
    lag_backward: int
    lag_forward: int
    target_features_cnt: int
    selected_channels: list[int]
    run_tensorboard: bool
    train_runner: TrainRunnerConfig
    model: SimpleNetConfig
    dataset: DatasetConfig


def setup_hydra() -> None:
    OmegaConf.register_new_resolver("len", lambda x: len(x))
    OmegaConf.register_new_resolver("python_range", lambda x: ListConfig(list(eval(x))))
    cs = ConfigStore.instance()
    cs.store(group="dataset/subject", name="meg_subject_config", node=Subject)
    cs.store(group="dataset", name="dataset_schema", node=DatasetConfig)
    cs.store(name="config_schema", node=Config)


def set_debug_level() -> None:
    for logger_name in logging.root.manager.loggerDict:
        logging.getLogger(logger_name).setLevel(logging.DEBUG)


def run_tensorboard(logdir: str) -> None:
    """Modified from https://stackoverflow.com/a/61960273"""
    import os
    import sys
    import threading

    venv_dir = os.path.dirname(sys.executable)
    tb_thread = threading.Thread(
        target=lambda: os.system(f"{venv_dir}/tensorboard --logdir={logdir}"), daemon=True
    )
    tb_thread.start()
