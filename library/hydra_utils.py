import logging
from dataclasses import dataclass
from typing import Any

from hydra.core.config_store import ConfigStore
from ndp.datasets.speech_meg import Subject
from omegaconf import ListConfig, OmegaConf

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
    type: str


@dataclass
class Config:
    debug: bool
    lag_backward: int
    lag_forward: int
    target_features_cnt: int
    selected_channels: list[int]
    model: SimpleNetConfig
    dataset: DatasetConfig
    batch_size: int
    n_steps: int
    metric_iter: int
    model_upd_freq: int
    train_test_ratio: float
    learning_rate: float


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


