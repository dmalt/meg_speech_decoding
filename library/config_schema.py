from dataclasses import dataclass
from typing import Any

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
class MainConfig:
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


def get_selected_params(cfg: MainConfig) -> dict[str, Any]:
    return dict(
        n_steps=cfg.n_steps,
        lag_backward=cfg.lag_backward,
        lag_forward=cfg.lag_forward,
        target_features_cnt=cfg.target_features_cnt,
    )
