from collections.abc import MutableMapping
from dataclasses import dataclass
from typing import Any, Generator, Union

from omegaconf.omegaconf import OmegaConf


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
class FeatureExtractorConfig:
    in_channels: int
    downsampling: int
    hidden_channels: int
    filtering_size: int
    envelope_size: int


@dataclass
class SimpleNetConfig:
    out_channels: int
    lag_backward: int
    lag_forward: int
    use_lstm: bool
    feature_extractor: FeatureExtractorConfig


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
    subject: str
    plot_loaded: bool = False


SimpleType = Union[int, float, bool, str]
ParamsDict = dict[str, SimpleType]


def get_selected_params(cfg: MainConfig) -> ParamsDict:
    dict_conf = OmegaConf.to_container(cfg)
    assert isinstance(dict_conf, dict)
    return flatten_dict(dict_conf)


def flatten_dict(d: MutableMapping, parent_key: str = "", sep: str = ".") -> ParamsDict:
    return dict(_flatten_dict_gen(d, parent_key, sep))


KeyValueGenerator = Generator[tuple[str, SimpleType], None, None]


def _flatten_dict_gen(d: MutableMapping, parent_key: str, sep: str) -> KeyValueGenerator:
    """
    Modified from
    https://www.freecodecamp.org/news/how-to-flatten-a-dictionary-in-python-in-4-different-ways/

    """
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, MutableMapping):
            yield from _flatten_dict_gen(v, new_key, sep=sep)
        else:
            # Complex types are not supported by tensorboard hparams
            if type(v) not in (int, float, str, bool):
                v = str(v)
            yield new_key, v
