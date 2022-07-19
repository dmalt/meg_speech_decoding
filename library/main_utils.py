import logging
import os

from hydra.core.config_store import ConfigStore
from hydra.core.global_hydra import GlobalHydra
from ndp.datasets.speech_meg import Subject
from omegaconf import ListConfig, OmegaConf

from library import git_utils

from .config_schema import DatasetConfig, MainConfig

log = logging.getLogger(__name__)


def setup_hydra() -> None:
    OmegaConf.register_new_resolver("len", lambda x: len(x))
    OmegaConf.register_new_resolver("python_range", lambda x: ListConfig(list(eval(x))))
    cs = ConfigStore.instance()
    cs.store(group="dataset/subject", name="meg_subject_config", node=Subject)
    cs.store(group="dataset", name="dataset_schema", node=DatasetConfig)
    cs.store(name="config_schema", node=MainConfig)


def set_debug_level() -> None:
    for logger_name in filter(
        lambda x: x.startswith(("library", "__main__")), logging.root.manager.loggerDict
    ):
        logging.getLogger(logger_name).setLevel(logging.DEBUG)


def create_dirs() -> None:
    if not os.path.isdir("model_dumps"):
        os.makedirs("model_dumps")
        log.debug("model_dumps dir created")


def dump_environment() -> None:
    env = os.popen("conda env export").read()
    assert env.startswith("name")
    with open("enironment_dump.yml", "w") as f:
        f.write(env)


def print_config(cfg: MainConfig) -> None:
    OmegaConf.resolve(cfg)  # type: ignore
    log.debug(OmegaConf.to_yaml(cfg))


def prepare_script(logger: logging.Logger, cfg: MainConfig) -> None:
    GlobalHydra.instance().clear()
    logger.debug(f"Current working directory is {os.getcwd()}")
    if cfg.debug:
        set_debug_level()
        print_config(cfg)
    create_dirs()
    dump_environment()
    git_utils.dump_commit_hash(cfg.debug)
