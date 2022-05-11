from omegaconf import ListConfig, OmegaConf  # type: ignore

OmegaConf.register_new_resolver("len", lambda x: len(x))
OmegaConf.register_new_resolver(
    "python_range", lambda x: ListConfig(list(eval(x)))
)

