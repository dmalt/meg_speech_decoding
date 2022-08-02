Installation
------------
Setup conda virtual env with

```bash
conda env create -f environment_freeze.yml
```

Activate the environment with
```bash
conda activate speechdl3.9
```

Launch
------
First, activate the environment with
```
conda activate speechdl3.9
```

Training for regression is launched via `regression_speech.py`'s CLI interface:
```
python regression_speech.py [CLI params]
```

Main configuration file: `configs/regression_speech_config.yaml`

The classification is launched via `classification_overtcovert.py`:
```
python classification_overtcovert.py [CLI params]
```

Main configuration file: `configs/classification_overtcovert_config.yaml`

Model dump, tensorboard stats, logs etc. are saved in `outputs/` under
unuque date and time subfolders.

Configuration
-------------
Configuration files are available at `configs/`.

Main configuration file for each script determines how other configuration files in `configs`
are composed together to achieve the final configuration.

The `CLI params` that can be passed to the script are determined by the final
composed configuration. For details, check out [hydra documentation](https://hydra.cc/).

