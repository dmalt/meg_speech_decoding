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

Training is launched via `main.py`'s CLI interface, which is provided by [hydra](https://hydra.cc/), e.g.
```
python main.py +experiment=meg_all_sensors
```

Model dump, tensorboard stats, logs etc. are saved in `outputs/` under
unuque date and time subfolders.

Configuration
-------------
- Possible configurations are available at `configs/`.
- Changes to configuration should be made via creating an experiment in `configs/experiment/`


Interpretable weights visualization
-----------------------------------
Select model's date and time in `run_interpreter.py` and launch it with

```
python run_interpreter.py
```

Ecog data download
------------------

Ecog data can be downloaded from GDrive with
```
./download_data.sh
```

Files are automatically saved under `$HOME/Data/speech_dl/Procenko`.
Make sure to change the configuration for launch
