Installation
------------
1. Clone this project with submodules:

```bash
git clone --recurse-submodules https://github.com/dmalt/ossadtchi-ml-test-bench-speech.git
```


2. Setup conda virtual env with

```bash
cd ossadtchi-ml-test-bench-speech
conda env create -f environment_freeze.yml
```

Activate the environment with
```bash
conda activate speechdl3.9
```

3. Install the submodules:

```
cd neural_data_preprocessing
pip install --no-deps -e .
cd ../speech_meg
pip install --no-deps -e .
```

4. Load the data
Use DVC to load the data stored on GDrive
(requires authorization; the data folder must be shared with you).

This step can be done in a separate environment to avoid packages clutter with dvc
which depends on a lot of stuff.

Install dvc and gdrive extension:
```bash
pip install dvc dvc[gdrive]
```

From `speech_meg` folder run
```
dvc pull
```

More info on the loaded data structure [here](https://github.com/dmalt/speech_meg)

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

