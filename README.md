Quickstart
===========
Instructions here are tested on Ubuntu Linux, but should work on any platform, although Windows
and MacOS will require different conda enviroment setup (see the note in step 3).

Installation prerequisites:
- [Anaconda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/download.html)
- Git

TL;DR (Linux)
-------------
Installation:
```bash
git clone --recurse-submodules https://github.com/dmalt/ossadtchi-ml-test-bench-speech.git && \
pip install dvc dvc[gdrive] && \
cd ossadtchi-ml-test-bench-speech/speech_meg && \
dvc pull -r test --glob "**/sub-test.dvc" && \
cd .. && \
conda env create -f environment_freeze.yml && \
conda activate speechdl3.9 && \
pip install --no-deps -e neural_data_preprocessing && \
pip install --no-deps -e speech_meg
```

Launch:
```
python regression_speech.py +experiment=test
```

Installation
------------
1. Clone this project with submodules:

```bash
git clone --recurse-submodules https://github.com/dmalt/ossadtchi-ml-test-bench-speech.git
```

2. Load the test data

Use DVC to load the data stored on GDrive.
Install dvc and gdrive extension:
```bash
pip install dvc dvc[gdrive]
```

From `ossadtchi-ml-test-bench-/speech_meg` folder run
```
dvc pull -r test
```

N.B.
> The data download should start after gmail account authentification. You'll
> see some warnings about "some cache files not existing locally nor on
> remote". This is normal behaviour: test remote stores only the test subject
> and doesn't contain all the files which dvc expects to find.

More info on the loaded data structure [here](https://github.com/dmalt/speech_meg)

3. Setup and activate conda virtual env

From `ossadtchi-ml-test-bench-speech` folder run:
```bash
conda env create -f environment_freeze.yml
conda activate speechdl3.9
```

N.B.
> The frozen enviroment file is for Linux only since conda packages are not
> cross-platform. On Windows or MacOS use `conda env create -f enviroment.yml`,
> which will solve the environment for you. Note, that solving the enviroment
> with conda might take forever (not tested). In case conda freezes we
> recommend trying [mamba](https://mamba.readthedocs.io/en/latest/) instead.

4. Install the submodules

From `ossadtchi-ml-test-bench-speech` folder run:
```
pip install --no-deps -e neural_data_preprocessing
pip install --no-deps -e speech_meg
```


Launch
------
Again, make sure `ossadtchi-ml-test-bench-speech` is the current working directory.

Launch training for regression with:
```
python regression_speech.py +experiment=test
```

Launch training for classification with:
```
python classification_overtcovert.py +experiment=test
```

Model dump, tensorboard stats, logs etc. will be saved in `outputs/` under
unuque date and time subfolders.

Configuration
-------------
- Main configuration file for regression: `configs/regression_speech_config.yaml`
- Main configuration file for classification: `configs/classification_overtcovert_config.yaml`

Configuration files are available at `configs/`.

Main configuration file for each script determines how other configuration files in `configs`
are composed together to achieve the final configuration.

The `CLI params` that can be passed to the scripts are determined by the final
composed configuration. For details, check out [hydra documentation](https://hydra.cc/).
