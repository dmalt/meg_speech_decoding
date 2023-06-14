# Quickstart

We tested the instructions on Ubuntu Linux, but in principle they should work
on any platform. Although Windows and MacOS will require different conda
enviroment setup (see the note in step 3).

Installation prerequisites:

- [Anaconda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/download.html)
- Git

## TL;DR (Linux)

Installation:

```bash
git clone --recurse-submodules https://github.com/dmalt/meg_speech_decoding.git && \
pip install dvc dvc[gdrive] && \
cd meg_speech_decoding/speech_meg && \
dvc pull -r test --glob "rawdata/derivatives/*/sub-test.dvc" && \
cd .. && \
conda env create -f environment_freeze.yml && \
conda activate speechdl3.9 && \
pip install --no-deps -e neural_data_preprocessing && \
pip install --no-deps -e speech_meg
```

Launch:

```bash
python regression_speech.py +experiment=test
```

## Installation

1. Clone this project with submodules:

    ```bash
    git clone --recurse-submodules https://github.com/dmalt/meg_speech_decoding.git
    ```

2. Load the test data

    Use DVC to load the data stored on GDrive.
    Install dvc and gdrive extension:

    ```bash
    pip install dvc dvc[gdrive]
    ```

    From `meg_speech_decoding/speech_meg` folder run

    ```bash
    dvc pull -r test --glob "rawdata/derivatives/*/sub-test.dvc"
    ```

    N.B.

    > The data download should start after gmail account authentification.
    > You'll see some warnings about "some cache files not existing locally nor
    > on remote". This is normal behaviour: test remote stores no subjects but
    > the test one and doesn't contain other files which dvc expects to find.

    More info on the loaded data structure [here](https://github.com/dmalt/speech_meg)

3. Setup and activate conda virtual env

    From `meg_speech_decoding` folder run:

    ```bash
    conda env create -f environment_freeze.yml
    conda activate speechdl3.9
    ```

    N.B.

    > The frozen enviroment file is for Linux since conda packages are not
    > cross-platform. On Windows or MacOS use `conda env create -f enviroment.yml`,
    > which will solve the environment for you. Note, that solving the enviroment
    > with conda might take forever (not tested). In case conda freezes we
    > recommend trying [mamba](https://mamba.readthedocs.io/en/latest/) instead.

4. Install the submodules

    From `meg_speech_decoding` folder run:

    ```bash
    pip install --no-deps -e neural_data_preprocessing
    pip install --no-deps -e speech_meg
    ```

## Launch

Again, make sure `meg_speech_decoding` is the current working directory.

Launch training for regression with:

```bash
python regression_speech.py +experiment=test
```

Launch training for classification with:

```bash
python classification_overtcovert.py +experiment=test
```

The script will save model dump, tensorboard stats, logs etc. in `outputs/`
under unuque date and time subfolders.

## Configuration

- Main configuration file for regression: `configs/regression_speech_config.yaml`
- Main configuration file for classification: `configs/classification_overtcovert_config.yaml`

Configuration files are available at `configs/`.

Main configuration file for each script determines how
[hydra](https://hydra.cc/) assembles the final configuration from the files in
`configs`. This final configuration allows hydra to generate the `CLI params`
that we can pass to the launch scripts.
