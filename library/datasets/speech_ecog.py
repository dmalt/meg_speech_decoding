from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Tuple

import h5py  # type: ignore
import numpy as np

from ..datasets import Continuous
from ..signal_processing import align_samples
from ..type_aliases import Signal, VoiceDetector
from . import Lags, Transformers

log = logging.getLogger(__name__)


@dataclass
class SubjectConfig:
    sampling_rate: float
    files_list: list[str]
    ecog_channels: list[int]
    sound_channel: int


@dataclass
class Info:
    pass


ContinuousDatasetPackage = Tuple[Continuous, VoiceDetector, Info]


def read(patient: SubjectConfig, lags: Lags, t: Transformers) -> ContinuousDatasetPackage:
    """Generate Continuous instance for ecog"""
    Xs, Ys = [], []
    sr = patient.sampling_rate
    for f in patient.files_list:
        ecog, sound = _read(f, patient.ecog_channels, patient.sound_channel)
        log.debug(f"{len(ecog) / sr=:.3f}, {len(sound) / sr=:.3f}")
        x, new_sr = t.transform_x(ecog, sr)
        y, new_sr_sound = t.transform_y(sound, sr)
        y = align_samples(y, new_sr_sound, x, new_sr)
        assert len(x) == len(y)
        Xs.append(x)
        Ys.append(y)

    X = np.concatenate(Xs, axis=0)
    Y = np.concatenate(Ys, axis=0)
    info = Info()
    detect_voice = t.transform_y.detect_voice
    return Continuous(X, Y, lags.backward, lags.forward, new_sr), detect_voice, info


def _read(h5_path: str, ecog_chs: list[int], sound_ch: int) -> tuple[Signal, Signal]:
    """Read ecog and audio signal from h5 file"""
    with h5py.File(h5_path, "r+") as input_file:
        data = input_file["RawData"]["Samples"][()]
    ecog = data[:, ecog_chs].astype("double")
    sound = data[:, sound_ch].astype("double")
    return ecog, sound
