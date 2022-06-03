from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import librosa as lb  # type: ignore
import mne  # type: ignore
import numpy as np

from ..type_aliases import Signal32
from . import signal_annotations
from .signal_annotations import Annotations

log = logging.getLogger(__name__)


@dataclass
class Info:
    mne_info: mne.Info


@dataclass
class Subject:
    """MEG subject configuration"""
    raw_path: str
    audio_path: str
    annotations_path: Optional[str]


def read(subject: Subject) -> tuple[Signal32, Signal32, Info, Annotations]:
    raw = mne.io.read_raw_fif(subject.raw_path, verbose="ERROR", preload=True)
    if subject.annotations_path is not None:
        annots = mne.read_annotations(subject.annotations_path)
        raw.set_annotations(annots)
    X_data = raw.get_data(picks="meg").astype("float32").T
    X = Signal32(X_data, raw.info["sfreq"])
    Y = read_wav(subject.audio_path)
    return X, Y, Info(raw.info), signal_annotations.from_raw(raw)


def read_wav(path: str, sr=None) -> Signal32:
    data, sr = lb.load(path, sr=sr)  # pyright: ignore
    return Signal32(data[:, np.newaxis], sr)
