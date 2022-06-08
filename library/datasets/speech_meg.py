from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import librosa as lb  # type: ignore
import mne  # type: ignore
import numpy as np
from mne.io import BaseRaw  # type: ignore

from ..signal import Annotation, Annotations
from ..type_aliases import Signal32

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
    X, info, annotations = _read_raw(subject.raw_path, subject.annotations_path)
    Y = _read_wav(subject.audio_path)
    return X, Y, info, annotations


def _read_wav(path: str, sr=None) -> Signal32:
    data, sr = lb.load(path, sr=sr)  # pyright: ignore
    return Signal32(data[:, np.newaxis], sr)


def _read_raw(raw_path: str, annot_path: Optional[str]) -> tuple[Signal32, Info, Annotations]:
    raw = mne.io.read_raw_fif(raw_path, verbose="ERROR", preload=True)
    if annot_path is not None:
        annots = mne.read_annotations(annot_path)
        raw.set_annotations(annots)
    X_data = raw.get_data(picks="meg").astype("float32").T
    return Signal32(X_data, raw.info["sfreq"]), Info(raw.info), _annotations_from_raw(raw)


def _annotations_from_raw(raw: BaseRaw) -> Annotations:
    if not hasattr(raw, "annotations"):
        return []
    onsets: list[float] = list(raw.annotations.onset)
    durations: list[float] = list(raw.annotations.duration)
    types: list[str] = list(raw.annotations.description)
    onsets = [o - raw.first_samp / raw.info["sfreq"] for o in raw.annotations.onset]
    return [Annotation(o, d, t) for o, d, t, in zip(onsets, durations, types)]
