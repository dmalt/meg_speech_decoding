from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import librosa as lb  # type: ignore
import mne  # type: ignore
import numpy as np
import numpy.typing as npt
from mne.io import BaseRaw  # type: ignore

from ..signal import Annotation, Annotations, Signal, Signal1D

log = logging.getLogger(__name__)


@dataclass
class Info:
    """Meg speech dataset info"""

    mne_info: mne.Info


@dataclass
class Subject:
    """MEG subject configuration"""

    raw_path: str
    audio_path: str
    annotations_path: Optional[str]


def read(subject: Subject) -> tuple[Signal[npt._32Bit], Signal1D[npt._32Bit], Info]:
    X, info = _read_raw(subject.raw_path, subject.annotations_path)
    Y = _read_wav(subject.audio_path)
    Y.annotations = X.annotations
    assert abs(X.duration - Y.duration) < 0.01, "inconsistent durations for audio and MEG"
    return X, Y, info


def _read_wav(path: str, sr: int | None = None) -> Signal1D[npt._32Bit]:
    data, sr_final = lb.load(path, sr=sr)  # pyright: ignore
    return Signal1D(data[:, np.newaxis], sr_final, [])


def _read_raw(raw_path: str, annot_path: str | None) -> tuple[Signal[npt._32Bit], Info]:
    raw = mne.io.read_raw_fif(raw_path, verbose="ERROR", preload=True)
    if annot_path is not None:
        annots = mne.read_annotations(annot_path)
        raw.set_annotations(annots)
    X_data = raw.get_data(picks="meg").astype("float32").T
    return Signal(X_data, raw.info["sfreq"], _annotations_from_raw(raw)), Info(raw.info)


def _annotations_from_raw(raw: BaseRaw) -> Annotations:
    if not hasattr(raw, "annotations"):
        return []
    onsets: list[float] = list(raw.annotations.onset)
    durations: list[float] = list(raw.annotations.duration)
    types: list[str] = list(raw.annotations.description)
    onsets = [o - raw.first_samp / raw.info["sfreq"] for o in raw.annotations.onset]
    return [Annotation(o, d, t) for o, d, t, in zip(onsets, durations, types)]
