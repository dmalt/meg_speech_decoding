from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Protocol, Tuple

import numpy as np  # type: ignore
import scipy.interpolate as sci  # type: ignore
import scipy.signal as scs  # type: ignore
import sklearn.preprocessing as skp  # type: ignore
from joblib import Memory  # type: ignore

from . import signal_processing as sp
from .signal_processing import SpectralFeatures

memory = Memory("/home/altukhov/Data/speech/cachedir", verbose=0)
log = logging.getLogger(__name__)


SignalAndSrate = Tuple[np.ndarray, float]


class Transformer(Protocol):
    def __call__(self, sound: np.ndarray, sr: float) -> SignalAndSrate:
        ...


class TargetTransformer(Transformer, Protocol):
    def detect_voice(self, y_batch: np.ndarray) -> Optional[np.ndarray]:
        ...


@dataclass
class MegProcessor:
    lowpass: float
    highpass: float
    notch_freqs: list[float]
    selected_channels: Optional[list[int]] = None

    def __call__(self, data: np.ndarray, sr: float) -> SignalAndSrate:
        filt = sp.Filter(data, sr)
        filt.bandpass(self.highpass, self.lowpass, order=5)
        for f in self.notch_freqs:
            filt.notch_narrow(f)
        if self.selected_channels is None:
            return skp.scale(filt.signal).astype("float32")
        return skp.scale(filt.signal[:, self.selected_channels]).astype("float32"), sr


class Scaler:
    def __call__(self, data: np.ndarray, sr: float) -> tuple[np.ndarray, float]:
        return skp.scale(data).astype("float32"), sr

    def detect_voice(self, y_batch: np.ndarray) -> None:
        return None


@dataclass
class ClassicEcogPipeline:
    dsamp_coef: int
    lowpass: float
    highpass: float
    notch_narrow_freqs: list[float]
    notch_wide_freqs: list[float]
    selected_channels: list[int]

    def __call__(self, ecog: np.ndarray, sr: float) -> SignalAndSrate:
        ecog = scs.decimate(ecog, self.dsamp_coef, axis=0)
        new_sr = int(sr / self.dsamp_coef)

        filt = sp.Filter(ecog, new_sr)
        filt.bandpass(self.highpass, self.lowpass, order=5)
        for p in self.notch_narrow_freqs:
            filt.notch_narrow(p)
        for p in self.notch_wide_freqs:
            filt.notch_wide(p)

        ecog = skp.scale(filt.signal, copy=False)
        return ecog.astype("float32")[:, self.selected_channels], new_sr


@dataclass
class ClassicMelspectrogramPipeline:
    dsamp_coef: int
    n_mels: int
    f_max: float

    def __call__(self, sound: np.ndarray, sr: float) -> SignalAndSrate:
        sound /= np.max(np.abs(sound))
        sf = SpectralFeatures(sound, sr)
        melspec, sr_new = sf.logmelspec(self.n_mels, self.f_max, self.dsamp_coef)
        del sound
        log.debug(f"{melspec.shape=}")
        melspec = skp.scale(melspec).astype("float32")
        if melspec.ndim == 1:
            melspec = melspec.reshape((-1, 1))
        return melspec, sr_new

    def detect_voice(self, y_batch: np.ndarray) -> np.ndarray:
        return np.sum(y_batch > 1, axis=1) > int(self.n_mels * 0.25)


def align_samples(
    x_from: np.ndarray, sr_from: float, x_to: np.ndarray, sr_to: float
) -> np.ndarray:
    if len(x_from) == len(x_to) and sr_from == sr_to:
        return x_from
    times_from = np.arange(len(x_from)) / sr_from
    itp = sci.interp1d(times_from, x_from, bounds_error=False, fill_value="extrapolate", axis=0)
    times_to = np.arange(len(x_to)) / sr_to
    return itp(times_to).astype("float32")


def classic_lpc_pipeline(sound, sampling_rate, downsampling_coef, ecog_size, order):
    WIN_LENGTH = 1001
    sound /= np.max(np.abs(sound))
    lpcs = sp.extract_lpcs(sound, order, WIN_LENGTH, downsampling_coef, ecog_size)
    lpcs = skp.scale(lpcs)
    return lpcs


def classic_mfcc_pipeline(sound, sampling_rate, downsampling_coef, ecog_size, n_mfcc):
    sound /= np.max(np.abs(sound))
    mfccs = sp.extract_mfccs(sound, sampling_rate, downsampling_coef, ecog_size, n_mfcc)
    mfccs = skp.scale(mfccs)
    return mfccs
