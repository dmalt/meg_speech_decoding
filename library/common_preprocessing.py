from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Protocol

import numpy as np
import numpy.typing as npt
import scipy.interpolate as sci  # type: ignore
import scipy.signal as scs  # type: ignore
import sklearn.preprocessing as skp  # type: ignore
from joblib import Memory  # type: ignore

from . import signal_processing as sp
from .signal_processing import SpectralFeatures
from .type_aliases import Array, Array32

memory = Memory("/home/altukhov/Data/speech/cachedir", verbose=0)
log = logging.getLogger(__name__)


class Transformer(Protocol):
    def __call__(self, sound: Array, sr: float) -> tuple[Array32, float]:
        ...


class TargetTransformer(Transformer, Protocol):
    def detect_voice(self, y_batch: Array) -> Optional[npt.NDArray[np.bool_]]:
        ...


@dataclass
class MegProcessor:
    lowpass: float
    highpass: float
    notch_freqs: list[float]
    selected_channels: Optional[list[int]] = None

    def __call__(self, data: Array, sr: float) -> tuple[Array32, float]:
        if self.selected_channels is not None:
            data = data[:, self.selected_channels]
        filt = sp.Filter(data, sr)
        filt.bandpass(self.highpass, self.lowpass, order=5)
        for f in self.notch_freqs:
            filt.notch_narrow(f)
        return skp.scale(filt.signal).astype("float32"), sr


class Scaler:
    def __call__(self, data: Array, sr: float) -> tuple[Array32, float]:
        return skp.scale(data).astype("float32"), sr

    def detect_voice(self, y_batch: Array) -> None:
        return None


@dataclass
class ClassicEcogPipeline:
    dsamp_coef: int
    lowpass: float
    highpass: float
    notch_narrow_freqs: list[float]
    notch_wide_freqs: list[float]
    selected_channels: list[int]

    def __call__(self, ecog: Array, sr: float) -> tuple[Array32, float]:
        if self.selected_channels:
            ecog = ecog[:, self.selected_channels]
        ecog = scs.decimate(ecog, self.dsamp_coef, axis=0)
        new_sr = int(sr / self.dsamp_coef)

        filt = sp.Filter(ecog, new_sr)
        filt.bandpass(self.highpass, self.lowpass, order=5)
        for p in self.notch_narrow_freqs:
            filt.notch_narrow(p)
        for p in self.notch_wide_freqs:
            filt.notch_wide(p)

        return skp.scale(filt.signal).astype("float32"), new_sr


@dataclass
class ClassicMelspectrogramPipeline:
    dsamp_coef: int
    n_mels: int
    f_max: float

    def __call__(self, sound: Array, sr: float) -> tuple[Array32, float]:
        sound /= np.max(np.abs(sound))
        sf = SpectralFeatures(sound, sr)
        melspec, sr_new = sf.logmelspec(self.n_mels, self.f_max, self.dsamp_coef)
        del sound
        log.debug(
            f"{melspec.shape=}, {melspec.std(axis=0)=},"
            + f" {melspec.max(axis=0)=}, {melspec.mean(axis=0)=}"
            + f"{np.mean(melspec - melspec.mean(axis=0), axis=0)=}"
        )
        # melspec_scaled: Array32 = skp.scale(melspec).astype(np.float32)
        melspec_scaled: Array32 = melspec.astype(np.float32)
        log.debug(
            f"{melspec.shape=}, {melspec.std(axis=0)=},"
            + f" {melspec.max(axis=0)=}, {melspec.mean(axis=0)=}"
            + f"{np.mean(melspec - melspec.mean(axis=0), axis=0)=}"
        )
        if melspec.ndim == 1:
            melspec = melspec.reshape((-1, 1))
        return melspec_scaled, sr_new

    def detect_voice(self, y_batch: Array) -> npt.NDArray[np.bool_]:
        res: npt.NDArray[np.bool_] = np.sum(y_batch > 1, axis=1) > int(self.n_mels * 0.25)
        return res


def align_samples(x_from: Array, sr_from: float, x_to: Array, sr_to: float) -> Array32:
    if len(x_from) == len(x_to) and sr_from == sr_to:
        return x_from
    times_from = np.arange(len(x_from)) / sr_from
    itp = sci.interp1d(times_from, x_from, bounds_error=False, fill_value="extrapolate", axis=0)
    times_to = np.arange(len(x_to)) / sr_to
    interp: Array32 = itp(times_to).astype("float32")
    return interp


# def classic_lpc_pipeline(sound, sampling_rate, downsampling_coef, ecog_size, order):
#     WIN_LENGTH = 1001
#     sound /= np.max(np.abs(sound))
#     lpcs = sp.extract_lpcs(sound, order, WIN_LENGTH, downsampling_coef, ecog_size)
#     lpcs = skp.scale(lpcs)
#     return lpcs


# def classic_mfcc_pipeline(sound, sampling_rate, downsampling_coef, ecog_size, n_mfcc):
#     sound /= np.max(np.abs(sound))
#     mfccs = sp.extract_mfccs(sound, sampling_rate, downsampling_coef, ecog_size, n_mfcc)
#     mfccs = skp.scale(mfccs)
#     return mfccs
