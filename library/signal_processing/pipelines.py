from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import numpy.typing as npt
import scipy.interpolate as sci  # type: ignore
import scipy.signal as scs  # type: ignore
import sklearn.preprocessing as skp  # type: ignore
from library.io.signal_annotations import Annotations  # type: ignore

from ..type_aliases import Array32, Signal, Signal32, as_float32
from .filtering import filter, highpass, lowpass, moving_avarage, notch_narrow, notch_wide
from .spectral import logmelspec

log = logging.getLogger(__name__)


def remove_eyes_artifacts(signal: Signal) -> npt.NDArray:
    return highpass(signal, lowcut=10, order=5).data


def remove_target_leakage(signal: Signal) -> npt.NDArray:
    return lowpass(signal, highcut=200, order=5).data


def remove_silent_noise(sound: Signal) -> npt.NDArray:
    NOISE_THRESHOLD = 0.01
    window = int(sound.sr / 2)
    smoothed_amp = moving_avarage(Signal(np.abs(sound.data), sound.sr), window)
    sound_without_noise: npt.NDArray = np.copy(sound.data)
    sound_without_noise[smoothed_amp.data < NOISE_THRESHOLD] = -1
    return sound_without_noise


def preprocess_meg(
    signal: Signal,
    selected_channels: Optional[list[int]],
    lowpass: Optional[float],
    highpass: Optional[float],
    notch_freqs: list[float],
    annotations: Annotations,
) -> Signal32:

    if selected_channels is not None:
        signal.data = signal.data[:, selected_channels]  # pyright: ignore
    signal = filter(signal, highpass, lowpass, order=5)
    for f in notch_freqs:
        signal = notch_narrow(signal, f)
    res = skp.scale(signal.data.astype("float64")).astype("float32")
    return Signal32(res, signal.sr)


def preprocess_ecog(
    ecog: Signal32,
    dsamp_coef: int,
    highpass: Optional[float],
    lowpass: Optional[float],
    notch_narrow_freqs: list[float],
    notch_wide_freqs: list[float],
    selected_channels: Optional[list[int]],
) -> Signal32:
    if selected_channels is not None:
        ecog.data = ecog.data[:, selected_channels]  # pyright: ignore
    ecog = scs.decimate(ecog.data, dsamp_coef, axis=0)
    new_sr = int(ecog.sr / dsamp_coef)

    ecog = as_float32(filter(ecog, highpass, lowpass, order=5))
    for p in notch_narrow_freqs:
        ecog = as_float32(notch_narrow(ecog, p))
    for p in notch_wide_freqs:
        ecog = as_float32(notch_wide(ecog, p))

    return Signal32(skp.scale(ecog).astype("float32"), new_sr)


def melspectrogram_pipeline(signal, n_mels, f_max, dsamp_coef):
    signal.data /= np.max(np.abs(signal.data))
    melspec = logmelspec(signal, n_mels, f_max, dsamp_coef)
    # Converting to float64 before scaling is necessary: otherwise it's not enough
    # precision which results in a warning about numerical error from sklearn
    melspec_scaled: Array32 = skp.scale(melspec.data.astype(np.float64)).astype(np.float32)
    if melspec_scaled.ndim == 1:
        melspec_scaled = melspec_scaled[:, np.newaxis]
    return melspec_scaled, melspec.sr


def align_samples(sig_from: Signal, sig_to: Signal) -> Signal:
    log.info("Aligning samples")
    log.info(f"{sig_from=}")
    log.info(f"{sig_to=}")
    if len(sig_from) == len(sig_to) and sig_from.sr == sig_from.sr:
        return Signal32(sig_to.data.astype("float32"), sig_to.sr)
    samp_from = np.arange(len(sig_from))
    interp_params = dict(bounds_error=False, fill_value="extrapolate", axis=0)
    itp = sci.interp1d(samp_from, sig_from.data, **interp_params)
    samp_to = (np.arange(len(sig_to)) * (sig_from.sr / sig_to.sr)).astype(int)
    interp = itp(samp_to)
    return Signal(interp, sig_to.sr)
