from __future__ import annotations

import logging
from functools import partial, reduce
from typing import Callable

import numpy as np
import numpy.typing as npt
import scipy.interpolate as sci  # type: ignore
import scipy.signal as scs  # type: ignore
import sklearn.preprocessing as skp  # type: ignore

from ..signal import Signal, T, drop_bad_segments
from ..type_aliases import SignalArray32
from .filtering import ButterFiltFilt, moving_avarage, notch_narrow, notch_wide
from .spectral import logmelspec

log = logging.getLogger(__name__)

SignalProcessor = Callable[[Signal[T]], Signal[T]]


def compose_processors(*functions: SignalProcessor) -> SignalProcessor:
    return reduce(lambda f, g: lambda x: g(f(x)), functions)


def log_envelope(signal: Signal[T], highcut: float = 200, lowcut: float = 200) -> Signal[T]:
    f = compose_processors(
        ButterFiltFilt(order=3, h_freq=highcut),
        lambda s: s.update(np.log(np.abs(s))),
        ButterFiltFilt(5, l_freq=lowcut),
    )
    return f(signal)


def remove_eyes_artifacts(signal: Signal[T]) -> Signal[T]:
    filter = ButterFiltFilt(order=5, l_freq=10)
    return filter(signal)


def remove_target_leakage(signal: Signal[T]) -> Signal[T]:
    filter = ButterFiltFilt(order=5, h_freq=200)
    return filter(signal)


def preprocess_meg(
    signal: Signal[T],
    selected_channels: list[int] | None,
    l_freq: float | None,
    h_freq: float | None,
    notch_freqs: list[float],
) -> Signal[T]:

    pipeline = compose_processors(
        partial(select_channels, selected_channels=selected_channels),
        ButterFiltFilt(order=5, l_freq=l_freq, h_freq=h_freq),
        compose_processors(*(partial(notch_narrow, freq=f) for f in notch_freqs)),
        drop_bad_segments,
        scale,
    )
    return pipeline(signal)


def select_channels(signal: Signal[T], selected_channels: list[int] | None) -> Signal[T]:
    if selected_channels is not None:
        return signal.update(signal.data[:, selected_channels])  # pyright: ignore
    return signal


def scale(signal: Signal[T]) -> Signal[T]:
    scaled_data = skp.scale(signal.data.astype("float64"))
    return signal.update(scaled_data.astype(signal.dtype))


# TODO: everything below need refactoring
def remove_silent_noise(sound: Signal[T]) -> npt.NDArray:
    NOISE_THRESHOLD = 0.01
    window = int(sound.sr / 2)
    smoothed_amp = moving_avarage(sound.update(np.abs(sound)), window)
    sound_without_noise: npt.NDArray = np.copy(sound)
    sound_without_noise[smoothed_amp.data < NOISE_THRESHOLD] = -1
    return sound_without_noise


def preprocess_ecog(
    ecog: Signal[T],
    dsamp_coef: int,
    highpass: float | None,
    lowpass: float | None,
    notch_narrow_freqs: list[float],
    notch_wide_freqs: list[float],
    selected_channels: list[int] | None,
) -> Signal[T]:
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


def melspectrogram_pipeline(
    signal: Signal[npt._32Bit], n_mels: int, f_max: float, dsamp_coef: int
) -> Signal[npt._32Bit]:
    signal.data /= np.max(np.abs(signal.data))
    melspec = logmelspec(signal, n_mels, f_max, dsamp_coef)
    melspec = drop_bad_segments(melspec)
    log.debug(f"{melspec.data.shape=}")
    # Converting to float64 before scaling is necessary: otherwise it's not enough
    # precision which results in a warning about numerical error from sklearn
    melspec_scaled = skp.scale(melspec.data.astype(np.float64)).astype(np.float32)
    log.debug(f"{melspec_scaled.data.shape=}")
    if melspec_scaled.ndim == 1:
        melspec_scaled = melspec_scaled[:, np.newaxis]
    return Signal(melspec_scaled, melspec.sr, signal.annotations)


def align_samples(sig_from: Signal[T], sig_to: Signal[T]) -> Signal[T]:
    log.info("Aligning samples")
    log.info(f"{sig_from=}")
    log.info(f"{sig_to=}")
    if len(sig_from) == len(sig_to) and sig_from.sr == sig_from.sr:
        return sig_to
    samp_from = np.arange(len(sig_from))
    interp_params = dict(bounds_error=False, fill_value="extrapolate", axis=0)
    itp = sci.interp1d(samp_from, sig_from.data, **interp_params)
    samp_to = (np.arange(len(sig_to)) * (sig_from.sr / sig_to.sr)).astype(int)
    interp = itp(samp_to)
    return Signal(interp, sig_to.sr, sig_to.annotations)
