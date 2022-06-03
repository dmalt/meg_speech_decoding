from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

import librosa as lb  # type: ignore
import librosa.feature as lbf  # type: ignore
import numpy as np
import numpy.typing as npt
import scipy.interpolate as sci  # type: ignore
import scipy.signal as scs  # type: ignore
import sklearn.preprocessing as skp  # type: ignore

from .type_aliases import Array32, Signal, Signal32

log = logging.getLogger(__name__)


def remove_eyes_artifacts(signal: Signal) -> npt.NDArray:
    return Filter(signal).highpass(lowcut=10, order=5).signal.data


def remove_target_leakage(signal: Signal) -> npt.NDArray:
    return Filter(signal).lowpass(highcut=200, order=5).signal.data


def remove_silent_noise(sound: Signal) -> npt.NDArray:
    NOISE_THRESHOLD = 0.01
    window = int(sound.sr / 2)
    smoothed_amp = Filter(Signal(np.abs(sound.data), sound.sr)).moving_avarage(window).signal
    sound_without_noise: npt.NDArray = np.copy(sound.data)
    sound_without_noise[smoothed_amp.data < NOISE_THRESHOLD] = -1
    return sound_without_noise


@dataclass
class Filter:
    """
    Inplace filtering tools

    Parameters
    ----------
    signal: Signal
        Target signal

    """

    signal: Signal

    def lowpass(self, highcut: float, order: int) -> "Filter":
        b, a = scs.butter(order, highcut, btype="low", fs=self.signal.sr)
        self.signal.data = scs.filtfilt(b, a, self.signal.data, axis=0)
        return self

    def highpass(self, lowcut: float, order: int) -> "Filter":
        b, a = scs.butter(order, lowcut, btype="highpass", fs=self.signal.sr)
        self.signal.data = scs.filtfilt(b, a, self.signal.data, axis=0)
        return self

    def bandpass(self, lowcut: float, highcut: float, order: int) -> "Filter":
        freqs = (lowcut, highcut)
        b, a = scs.butter(order, freqs, btype="bandpass", fs=self.signal.sr)
        self.signal.data = scs.filtfilt(b, a, self.signal.data, axis=0)
        return self

    def filter(self, lowcut: Optional[float], highcut: Optional[float], order: int) -> "Filter":
        """
        Filter in lowpass, highpass or bandpass mode depending on
        the presence of lowcut and highcut arguments

        """
        if lowcut is None and highcut is not None:
            self.lowpass(highcut, order=order)
        elif highcut is None and lowcut is not None:
            self.highpass(lowcut, order=order)
        elif highcut is not None and lowcut is not None:
            self.bandpass(lowcut, highcut, order=order)
        return self

    def moving_avarage(self, window: int) -> "Filter":
        w2 = int(window // 2)
        padded: npt.NDArray = np.pad(self.signal.data, (w2, w2), "constant", constant_values=0)
        pad_cumsum = np.cumsum(padded)
        self.signal.data = (pad_cumsum[w2 * 2 :] - pad_cumsum[: -w2 * 2]) / window
        return self

    def log_envelope(self, highcut: float = 200, lowcut: float = 200) -> "Filter":
        self.lowpass(highcut=highcut, order=3)
        self.signal.data = np.log(np.abs(self.signal.data))
        self.highpass(lowcut, order=5)
        return self

    def envelope(self) -> "Filter":
        self.signal.data = np.abs(scs.hilbert(self.signal.data, axis=0))
        return self

    def notch(self, freq: float, Q: float) -> "Filter":
        b, a = scs.iirnotch(freq, Q=Q, fs=self.signal.sr)
        self.signal.data = scs.filtfilt(b, a, self.signal.data, axis=0)
        return self

    def notch_narrow(self, freq: float) -> "Filter":
        return self.notch(freq, freq)

    def notch_wide(self, freq: float) -> "Filter":
        return self.notch(freq, freq // 2)


@dataclass
class SpectralFeatures:
    """
    Tools for spectral features computation

    Passed signal is not modified. Instead, the result is returned.

    Parameters
    ----------
    signal: Signal
        Target signal

    """

    signal: Signal

    def asd(self, n: Optional[int] = None) -> tuple[npt.NDArray[Any], npt.NDArray]:
        """
        Compute amplitude spectral density along the last axis

        Parameters
        ----------
        n: int, optional
            Signal length in samples; if doesn't equal len(signal), the signal
            is cropped or padded to match n (see numpy.fft.fft)

        Returns
        -------
        freqs: array
            1D array of frequencies
        amp_spec: array
            Amplitude spectrum

        See also
        --------
        numpy.fft.fft
        numpy.fft.fftfreq

        """
        apm_spec = np.abs(np.fft.fft(self.signal.data, n, axis=0))
        n = self.signal.data.shape[-1] if n is None else n
        freqs = np.fft.fftfreq(n, 1 / self.signal.sr)
        end = len(freqs) // 2
        assert n // 2 == end, f"{n=}, {end=}, {self.signal.data.shape=}, {freqs=}"
        return freqs[:end], apm_spec[..., :end]

    def mfccs(self, d: int, out_nsamp: int, n_mfcc: int) -> npt.NDArray:
        m = lbf.mfcc(y=self.signal, sr=int(self.signal.sr), hop_length=d, n_mfcc=n_mfcc)
        mfccs_resampled: npt.NDArray = scs.resample(x=m.T, num=out_nsamp)
        return mfccs_resampled

    def logmelspec(self, n: int, f_max: float, d: int) -> tuple[npt.NDArray[np.double], float]:
        melspec = lbf.melspectrogram(
            y=self.signal, sr=int(self.signal.sr), n_mels=n, fmax=f_max, hop_length=d
        )
        return lb.power_to_db(melspec, ref=np.max).T, self.signal.sr / d  # pyright: ignore


def preprocess_meg(
    signal: Signal,
    selected_channels: Optional[list[int]],
    lowpass: Optional[float],
    highpass: Optional[float],
    notch_freqs: list[float],
) -> Signal32:

    if selected_channels is not None:
        signal.data = signal.data[:, selected_channels]  # pyright: ignore
    filt = Filter(signal)
    filt.filter(highpass, lowpass, order=5)
    for f in notch_freqs:
        filt.notch_narrow(f)
    res = skp.scale(filt.signal.data.astype("float64")).astype("float32")
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

    filt = Filter(ecog)
    filt.filter(highpass, lowpass, order=5)
    for p in notch_narrow_freqs:
        filt.notch_narrow(p)
    for p in notch_wide_freqs:
        filt.notch_wide(p)

    return Signal32(skp.scale(filt.signal).astype("float32"), new_sr)


def melspectrogram_pipeline(signal, n_mels, f_max, dsamp_coef):
    signal.data /= np.max(np.abs(signal.data))
    sf = SpectralFeatures(signal)
    melspec, sr_new = sf.logmelspec(n_mels, f_max, dsamp_coef)
    # Converting to float64 before scaling is necessary: otherwise it's not enough
    # precision which results in a warning about numerical error from sklearn
    melspec_scaled: Array32 = skp.scale(melspec.astype(np.float64)).astype(np.float32)
    if melspec.ndim == 1:
        melspec = melspec.reshape((-1, 1))
    return melspec_scaled, sr_new


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
# def lpcs(self, order: int, window: int, d: int, out_nsamp: int) -> Optional[npt.NDArray]:
#     w2 = window // 2
#     pad: npt.NDArray = np.pad(self.signal, (w2, w2))
#     r = [lb.lpc(pad[i - w2 : i + w2 + 1], order) for i in range(w2, len(pad) - w2 + 1, d)]
#     res: npt.NDArray = np.array(r)[:, 1:]
#     # todo: remove this terrible ifs
#     if res.shape[0] == out_nsamp:
#         return res
#     elif res.shape[0] - 1 == out_nsamp:
#         return res[:-1]
#     else:
#         raise ValueError
