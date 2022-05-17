from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import librosa as lb  # type: ignore
import librosa.feature as lbf  # type: ignore
import numpy as np
import numpy.typing as npt
import scipy.signal as scs  # type: ignore


def remove_eyes_artifacts(signal: npt.NDArray, sr: float) -> npt.NDArray:
    return Filter(signal, sr).highpass(lowcut=10, order=5).signal


def remove_target_leakage(signal: npt.NDArray, sr: float) -> npt.NDArray:
    return Filter(signal, sr).lowpass(highcut=200, order=5).signal


def remove_silent_noise(sound: npt.NDArray, sr: float) -> npt.NDArray:
    NOISE_THRESHOLD = 0.01
    window = int(sr / 2)
    smoothed_amp = Filter(np.abs(sound), sr).moving_avarage(window).signal
    sound_without_noise: npt.NDArray = np.copy(sound)
    sound_without_noise[smoothed_amp < NOISE_THRESHOLD] = -1
    return sound_without_noise


@dataclass
class Filter():
    """
    Inplace filtering tools

    Parameters
    ----------
    signal: array of shape(n_samples, ...)
        Target signal
    sr: float
        Sampling rate

    """

    signal: npt.NDArray
    sr: float

    def lowpass(self, highcut: float, order: int) -> "Filter":
        b, a = scs.butter(order, highcut, btype="low", fs=self.sr)
        self.signal = scs.filtfilt(b, a, self.signal, axis=0)
        return self

    def highpass(self, lowcut: float, order: int) -> "Filter":
        b, a = scs.butter(order, lowcut, btype="highpass", fs=self.sr)
        self.signal = scs.filtfilt(b, a, self.signal, axis=0)
        return self

    def bandpass(self, lowcut: float, highcut: float, order: int) -> "Filter":
        freqs = (lowcut, highcut)
        b, a = scs.butter(order, freqs, btype="bandpass", fs=self.sr)
        self.signal = scs.filtfilt(b, a, self.signal, axis=0)
        return self

    def moving_avarage(self, window: int) -> "Filter":
        w2 = int(window // 2)
        padded: npt.NDArray = np.pad(self.signal, (w2, w2), "constant", constant_values=0)
        pad_cumsum = np.cumsum(padded)
        self.signal = (pad_cumsum[w2 * 2 :] - pad_cumsum[: -w2 * 2]) / window
        return self

    def log_envelope(self, highcut: float = 200, lowcut: float = 200) -> "Filter":
        self.lowpass(highcut=highcut, order=3)
        self.signal = np.log(np.abs(self.signal))
        self.highpass(lowcut, order=5)
        return self

    def envelope(self) -> "Filter":
        self.signal = np.abs(scs.hilbert(self.signal, axis=0))
        return self

    def notch(self, freq: float, Q: float) -> "Filter":
        b, a = scs.iirnotch(freq, Q=Q, fs=self.sr)
        self.signal = scs.filtfilt(b, a, self.signal, axis=0)
        return self

    def notch_narrow(self, freq: float) -> "Filter":
        return self.notch(freq, freq)

    def notch_wide(self, freq: float) -> "Filter":
        return self.notch(freq, freq // 2)


@dataclass
class SpectralFeatures():
    """
    Tools for spectral features computation

    Passed signal is not modified. Instead, the result is returned.

    Parameters
    ----------
    signal: array of shape(n_samples, ...)
        Target signal
    sr: float
        Sampling rate

    """

    signal: npt.NDArray
    sr: float

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
        apm_spec = np.abs(np.fft.fft(self.signal, n, axis=0))
        n = self.signal.shape[-1] if n is None else n
        freqs = np.fft.fftfreq(n, 1 / self.sr)
        end = len(freqs) // 2
        assert n // 2 == end, f"{n=}, {end=}, {self.signal.shape=}, {freqs=}"
        return freqs[:end], apm_spec[..., :end]

    def mfccs(self, d: int, out_nsamp: int, n_mfcc: int) -> npt.NDArray:
        m = lbf.mfcc(y=self.signal, sr=self.sr, hop_length=d, n_mfcc=n_mfcc)
        mfccs_resampled: npt.NDArray = scs.resample(x=m.T, num=out_nsamp)
        return mfccs_resampled

    def logmelspec(self, n: int, f_max: float, d: int) -> tuple[npt.NDArray[np.double], float]:
        melspec = lbf.melspectrogram(y=self.signal, sr=self.sr, n_mels=n, fmax=f_max, hop_length=d)
        return lb.power_to_db(melspec, ref=np.max).T, self.sr / d

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
