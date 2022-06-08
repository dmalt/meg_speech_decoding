from typing import Optional

import numpy as np
import numpy.typing as npt
import scipy.signal as scs  # type: ignore

from . import Signal, SignalArray, T


class ButterFiltFilt:
    def __init__(self, order: int, l_freq: Optional[float] = None, h_freq: Optional[float] = None):
        self.order = order
        self.l_freq = l_freq
        self.h_freq = h_freq

    def __call__(self, signal: Signal[T]) -> Signal[T]:
        """
        Filter in lowpass, highpass or bandpass mode depending on
        the presence of lowcut and highcut arguments

        """
        data = signal.data
        if self.l_freq is None and self.h_freq is not None:
            data = self._lowpass(signal)
        elif self.h_freq is None and self.l_freq is not None:
            data = self._highpass(signal)
        elif self.h_freq is not None and self.l_freq is not None:
            data = self._bandpass(signal)
        return signal.update(data)

    def _lowpass(self, signal: Signal[T]) -> npt.NDArray["np.floating[T]"]:
        b, a = scs.butter(self.order, self.h_freq, btype="low", fs=signal.sr)
        return scs.filtfilt(b, a, signal, axis=0).astype(signal.dtype)

    def _highpass(self, signal: Signal[T]) -> npt.NDArray["np.floating[T]"]:
        b, a = scs.butter(self.order, self.l_freq, btype="highpass", fs=signal.sr)
        return scs.filtfilt(b, a, signal.data, axis=0).astype(signal.dtype)

    def _bandpass(self, signal: Signal[T]) -> npt.NDArray["np.floating[T]"]:
        freqs = (self.l_freq, self.h_freq)
        b, a = scs.butter(self.order, freqs, btype="bandpass", fs=signal.sr)
        return scs.filtfilt(b, a, signal.data, axis=0).astype(signal.dtype)


def moving_avarage(signal: Signal, window: int) -> Signal:
    w2 = int(window // 2)
    padded: SignalArray = np.pad(signal.data, (w2, w2), "constant", constant_values=0)
    pad_cumsum = np.cumsum(padded)
    data = (pad_cumsum[w2 * 2 :] - pad_cumsum[: -w2 * 2]) / window  # pyright: ignore
    return Signal(data, signal.sr)


def hilbert_envelope(signal: Signal[T]) -> Signal[T]:
    data = np.abs(scs.hilbert(signal, axis=0))
    return signal.update(data)


def notch_narrow(signal: Signal[T], freq: float) -> Signal[T]:
    return iir_filtfilt_notch(signal, freq, freq)


def notch_wide(signal: Signal[T], freq: float) -> Signal[T]:
    return iir_filtfilt_notch(signal, freq, freq // 2)


def iir_filtfilt_notch(signal: Signal[T], freq: float, Q: float) -> Signal[T]:
    b, a = scs.iirnotch(freq, Q=Q, fs=signal.sr)
    data = scs.filtfilt(b, a, signal, axis=0).astype(signal.dtype)
    return signal.update(data)
