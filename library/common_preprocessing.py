from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol

import librosa as lb  # type: ignore
import librosa.feature as lbf  # type: ignore
import numpy as np  # type: ignore
import scipy  # type: ignore
import scipy.interpolate as sci  # type: ignore
import scipy.io  # type: ignore
import scipy.signal  # type: ignore
import sklearn  # type: ignore
import sklearn.preprocessing  # type: ignore
from joblib import Memory  # type: ignore

memory = Memory("/home/altukhov/Data/speech/cachedir", verbose=0)


def notch_filtering_simple(ecog, frequency):
    ecog_filtered = ecog
    for w0 in [48, 49, 50, 100, 150]:
        norch_b, norch_a = scipy.signal.iirnotch(w0, Q=10, fs=frequency)
        ecog_filtered = scipy.signal.filtfilt(
            norch_b, norch_a, ecog_filtered, axis=0
        )
    return ecog_filtered


def notch_filtering_advanced(ecog, frequency):
    ecog_filtered = ecog
    w0_narrow = set(range(100, 200, 50)).union(set(range(30, 300, 30)), [70])
    w0_wide = set(range(200, 350, 50)).union(set([50]))
    for w0 in w0_narrow:
        norch_b, norch_a = scipy.signal.iirnotch(w0, Q=w0, fs=frequency)
        ecog_filtered = scipy.signal.filtfilt(
            norch_b, norch_a, ecog_filtered, axis=0
        )
    for w0 in w0_wide:
        # print(w0)
        norch_b, norch_a = scipy.signal.iirnotch(w0, Q=w0 // 2, fs=frequency)
        ecog_filtered = scipy.signal.filtfilt(
            norch_b, norch_a, ecog_filtered, axis=0
        )
    return ecog_filtered


def extract_sound_log_envelope(sound, frequency):
    LOW_PASS_FREQUENCY = 200
    HIGH_PASS_FREQUENCY = 200
    blp, alp = scipy.signal.butter(
        3, LOW_PASS_FREQUENCY / (frequency / 2), btype="low", analog=False
    )
    bhp, ahp = scipy.signal.butter(
        5, HIGH_PASS_FREQUENCY / (frequency / 2), btype="high", analog=False
    )

    sound_filtered = scipy.signal.filtfilt(bhp, ahp, sound)
    envelope = scipy.signal.filtfilt(blp, alp, np.log(np.abs(sound_filtered)))
    return envelope


def envelope_signals(signals):
    enveloped_signals = np.zeros(signals.shape)
    for i in range(signals.shape[1]):
        enveloped_signals[:, i] = np.abs(scipy.signal.hilbert(signals[:, i]))
    return enveloped_signals


def remove_eyes_artifacts(ecog, frequency, high_pass_frequency):
    bgamma, agamma = scipy.signal.butter(
        5, high_pass_frequency / (frequency / 2), btype="high"
    )
    return scipy.signal.filtfilt(bgamma, agamma, ecog, axis=0)


def remove_target_leakage(ecog, frequency, low_pass_frequency):
    bgamma, agamma = scipy.signal.butter(
        5, low_pass_frequency / (frequency / 2), btype="low"
    )
    return scipy.signal.filtfilt(bgamma, agamma, ecog, axis=0)


def moving_avarage(signal, window):
    half_window = int(window / 2)
    cumsum_with_padding = np.cumsum(
        np.pad(
            signal, (half_window, half_window), "constant", constant_values=0
        )
    )
    smoothed = (
        cumsum_with_padding[half_window * 2 :]
        - cumsum_with_padding[: -half_window * 2]
    ) / window
    assert (
        smoothed.shape[0] == signal.shape[0]
    ), f"{smoothed.shape[0]}!={signal.shape[0]}"
    return smoothed


def remove_silent_noise(sound, frequency):
    NOISE_THRESHOLD = 0.01
    sound_power_smoothed = moving_avarage(np.abs(sound), int(frequency * 0.5))
    sound_without_noise = np.copy(sound)
    sound_without_noise[sound_power_smoothed < NOISE_THRESHOLD] = -1
    return sound_without_noise


def extract_lpcs(sound, order, window_size, hop_size, out_length):
    half_window = int(window_size / 2)
    sound_padded = np.pad(sound, (half_window, half_window))
    lpcs = [
        lb.lpc(sound_padded[i - half_window : i + half_window + 1], order)
        for i in range(
            half_window, len(sound_padded) - half_window + 1, hop_size
        )
    ]
    lpcs = np.array(lpcs)[:, 1:]
    # todo: remove this terrible ifs
    if lpcs.shape[0] == out_length:
        return lpcs
    elif lpcs.shape[0] - 1 == out_length:
        return lpcs[:-1]
    else:
        raise ValueError
    return


def extract_mfccs(sound, sampling_rate, downsampling_coef, out_length, n_mfcc):
    mfccs = lbf.mfcc(
        y=sound, sr=sampling_rate, hop_length=downsampling_coef, n_mfcc=n_mfcc
    )
    mfccs_resampled = scipy.signal.resample(x=mfccs.T, num=out_length)
    return mfccs_resampled


class MegProcessor:
    def __call__(self, data):
        return sklearn.preprocessing.scale(data, copy=True).astype("float32")


class Scaler:
    def __call__(self, data):
        return self.transform(data)

    def transform(self, data):
        scaler = sklearn.preprocessing.StandardScaler()
        return scaler.fit_transform(data).astype("float32")

    def detect_voice(self, y_batch):
        # return np.sum(y_batch > 1, axis=1) > 5
        return None


@dataclass
class ClassicEcogPipeline:
    dsamp_coef: int
    lowpass: float
    highpass: float
    selected_channels: list[int]
    sampling_rate: int

    def __post_init__(self):
        self._transform = memory.cache(self._transform)

    def __call__(self, ecog):
        return self._transform(ecog, self.sampling_rate)

    def _transform(self, ecog, sr):
        ecog = scipy.signal.decimate(ecog, self.dsamp_coef, axis=0)
        new_sr = int(sr / self.dsamp_coef)
        ecog = remove_eyes_artifacts(ecog, new_sr, self.highpass)
        ecog = notch_filtering_advanced(ecog, new_sr)
        ecog = remove_target_leakage(ecog, new_sr, self.lowpass)
        ecog = sklearn.preprocessing.scale(ecog, copy=False)
        return ecog.astype("float32")[:, self.selected_channels]


class TargetTransformer(Protocol):
    def transform(self, sound, sr, out_length):
        pass

    def detect_voice(self, y_batch):
        pass


@dataclass
class ClassicMelspectrogramPipeline:
    dsamp_coef: int
    n_mels: int
    f_max: float

    def __post_init__(self):
        self.transform = memory.cache(self.transform)

    def transform(self, sound, sr, out_length):
        sound /= np.max(np.abs(sound))
        m = self._extract_melspectrogram(sound, sr, out_length)
        del sound
        m = sklearn.preprocessing.scale(m).astype("float32")
        if m.ndim == 1:
            m = m.reshape((-1, 1))
        return m

    def _extract_melspectrogram(self, sound, sr, out_length):
        melspec = lbf.melspectrogram(
            y=sound,
            sr=sr,
            n_mels=self.n_mels,
            fmax=self.f_max,
            hop_length=self.dsamp_coef,
        )
        logmelspec = lb.power_to_db(melspec, ref=np.max).T
        # todo: remove this terrible ifs
        if logmelspec.shape[0] == out_length:
            return logmelspec
        elif logmelspec.shape[0] - 1 == out_length:
            return logmelspec[:-1]
        else:
            return self._interpolate_to_out_length(
                logmelspec, out_length, len(sound)
            )

    def _interpolate_to_out_length(self, y, out_length, orig_length):
        x = np.arange(0, orig_length, self.dsamp_coef)
        itp = sci.interp1d(
            x, y, bounds_error=False, fill_value="extrapolate", axis=0
        )
        sr_ratio = orig_length / out_length
        meg_samp = (np.arange(out_length) * sr_ratio).astype(int)
        return itp(meg_samp)

    def detect_voice(self, y_batch):
        return np.sum(y_batch > 1, axis=1) > int(self.n_mels * 0.25)


def classic_lpc_pipeline(
    sound, sampling_rate, downsampling_coef, ecog_size, order
):
    WIN_LENGTH = 1001
    sound /= np.max(np.abs(sound))
    lpcs = extract_lpcs(sound, order, WIN_LENGTH, downsampling_coef, ecog_size)
    lpcs = sklearn.preprocessing.scale(lpcs)
    return lpcs


def classic_mfcc_pipeline(
    sound, sampling_rate, downsampling_coef, ecog_size, n_mfcc
):
    sound /= np.max(np.abs(sound))
    mfccs = extract_mfccs(
        sound, sampling_rate, downsampling_coef, ecog_size, n_mfcc
    )
    mfccs = sklearn.preprocessing.scale(mfccs)
    return mfccs


def get_amplitude_spectrum(
    signal: np.ndarray, sr: float, n: Optional[int] = None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute amplitude spectrum for a signal array along the last axis

    Parameters
    ----------
    signal: array of shape(..., n_samples)
        Target signal
    sr: float
        Sampling rate
    n: int, optional
        Signal length in samples; if doesn't equal len(signal), the signal is
        cropped or padded to match n (see numpy.fft.fft)

    Returns
    -------
    freqs: array
        1D array of frequencies
    amp_spce: array
        Amplitude spectrum

    See also
    --------
    numpy.fft.fft
    numpy.fft.fftfreq

    """
    apm_spec = np.abs(np.fft.fft(signal, n))
    n = signal.shape[-1] if n is None else n
    freqs = np.fft.fftfreq(n, 1 / sr)
    end = len(freqs) // 2
    assert n // 2 == end, f"{n=}, {end=}, {signal.shape=}, {freqs=}"
    return freqs[:end], apm_spec[..., :end]
