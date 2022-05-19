from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Optional, TypeVar

import numpy as np
import scipy.signal as scs  # type: ignore
from torch.utils.data import Dataset

from .common_preprocessing import TargetTransformer
from .config_schema import EcogPatientConfig, MegPatientConfig
from .io import read_audio, read_ecog, read_meg
from .signal_processing import align_samples
from .type_aliases import Array, Array32

log = logging.getLogger(__name__)

TDataset = TypeVar("TDataset", bound="SpeechDataset")


@dataclass
class SpeechDataset(Dataset):
    """
    Parameters
    ----------
    X : float32 array of shape(n_samples, n_channels)
    Y : float32 array of shape(n_samples, n_features)
    lag_backward : int
        Number of samples before the target sample to include in one chunk
    lag_forward : int
        Number of samples after the target sample to include in one chunk
    detect_voice : callable
        Function to get boolean mask of voice in transformed sound batch;
        accepts a batch of target data and returns a boolean mask array of
        the same size
    info: dict
        Any additional info, e.g. sampling rate,  mixing matrix for
        simulated dataset or raw.info for MEG

    """

    X: Array32
    Y: Array32
    lag_backward: int
    lag_forward: int
    detect_voice: Callable[[Array], Optional[Array]]
    info: dict[str, Any]

    def __post_init__(self) -> None:
        self._n_samp = self.X.shape[0]

    @classmethod
    def from_config(
        cls: type[TDataset],
        patient: Any,
        lag_backward: int,
        lag_forward: int,
        transform: Callable[[Array, float], tuple[Array32, float]],
        target_transform: TargetTransformer,
    ) -> TDataset:
        """Alternative constructor to generate instance from hydra configuration"""
        raise NotImplementedError

    def train_test_split(self: TDataset, ratio: float) -> tuple[TDataset, TDataset]:
        train_size = int(self._n_samp * ratio)
        X_train, Y_train = self.X[:train_size], self.Y[:train_size]
        X_test, Y_test = self.X[train_size:], self.Y[train_size:]
        lb, lf = self.lag_backward, self.lag_forward
        dv, info = self.detect_voice, self.info
        dataset_train = self.__class__(X_train, Y_train, lb, lf, dv, info)
        dataset_test = self.__class__(X_test, Y_test, lb, lf, dv, info)
        return dataset_train, dataset_test

    def __len__(self) -> int:
        return self._n_samp - self.lag_backward - self.lag_forward

    def __getitem__(self, i: int) -> tuple[Array, Array]:
        X = self.X[i : i + self.lag_forward + self.lag_backward + 1].T
        Y = self.Y[i + self.lag_backward]
        return X, Y


class EcogDataset(SpeechDataset):
    @classmethod
    def from_config(
        cls,
        patient: EcogPatientConfig,
        lag_backward: int,
        lag_forward: int,
        transform: Callable[[Array, float], tuple[Array32, float]],
        target_transform: TargetTransformer,
    ) -> EcogDataset:
        """Alternative constructor to generate EcogDataset instance from hydra configuration"""
        Xs, Ys = [], []
        sr = patient.sampling_rate
        for f in patient.files_list:
            ecog, sound = read_ecog(f, patient.ecog_channels, patient.sound_channel)
            x, new_sr = transform(ecog, sr)
            y, new_sr_sound = target_transform(sound, sr)
            y = align_samples(y, new_sr_sound, x, new_sr)
            assert len(x) == len(y)
            Xs.append(x)
            Ys.append(y)

        X = np.concatenate(Xs, axis=0)
        Y = np.concatenate(Ys, axis=0)
        info = {"sampling_rate": new_sr}
        return cls(X, Y, lag_backward, lag_forward, target_transform.detect_voice, info)


class MegDataset(SpeechDataset):
    @classmethod
    def from_config(
        cls,
        patient: MegPatientConfig,
        lag_backward: int,
        lag_forward: int,
        transform: Callable[[Array, float], tuple[Array32, float]],
        target_transform: TargetTransformer,
    ) -> MegDataset:

        X, mne_info = read_meg(patient.raw_path)
        X, new_sr = transform(X, mne_info["sfreq"])
        info = {"mne_info": mne_info, "sampling_rate": new_sr}

        Y, Y_sr = read_audio(patient.audio_align_path)
        log.debug(f"{Y.dtype=}")
        Y, new_sound_sr = target_transform(Y, Y_sr)  # , len(X))
        log.debug("Finished transforming target")
        Y = align_samples(Y, new_sound_sr, X, new_sr)
        assert len(X) == len(Y), f"{len(X)=} != {len(Y)=}"

        return cls(X, Y, lag_backward, lag_forward, target_transform.detect_voice, info)


class SimulatedDataset(SpeechDataset):
    RANDOM_SEED = 67
    FREQUENCY = 1000
    BETA = 1

    filters = [
        scs.firwin(100, [30, 80], fs=FREQUENCY, pass_zero=False),
        scs.firwin(100, [80, 120], fs=FREQUENCY, pass_zero=False),
        scs.firwin(100, [120, 170], fs=FREQUENCY, pass_zero=False),
        scs.firwin(100, [170, 220], fs=FREQUENCY, pass_zero=False),
    ]

    noisy_filters = [
        scs.firwin(100, [40, 70], fs=FREQUENCY, pass_zero=False),
        scs.firwin(100, [90, 110], fs=FREQUENCY, pass_zero=False),
        scs.firwin(100, [130, 160], fs=FREQUENCY, pass_zero=False),
        scs.firwin(100, [180, 210], fs=FREQUENCY, pass_zero=False),
    ] * 10

    @classmethod
    def from_config(
        cls,
        patient,
        lag_backward: int,
        lag_forward: int,
        transform: Callable[[Array, float], tuple[Array32, float]],
        target_transform: TargetTransformer,
    ) -> SimulatedDataset:
        from colorednoise import powerlaw_psd_gaussian  # type: ignore

        np.random.seed(cls.RANDOM_SEED)

        n_sig, n_sen, n_src = patient.sig_len, patient.sen_dim, patient.src_dim
        lag, sr = patient.target_lag, patient.sampling_rate

        sen_signals, Y, mix_matr = cls.gen_signal(n_sig, n_src, n_sen, lag)

        sen_noises = cls.gen_noise(n_sig, n_sen, patient.noise_dim)
        minor_noise = powerlaw_psd_gaussian(cls.BETA, n_sig * n_sen)
        minor_noise = minor_noise.reshape((n_sig, n_sen))

        X = sen_signals + 1 * sen_noises + 0.1 * minor_noise

        assert len(X) == len(Y), f"{len(X)} != {len(Y)}"
        assert np.linalg.matrix_rank(X) == n_sen

        log.debug(f"{X.shape=}, {Y.shape=}")
        (X, new_sr), (Y, _) = transform(X, sr), target_transform(Y, sr)
        info = {
            "mixing_matrix": mix_matr,
            "sampling_rate": new_sr,
            "target_lag": lag,
        }
        return cls(X, Y, lag_backward, lag_forward, target_transform.detect_voice, info)

    @classmethod
    def gen_signal(
        cls, sig_len: int, src_dim: int, sen_dim: int, target_lag: int
    ) -> tuple[Array, Array32, Array]:
        signals = np.random.normal(0, 1, (sig_len, src_dim))
        assert len(cls.filters) == src_dim
        filtered_signals = cls.filter_signals(signals, cls.filters)

        mixing_matrix = np.random.uniform(0, 1, (src_dim, sen_dim))
        mixed_signals = np.matmul(filtered_signals, mixing_matrix)

        weight_matrix = np.random.uniform(0, 1, (target_lag, src_dim))

        envelopes = cls.get_envelopes(filtered_signals)
        target = scs.convolve2d(
            np.pad(envelopes, pad_width=((2, 2), (0, 0)), mode="edge"),
            weight_matrix,
            mode="valid",
        )
        # reveal_type(target)
        return mixed_signals, target, mixing_matrix

    @classmethod
    def gen_noise(cls, sig_len: int, sen_dim: int, noise_dim: int) -> Array:
        from colorednoise import powerlaw_psd_gaussian

        noisy_signals = powerlaw_psd_gaussian(cls.BETA, sig_len * noise_dim)
        noisy_signals = noisy_signals.reshape((sig_len, noise_dim))

        assert len(cls.noisy_filters) == noise_dim, f"{len(cls.noisy_filters)} != {noise_dim}"
        filt_noisy_signals = cls.filter_signals(noisy_signals, cls.noisy_filters)
        noise_mixing_matrix = np.random.uniform(0, 1, (noise_dim, sen_dim))
        return np.matmul(filt_noisy_signals, noise_mixing_matrix)

    @staticmethod
    def filter_signals(signals: Array, filters: list) -> Array:
        number_of_signals = signals.shape[1]
        assert number_of_signals == len(filters)
        filtered_signals = np.copy(signals)
        for index, single_filter in enumerate(filters):
            if single_filter is None:
                filtered_signals[:, index] = signals[:, index]
                continue
            filtered_signals[:, index] = np.convolve(signals[:, index], single_filter, mode="same")
        return filtered_signals

    @staticmethod
    def get_envelopes(signals: Array) -> Array:
        enveloped_signals = np.zeros(signals.shape)
        for i in range(signals.shape[1]):
            enveloped_signals[:, i] = np.abs(scs.hilbert(signals[:, i]))
        return enveloped_signals
