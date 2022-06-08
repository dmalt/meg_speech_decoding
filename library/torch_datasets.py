from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Sequence, TypeVar

import numpy as np
import numpy.typing as npt
import scipy.signal as scs  # type: ignore
from torch.utils.data import Dataset

from .signal import Signal, SignalArray
from .type_aliases import ChannelsVector32, SignalArray32_T

log = logging.getLogger(__name__)

TDataset = TypeVar("TDataset", bound="Continuous")


@dataclass
class Continuous(Dataset):
    """
    Parameters
    ----------
    X : float32 array of shape(n_samples, n_channels)
    Y : float32 array of shape(n_samples, n_features)
    lag_backward : int
        Number of samples before the target sample to include in one chunk
    lag_forward : int
        Number of samples after the target sample to include in one chunk

    """

    X: SignalArray[npt._32Bit]
    Y: SignalArray[npt._32Bit]
    lag_backward: int
    lag_forward: int

    def __len__(self) -> int:
        return len(self.X) - self.lag_backward - self.lag_forward

    def __getitem__(self, i: int) -> tuple[SignalArray32_T, ChannelsVector32]:
        X = self.X[i : i + self.lag_forward + self.lag_backward + 1].T
        Y = self.Y[i + self.lag_backward]
        return X, Y

    def train_test_split(self: TDataset, ratio: float) -> tuple[TDataset, TDataset]:
        train_size = int(len(self.X) * ratio)
        X_train, Y_train = self.X[:train_size], self.Y[:train_size]
        X_test, Y_test = self.X[train_size:], self.Y[train_size:]
        dataset_train = self.__class__(X_train, Y_train, self.lag_backward, self.lag_forward)
        dataset_test = self.__class__(X_test, Y_test, self.lag_backward, self.lag_forward)
        return dataset_train, dataset_test


@dataclass
class Composite(Dataset):
    sampling_rate: float
    datasets: Sequence[Continuous]

    def __len__(self) -> int:
        return sum(len(d) for d in self.datasets)

    def __getitem__(self, i: int) -> tuple[SignalArray32_T, ChannelsVector32]:
        if i > len(self):
            raise IndexError(f"Index {i} is out of bounds for dataset of size {len(self)}")
        for d in self.datasets:
            if i // len(d):
                i -= len(d)
            else:
                return d[i]
        raise IndexError("Dataset is empty")

    def train_test_split(self, ratio: float) -> tuple[Composite, Composite]:
        datasets_train, datasets_test = [], []
        cumlen = 0.0
        for d in self.datasets:
            thislen = len(d) / len(self)
            if cumlen + thislen < ratio:
                datasets_train.append(d)
                cumlen += thislen
            else:
                datasets_test.append(d)
        train = self.__class__(self.sampling_rate, datasets_train)
        test = self.__class__(self.sampling_rate, datasets_test)
        log.debug(f"{len(train)=}, {len(test)=}")
        return train, test

    @property
    def X(self) -> SignalArray[npt._32Bit]:
        return np.concatenate([d.X for d in self.datasets], axis=0).astype("float32")

    @property
    def Y(self) -> SignalArray[npt._32Bit]:
        return np.concatenate([d.Y for d in self.datasets], axis=0).astype("float32")


class SimulatedDataset(Continuous):
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
        transform: Callable[[Signal, float], tuple[Signal32, float]],
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
            "target_lag": lag,
        }
        return cls(X, Y, lag_backward, lag_forward, new_sr, target_transform.detect_voice, info)

    @classmethod
    def gen_signal(
        cls, sig_len: int, src_dim: int, sen_dim: int, target_lag: int
    ) -> tuple[Array, Array, Array]:
        signals = np.random.normal(0, 1, (sig_len, src_dim))
        assert len(cls.filters) == src_dim
        filtered_signals = cls.filter_signals(signals, cls.filters)

        mixing_matrix = np.random.uniform(0, 1, (src_dim, sen_dim))
        mixed_signals = np.matmul(filtered_signals, mixing_matrix)

        weight_matrix = np.random.uniform(0, 1, (target_lag, src_dim))

        envelopes = cls.get_envelopes(filtered_signals)
        target = scs.convolve2d(
            np.pad(envelopes, pad_width=((2, 2), (0, 0)), mode="edge"),  # type: ignore
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
