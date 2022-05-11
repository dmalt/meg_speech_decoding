import logging

import h5py  # type: ignore
import hydra  # type: ignore  # noqa
import librosa as lb  # type: ignore
import mne  # type: ignore
import numpy as np  # type: ignore
import scipy.signal as scs  # type: ignore
from torch.utils.data import Dataset  # type: ignore

from .common_preprocessing import TargetTransformer

log = logging.getLogger(__name__)


class TimeseriesDataset(Dataset):
    @classmethod
    def from_config(
        cls, patient, lag_backward, lag_forward, transform, target_transformer
    ):
        raise NotImplementedError

    def __init__(
        self,
        X,
        Y,
        lag_backward: int,
        lag_forward: int,
        target_transformer: TargetTransformer,
        info: dict,
    ):
        self.X = X
        self._n_samp = self.X.shape[0]
        self.Y = Y
        self.lag_backward = lag_backward
        self.lag_forward = lag_forward
        self.target_transformer = target_transformer
        self.info = info

    def train_test_split(self, ratio: float):
        train_size = int(self._n_samp * ratio)
        X_train, Y_train = self.X[:train_size], self.Y[:train_size]
        X_test, Y_test = self.X[train_size:], self.Y[train_size:]
        lb, lf = self.lag_backward, self.lag_forward
        tf, info = self.target_transformer, self.info
        dataset_train = self.__class__(X_train, Y_train, lb, lf, tf, info)
        dataset_test = self.__class__(X_test, Y_test, lb, lf, tf, info)
        return dataset_train, dataset_test

    def __len__(self):
        return self._n_samp - self.lag_backward - self.lag_forward

    def __getitem__(self, i):
        X = self.X[i : i + self.lag_forward + self.lag_backward + 1].T
        Y = self.Y[i + self.lag_backward]
        return X, Y


class SpeechDataset(TimeseriesDataset):
    def detect_voice(self, y_batch):
        return self.target_transformer.detect_voice(y_batch)


class EcogDataset(SpeechDataset):
    @classmethod
    def from_config(
        cls, patient, lag_backward, lag_forward, transform, target_transformer
    ):
        X, Y = [], []
        sr = patient.sampling_rate
        for filepath in patient.files_list:
            data = cls._read_h5_file(filepath)
            ecog = data[:, patient.ecog_channels].astype("double")
            sound = data[:, patient.sound_channel].astype("double")
            x = transform(ecog)
            y = target_transformer.transform(sound, sr, x.shape[0])
            X.append(x)
            Y.append(y)

        X = np.concatenate(X, axis=0)
        Y = np.concatenate(Y, axis=0)
        info = {"sampling_rate": transform.sampling_rate}
        return cls(X, Y, lag_backward, lag_forward, target_transformer, info)

    @staticmethod
    def _read_h5_file(filepath):
        with h5py.File(filepath, "r+") as input_file:
            return input_file["RawData"]["Samples"][()]


class MegDataset(SpeechDataset):
    @classmethod
    def from_config(
        cls, patient, transform, target_transformer, lag_backward, lag_forward
    ):
        raw = mne.io.read_raw_fif(
            patient.raw_path, verbose="ERROR", preload=True
        )
        raw.pick_types(meg=True)
        raw.notch_filter(freqs=(50, 100, 150))
        raw.filter(l_freq=10, h_freq=120)
        log.debug(raw.info)
        X = raw.get_data(reject_by_annotation="omit").T
        info = {"mne_info": raw.info, "sampling_rate": raw.info["sfreq"]}
        log.info(f"Data length={len(X) / raw.info['sfreq']} sec")
        del raw
        X = transform(X)
        n_samp = len(X)
        sound, sound_sr = lb.load(patient.audio_align_path, sr=None)
        log.info(f"Sound length={len(sound) / sound_sr:.2f} sec, {sound_sr=}")
        Y = target_transformer.transform(sound, sound_sr, n_samp)
        return cls(X, Y, lag_backward, lag_forward, target_transformer, info)


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
        transform,
        target_transformer,
        lag_backward,
        lag_forward,
    ):
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
        X, Y = transform(X), target_transformer.transform(Y)
        info = {
            "mixing_matrix": mix_matr,
            "sampling_rate": sr,
            "target_lag": lag,
        }
        return cls(X, Y, lag_backward, lag_forward, target_transformer, info)

    @classmethod
    def gen_signal(
        cls, sig_len: int, src_dim: int, sen_dim: int, target_lag: int
    ):
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
        return mixed_signals, target, mixing_matrix

    @classmethod
    def gen_noise(cls, sig_len, sen_dim, noise_dim):
        from colorednoise import powerlaw_psd_gaussian  # type: ignore

        noisy_signals = powerlaw_psd_gaussian(cls.BETA, sig_len * noise_dim)
        noisy_signals = noisy_signals.reshape((sig_len, noise_dim))

        assert (
            len(cls.noisy_filters) == noise_dim
        ), f"{len(cls.noisy_filters)} != {noise_dim}"
        filt_noisy_signals = cls.filter_signals(
            noisy_signals, cls.noisy_filters
        )
        noise_mixing_matrix = np.random.uniform(0, 1, (noise_dim, sen_dim))
        return np.matmul(filt_noisy_signals, noise_mixing_matrix)

    @staticmethod
    def filter_signals(signals, filters):
        number_of_signals = signals.shape[1]
        assert number_of_signals == len(filters)
        filtered_signals = np.copy(signals)
        for index, single_filter in enumerate(filters):
            if single_filter is None:
                filtered_signals[:, index] = signals[:, index]
                continue
            filtered_signals[:, index] = np.convolve(
                signals[:, index], single_filter, mode="same"
            )
        return filtered_signals

    @staticmethod
    def get_envelopes(signals):
        enveloped_signals = np.zeros(signals.shape)
        for i in range(signals.shape[1]):
            enveloped_signals[:, i] = np.abs(scs.hilbert(signals[:, i]))
        return enveloped_signals
