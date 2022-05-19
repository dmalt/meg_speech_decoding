from __future__ import annotations

from itertools import product

import numpy as np  # type: ignore
import scipy.signal as scs  # type: ignore
import sklearn.preprocessing as skp  # type: ignore
import torch  # type: ignore
from torch.autograd import Variable  # type: ignore
from torch.utils.data import DataLoader  # type: ignore
from tqdm import tqdm  # type: ignore

from .signal_processing import SpectralFeatures


class ModelInterpreter:
    def __init__(self, model, dataset, batch_size):
        self.model = model
        self.dataset = dataset
        self.batch_size = batch_size

    def get_x_unmixed(self) -> np.ndarray:
        X_unmixed = []
        data_loader = DataLoader(
            self.dataset, batch_size=self.batch_size, shuffle=False
        )
        i = 0
        for x_batch, _ in tqdm(data_loader):
            i += 1
            if i >= 3000:
                break
            self.model.eval()
            x_batch = Variable(torch.FloatTensor(x_batch))
            if torch.cuda.is_available():
                x_batch = x_batch.cuda()
            self.model(x_batch)
            # TODO: take only the last sample. worth fixing it
            # so batches are taken without overlap and we take a whole segment
            X_unmixed.append(self.model.get_unmixed_batch()[:, :, -1])
        return np.concatenate(X_unmixed, axis=0)

    def get_temporal(self, nperseg: int) -> tuple[np.ndarray, list, list]:
        """
        Get temporal filter weights in frequency domain

        Parameters
        ----------
        nperseg: int
            Number of samples per segment for psd

        Returns
        -------
        freqs: np.ndarray
            Frequencies for which spectral weights are computed
        spec_weights: list[np.ndarray]
            Model's filter weights in frequency domain
        spec_patterns: list[np.ndarray]
            Model's temporal patterns (filtered signal in freq domain)

        See also
        --------
        scipy.signal.welch

        """
        X, sr = self.get_x_unmixed(), self.dataset.info["sampling_rate"]

        unmixed_ch_cnt = X.shape[1]
        conv_weights = self.model.get_conv_filtering_weights()
        assert conv_weights.shape[0] % unmixed_ch_cnt == 0
        filt_per_ch = conv_weights.shape[0] // unmixed_ch_cnt

        spec_weights = []
        spec_patterns = []
        for i_u, i_f in product(range(unmixed_ch_cnt), range(filt_per_ch)):
            w = conv_weights[i_u * filt_per_ch + i_f, :]
            freqs, sw = SpectralFeatures(w, sr).asd(nperseg)
            sw = skp.minmax_scale(sw)
            x = X[:, i_u]
            _, x_psd = scs.welch(x, sr, nperseg=nperseg, detrend="linear")
            x_psd = skp.minmax_scale(x_psd[:-1])

            spec_weights.append(sw)
            spec_patterns.append(skp.minmax_scale(sw * x_psd))
        return freqs, spec_weights, spec_patterns

    def get_spatial_weigts(self) -> np.ndarray:
        spatial_weights = self.model.get_spatial()
        scaler = skp.StandardScaler()
        scaler.fit(self.dataset.X)
        spatial_weights /= scaler.var_[:, np.newaxis]
        return spatial_weights

    def get_spatial_patterns(self) -> list[np.ndarray]:
        SW = self.get_spatial_weigts()
        TW = self.model.get_conv_filtering_weights()
        patterns = []
        X_filt = np.zeros_like(self.dataset.X)
        for i_comp in range(TW.shape[0]):
            for i_ch in range(self.dataset.X.shape[1]):
                x = self.dataset.X[:, i_ch]
                X_filt[:, i_ch] = np.convolve(x, TW[i_comp, :], mode="same")
            patterns.append(np.cov(X_filt.T) @ SW[:, i_comp])
        return patterns

    def get_naive(self) -> list[np.ndarray]:
        spatial_weights = self.get_spatial_weigts()
        return list((np.cov(self.dataset.X.T) @ spatial_weights).T)
        # return list(spatial_weights.T @ np.cov(self.dataset.X.T))
