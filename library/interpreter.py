from __future__ import annotations

import logging
from itertools import product

import numpy as np
import numpy.typing as npt
import scipy.signal as scs  # type: ignore
import sklearn.preprocessing as skp  # type: ignore
import torch
from ndp.signal import Signal
from ndp.signal.spectral import asd
from torch.autograd import Variable

from .func_utils import log_execution_time
from .models_regression import SimpleNet

log = logging.getLogger(__name__)


class ModelInterpreter:
    def __init__(self, model: SimpleNet, X: Signal[npt._32Bit]):
        self.model = model
        self.unmixing_layer = model.unmixing_layer
        self.X = X

    @log_execution_time(desc="unmixing signal")
    def unmix_signal(self) -> Signal[npt._32Bit]:
        self.unmixing_layer.eval()
        X_tensor = Variable(torch.FloatTensor(np.asarray(self.X).T))
        if torch.cuda.is_available():
            X_tensor = X_tensor.cuda()
        with torch.no_grad():
            X_unmixed = self.unmixing_layer(X_tensor).cpu().detach().numpy().T
        return self.X.update(X_unmixed)

    @log_execution_time(desc="getting temporal patterns")
    def get_temporal(self, nperseg: int) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray]]:
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
        X_unmixed = self.unmix_signal()

        conv_weights = self.model.get_conv_filtering_weights()
        assert conv_weights.shape[0] % X_unmixed.n_channels == 0
        filt_per_ch = conv_weights.shape[0] // X_unmixed.n_channels

        spec_weights = []
        spec_patterns = []
        for i_u, i_f in product(range(X_unmixed.n_channels), range(filt_per_ch)):
            w = conv_weights[i_u * filt_per_ch + i_f, :]
            freqs, sw = asd(Signal(w[:, np.newaxis], self.X.sr, annotations=[]), nperseg)
            log.debug(f"{sw.shape=}")
            sw = skp.minmax_scale(sw)
            x = X_unmixed.data[:, i_u]
            _, x_psd = scs.welch(x, self.X.sr, nperseg=nperseg, detrend="linear")
            x_psd = skp.minmax_scale(x_psd[:-1])
            log.debug(f"{x_psd.shape=}")

            spec_weights.append(sw)
            spec_patterns.append(skp.minmax_scale(sw[:, 0] * x_psd))
        return freqs, spec_weights, spec_patterns

    @log_execution_time(desc="getting spatial weights")
    def get_spatial_weigts(self) -> np.ndarray:
        spatial_weights = self.model.get_spatial()
        scaler = skp.StandardScaler()
        scaler.fit(self.X)
        spatial_weights /= scaler.var_[:, np.newaxis]  # pyright: ignore
        return spatial_weights

    @log_execution_time(desc="getting spatial patterns")
    def get_spatial_patterns(self) -> list[np.ndarray]:
        spectral_weights = self.get_spatial_weigts()
        TW = self.model.get_conv_filtering_weights()
        patterns = []
        X_filt = np.zeros_like(self.X)
        for i_comp in range(TW.shape[0]):
            for i_ch in range(self.X.n_channels):
                x = self.X.data[:, i_ch]
                X_filt[:, i_ch] = np.convolve(x, TW[i_comp, :], mode="same")
            patterns.append(np.cov(X_filt.T) @ spectral_weights[:, i_comp])
        return patterns

    @log_execution_time(desc="getting naive patterns")
    def get_naive(self) -> list[np.ndarray]:
        spatial_weights = self.get_spatial_weigts()
        return list((np.cov(self.X.data.T) @ spatial_weights).T)
        # return list(spatial_weights.T @ np.cov(self.dataset.X.T))
