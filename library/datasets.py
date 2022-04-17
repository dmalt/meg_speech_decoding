import h5py  # type: ignore
import numpy as np  # type: ignore
from torch.utils.data import Dataset  # type: ignore


class EcogDataset(Dataset):
    def __init__(
        self,
        cfg,
        transform,
        target_transform,
        lag_backward: int,
        lag_forward: int,
    ):
        X, Y = [], []
        for filepath in cfg.files_list:
            data = self.read_h5_file(filepath)
            ecog = data[:, cfg.ecog_channels].astype("double")
            sound = data[:, cfg.sound_channel].astype("double")
            x = transform(ecog, cfg.sampling_rate)
            y = target_transform(sound, cfg.sampling_rate, x.shape[0])
            X.append(x)
            Y.append(y)

        self.X = np.concatenate(X, axis=0)
        self.Y = np.concatenate(Y, axis=0)
        self._n_samp = self.X.shape[0]
        self.lag_backward = lag_backward
        self.lag_forward = lag_forward
        assert self.__len__() > 0, (
            "No samples in the dataset, "
            + f"{self._n_samp=}, {self.lag_backward=}, {self.lag_backward=}"
        )

    def __len__(self):
        return self._n_samp - self.lag_backward - self.lag_forward

    def __getitem__(self, i):
        X = self.X[i : i + self.lag_forward + self.lag_backward + 1].T
        Y = self.Y[i + self.lag_backward]
        return X, Y

    def read_h5_file(self, filepath):
        with h5py.File(filepath, "r+") as input_file:
            return input_file["RawData"]["Samples"][()]
