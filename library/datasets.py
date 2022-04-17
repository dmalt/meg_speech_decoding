import h5py  # type: ignore
import numpy as np  # type: ignore
from torch.utils.data import Dataset  # type: ignore


class EcogDataset(Dataset):
    def __init__(
        self,
        subject_cfg,
        transform,
        target_transform,
        lag_backward,
        lag_forward,
        selected_channels,
        downsampling_coef,
        lowpass,
        highpass,
        n_mels,
        f_max,
    ):
        X = []
        Y = []
        for filepath in subject_cfg.files_list:
            with h5py.File(filepath, "r+") as input_file:
                data = input_file["RawData"]["Samples"][()]

            ecog = data[:, subject_cfg.ecog_channels].astype("double")
            sound = data[:, subject_cfg.sound_channel].astype("double")

            x = transform(
                ecog,
                subject_cfg.sampling_rate,
                downsampling_coef,
                lowpass,
                highpass,
            ).astype("float32")[:, selected_channels]
            y = target_transform(
                sound,
                subject_cfg.sampling_rate,
                downsampling_coef,
                x.shape[0],
                n_mels,
                f_max,
            ).astype("float32")

            if len(y.shape) == 1:
                y = y.reshape((-1, 1))

            X.append(x)
            Y.append(y)

        self.X = np.concatenate(X, axis=0)
        self.Y = np.concatenate(Y, axis=0)
        self.lag_backward = lag_backward
        self.lag_forward = lag_forward

    def __len__(self):
        return self.X.shape[0] - self.lag_backward - self.lag_forward

    def __getitem__(self, i):
        X = self.X[i : i + self.lag_forward + self.lag_backward + 1].T
        assert (
            X.size
        ), f"Zero size for index {i} of {len(self)} with {self.X.shape=}"
        return X, self.Y[i + self.lag_backward - self.lag_forward]
