import numpy as np
import numpy.typing as npt
import torch

SignalArray32_T = npt.NDArray[np.float32]  # array of shape (n_sensors, n_samples)
SignalArray32 = npt.NDArray[np.float32]  # array of shape (n_samples, n_sensors)

SigBatchTensor = torch.Tensor  # tensor of shape (batch_size, n_sensors, n_samples)
ChanBatchTensor = torch.Tensor

# array of shape (batch_size, n_sensors, n_samples)
SignalBatch = npt.NDArray[np.floating[npt.NBitBase]]

ChanBatch = npt.NDArray[np.floating[npt.NBitBase]]  # array of shape (batch_size, n_sensors)

SamplesVector32 = npt.NDArray[np.float32]  # array of shape (n_samples,)
ChannelsVector32 = npt.NDArray[np.float32]  # array of shape (n_sensors,)
