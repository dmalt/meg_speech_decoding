from typing import Any

import numpy as np
import numpy.typing as npt

SignalArray32_T = npt.NDArray[np.float32]  # array of shape (n_sensors, n_samples)

# TODO: check batch shape. Maybe sensors and samples should be swapped
BatchSignal = npt.NDArray[Any]  # array of shape (batch_size, n_samples, n_sensors)
BatchSignal32 = npt.NDArray[np.float32]  # array of shape (batch_size, n_samples, n_sensors)
BatchSignalMask = npt.NDArray[np.bool_]  # array of shape (n_samples, n_sensors)

SamplesVector32 = npt.NDArray[np.float32]  # array of shape (n_samples,)
ChannelsVector32 = npt.NDArray[np.float32]  # array of shape (n_sensors,)
