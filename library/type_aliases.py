from typing import Any, Callable, Optional, Tuple

import numpy as np
import numpy.typing as npt

Array = npt.NDArray[Any]
Array32 = npt.NDArray[np.float32]

Signal = npt.NDArray[Any]  # array of shape (n_samples, n_sensors)
Signal32 = npt.NDArray[np.float32]  # array of shape (n_samples, n_sensors)
Signal32_T = npt.NDArray[np.float32]  # array of shape (n_sensors, n_samples)

# TODO: check batch shape. Maybe sensors and samples should be swapped
BatchSignal = npt.NDArray[Any]  # array of shape (batch_size, n_samples, n_sensors)
BatchSignal32 = npt.NDArray[np.float32]  # array of shape (batch_size, n_samples, n_sensors)
BatchSignalMask = npt.NDArray[np.bool_]  # array of shape (n_samples, n_sensors)

Samples32 = npt.NDArray[np.float32]  # array of shape (n_samples,)
Sensors32 = npt.NDArray[np.float32]  # array of shape (n_sensors,)

VoiceDetector = Callable[[BatchSignal], Optional[BatchSignalMask]]
Transformer = Callable[[Signal, float], Tuple[Signal32, float]]
