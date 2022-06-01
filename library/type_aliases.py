from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import numpy.typing as npt

Array = npt.NDArray[Any]
Array32 = npt.NDArray[np.float32]

SignalArray = npt.NDArray[Any]  # array of shape (n_samples, n_sensors)
SignalArray32 = npt.NDArray[np.float32]  # array of shape (n_samples, n_sensors)

BatchSignalArray = npt.NDArray[Any]  # array of shape (batch_size, n_samples, n_sensors)
BatchSignalArray32 = npt.NDArray[np.float32]  # array of shape (batch_size, n_samples, n_sensors)
BatchSignalArrayMask = npt.NDArray[np.bool_]  # array of shape (n_samples, n_sensors)

VoiceDetector = Callable[[BatchSignalArray], Optional[BatchSignalArrayMask]]

Transformer = Callable[[SignalArray, float], Tuple[SignalArray32, float]]

Info = Dict[str, Any]
