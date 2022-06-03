from dataclasses import dataclass
from typing import Any, Callable, Optional, Tuple

import numpy as np
import numpy.typing as npt

Array = npt.NDArray[Any]
Array32 = npt.NDArray[np.float32]

SignalArray = npt.NDArray[Any]  # array of shape (n_samples, n_sensors)
SignalArray32 = npt.NDArray[np.float32]  # array of shape (n_samples, n_sensors)
SignalArray32_T = npt.NDArray[np.float32]  # array of shape (n_sensors, n_samples)

# TODO: check batch shape. Maybe sensors and samples should be swapped
BatchSignal = npt.NDArray[Any]  # array of shape (batch_size, n_samples, n_sensors)
BatchSignal32 = npt.NDArray[np.float32]  # array of shape (batch_size, n_samples, n_sensors)
BatchSignalMask = npt.NDArray[np.bool_]  # array of shape (n_samples, n_sensors)

SamplesVector32 = npt.NDArray[np.float32]  # array of shape (n_samples,)
Sensors32 = npt.NDArray[np.float32]  # array of shape (n_sensors,)

VoiceDetector = Callable[[BatchSignal], Optional[BatchSignalMask]]


@dataclass
class Signal:
    """
    Timeseries stored as numpy array with sampling rate

    Parameters
    ----------
    data : SignalArray
        Timeseries samples
    sr : float
        Sampling rate

    """

    data: SignalArray
    sr: float

    def __len__(self) -> int:
        """Signal length in samples"""
        return len(self.data)

    def __str__(self) -> str:
        return (
            f"signal of {len(self) / self.sr} seconds duration"
            + f"with {len(self)} data points sampled at {self.sr} Hz,"
        )


@dataclass
class Signal32(Signal):
    """
    Timeseries stored as numpy array of 32-bit floats with sampling rate

    Parameters
    ----------
    data : SignalArray32
        Timeseries samples
    sr : float
        Sampling rate

    """
    data: SignalArray32


def as_32(signal: Signal) -> Signal32:
    """Convert Signal to Signal32"""
    return Signal32(signal.data.astype("float32"), signal.sr)


Transformer = Callable[[Signal, float], Tuple[Signal32, float]]
