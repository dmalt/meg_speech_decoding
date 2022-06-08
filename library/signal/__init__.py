from __future__ import annotations

from dataclasses import dataclass, field
from typing import Generic, List, NamedTuple, TypeVar

import numpy as np
import numpy.typing as npt

T = TypeVar("T", bound=npt.NBitBase)


class Annotation(NamedTuple):
    onset: float
    duration: float
    type: str


Annotations = List[Annotation]

SignalArray = npt.NDArray[np.floating[T]]  # array of shape (n_samples, n_sensors)


@dataclass
class Signal(Generic[T]):
    """
    Timeseries stored as numpy array with sampling rate

    Parameters
    ----------
    data : ndarray of shape (n_samples, n_channels)
        Signal samples
    sr : float
        Sampling rate
    annotations: Annotations

    """

    data: SignalArray[T]
    sr: float
    annotations: Annotations = field(default_factory=list)

    def __post_init__(self):
        assert self.data.ndim == 2, f"2-d array is required; got data of shape {self.data.shape}"

    def __len__(self) -> int:
        """Signal length in samples"""
        return self.n_samples

    def __array__(self, dtype=None) -> npt.NDArray[np.floating[T]]:
        if dtype is not None:
            return self.data.astype(dtype)
        return self.data

    def __str__(self):
        return (
            f"signal of shape={self.data.shape} sampled at {self.sr} Hz; duration={self.duration}"
        )

    def update(self, data: npt.NDArray[np.floating[T]]) -> Signal[T]:
        assert len(data) == len(self), f"Can only update with equal length data; got {len(data)=}"
        return self.__class__(data, self.sr, self.annotations)

    @property
    def n_samples(self) -> int:
        return self.data.shape[0]

    @property
    def n_channels(self) -> int:
        return self.data.shape[1]

    @property
    def dtype(self) -> np.dtype:
        return self.data.dtype

    @property
    def duration(self) -> float:
        """Signal duration in seconds"""
        return self.n_samples / self.sr
