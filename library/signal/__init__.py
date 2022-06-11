from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Generic, List, NamedTuple, TypeVar, overload

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
    Timeseries stored as numpy array together with sampling rate and
    annotations

    Parameters
    ----------
    data : ndarray of shape (n_samples, n_channels)
        Array with signal data
    sr : float
        Sampling rate
    annotations: Annotations
        Signal annotations

    """

    data: SignalArray[T]
    sr: float
    annotations: Annotations

    def __post_init__(self) -> None:
        assert self.data.ndim == 2, f"2-d array is required; got data of shape {self.data.shape}"

    def __len__(self) -> int:
        """Signal length in samples"""
        return self.n_samples

    @overload
    def __array__(self) -> SignalArray[T]:
        ...

    @overload
    def __array__(self, dtype: npt.DTypeLike) -> Any:
        ...

    def __array__(self, dtype: npt.DTypeLike | None = None) -> SignalArray[T] | npt.NDArray[Any]:
        if dtype is not None:
            return self.data.astype(dtype)
        return self.data

    def __str__(self) -> str:
        return (
            f"signal of shape {self.n_samples} samples X {self.n_channels} channel(s) "
            + f" sampled at {self.sr:.2f} Hz;"
            + f" duration={round(self.duration, 2)} sec;"
            + f" {len(self.annotations)} annotated segments"
        )

    def update(self, data: SignalArray[T]) -> Signal[T]:
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


class Signal1D(Signal[T], Generic[T]):
    def __post_init__(self) -> None:
        super().__post_init__()
        assert self.n_channels == 1
