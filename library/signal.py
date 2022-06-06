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

    data: npt.NDArray["np.floating[T]"]
    sr: float
    annotations: Annotations = field(default_factory=list)

    def __post_init__(self):
        assert self.data.ndim == 2, f"2-d array is required; got data of shape {self.data.shape}"

    def __len__(self) -> int:
        """Signal length in samples"""
        return self.n_samples

    def __array__(self, dtype=None) -> npt.NDArray["np.floating[T]"]:
        if dtype is not None:
            return self.data.astype(dtype)
        return self.data

    def update(self, data: npt.NDArray["np.floating[T]"]) -> Signal[T]:
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


def drop_bad_segments(signal: Signal[T]) -> Signal[T]:
    good_slices = get_good_slices(signal.annotations, signal.sr)
    good_data: npt.NDArray[np.floating[T]] = np.concatenate([signal.data[s] for s in good_slices])
    continuous_annotations: list[Annotation] = []
    prev_time = 0.0
    for s in good_slices:
        onset = prev_time
        stop_samp = len(signal) if s.stop is None else s.stop
        duration = (stop_samp - s.start) / signal.sr
        continuous_annotations.append(Annotation(onset, duration, type="CONTINUOUS"))
        prev_time = duration

    return Signal(good_data, signal.sr, continuous_annotations)


def get_good_slices(annotations: Annotations, sr: float) -> list[slice]:
    """Get slices for periods with types not starting with 'BAD'"""
    events = []
    # when start and end occur simoultaneosly, first add new bad segment, then remove old
    START = 0
    END = 1
    for onset, duration, _ in filter(lambda a: a.type.startswith("BAD"), annotations):
        events.append((int(onset * sr), START))
        events.append((int((onset + duration) * sr), END))
    events.sort()
    bad_overlap_cnt = 0

    res = []
    last_start = 0
    for ev in events:
        if ev[1] == START:
            if bad_overlap_cnt == 0:
                res.append(slice(last_start, ev[0]))
            bad_overlap_cnt += 1
        elif ev[1] == END:
            bad_overlap_cnt -= 1
            if bad_overlap_cnt == 0:
                last_start = ev[0]
    # process last event
    if bad_overlap_cnt == 0:
        res.append(slice(last_start, None))
    return res
