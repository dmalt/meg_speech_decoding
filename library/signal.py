from __future__ import annotations

from dataclasses import dataclass, field
from typing import Generic, List, NamedTuple, Optional, TypeVar

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

    data: npt.NDArray[np.floating[T]]
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


def drop_bad_segments(signal: Signal[T]) -> Signal[T]:
    normalized_annots = AnnotationsOverlapProcessor(signal.annotations).normalize_bad_segments()
    # good_slices = _get_good_slices(signal.annotations)
    removed_dur = 0.0
    new_annots: Annotations = []
    good_segments_mask = np.ones(len(signal), dtype=bool)

    for a in normalized_annots:
        if not a.type.startswith("BAD"):
            new_annots.append(Annotation(a.onset - removed_dur, a.duration, a.type))
        else:
            segment_start_samp = int(a.onset * signal.sr)
            segment_end_samp = int((a.onset + a.duration) * signal.sr)
            good_segments_mask[segment_start_samp:segment_end_samp] = False
            new_annots.append(Annotation(a.onset - removed_dur, duration=0, type="BAD_BOUNDARY"))
            removed_dur += a.duration

    return Signal(signal.data[good_segments_mask], signal.sr, new_annots)


class AnnotationsOverlapProcessor:
    # order important: when segment start and end occur simoultaneosly, first
    # add new bad segment, then remove old
    START = 0
    END = 1

    def __init__(self, annotations: Annotations):
        self.events = []
        for a in annotations:
            self.events.append((a.onset, self.START, a))
            self.events.append((a.onset + a.duration, self.END, a))
        self.events.sort()
        self.bad_overlap_cnt = 0
        self.res: Annotations = []
        self.open_segments: dict[Annotation, float] = {}
        self.bad_start: Optional[float] = None
        self._process_all()

    def normalize_bad_segments(self) -> Annotations:
        return self.res

    def _process_all(self) -> None:
        for ev_onset, ev_type, ev_annotation in self.events:
            self._process_event(ev_onset, ev_type, ev_annotation)
        assert not self.open_segments

    def _process_event(self, ev_onset: float, ev_type, ev_annotation: Annotation) -> None:
        is_bad = ev_annotation.type.startswith("BAD")
        if ev_type == self.START and is_bad:
            if self.bad_overlap_cnt == 0:
                self._finalize_all_open_segments(end=ev_onset)
                self.bad_start = ev_onset
            self.bad_overlap_cnt += 1
        elif ev_type == self.START and not is_bad:
            self._start_tracking_good_segment(ev_annotation)
        elif ev_type == self.END and is_bad:
            self.bad_overlap_cnt -= 1
            if self.bad_overlap_cnt == 0:
                self._change_tracked_segments_onset(new_onset=ev_onset)
                assert self.bad_start is not None
                self.res.append(Annotation(self.bad_start, ev_onset - self.bad_start, "BAD"))
        else:  # end and good
            self._stop_tracking_good_segment(ev_onset, ev_annotation)

    def _start_tracking_good_segment(self, annotation):
        self.open_segments[annotation] = annotation.onset

    def _change_tracked_segments_onset(self, new_onset):
        self.open_segments.update((a, new_onset) for a in self.open_segments)

    def _finalize_all_open_segments(self, end):
        self.res.extend(Annotation(o, end - o, a.type) for a, o in self.open_segments.items())

    def _stop_tracking_good_segment(self, ev_onset, annotation):
        annot_onset = self.open_segments.pop(annotation)
        if self.bad_overlap_cnt == 0:
            self._finalize_segment(ev_onset, annot_onset, annotation)

    def _finalize_segment(self, ev_onset, annot_onset, annot):
        self.res.append(Annotation(annot_onset, ev_onset - annot_onset, annot.type))
