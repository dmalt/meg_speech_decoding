from __future__ import annotations

from typing import List, NamedTuple, Sequence

from mne.io import BaseRaw  # type: ignore


class Annotation(NamedTuple):
    onset: float
    duration: float
    type: str


Annotations = List[Annotation]


def from_raw(raw: BaseRaw) -> Annotations:
    if not hasattr(raw, "annotations"):
        return []
    onsets: list[float] = list(raw.annotations.onset)
    durations: list[float] = list(raw.annotations.duration)
    types: list[str] = list(raw.annotations.description)
    onsets = [o - raw.first_samp / raw.info["sfreq"] for o in raw.annotations.onset]
    return [Annotation(o, d, t) for o, d, t, in zip(onsets, durations, types)]


def get_good_slices(annotations: Sequence[Annotation], sr: float) -> list[slice]:
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
