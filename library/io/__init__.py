from __future__ import annotations

from dataclasses import dataclass

from ..transformers import TargetTransformer
from ..type_aliases import Transformer


@dataclass
class Lags:
    backward: int
    forward: int


@dataclass
class Transformers:
    transform_x: Transformer
    transform_y: TargetTransformer


@dataclass
class Annotations:
    onsets: list[float]
    durations: list[float]
    types: list[str]

    def __post_init__(self):
        msg1 = f"Inconsistent sizes: {len(self.onsets)=} != {len(self.durations)=}"
        msg2 = f"Inconsistent sizes: {len(self.types)=} != {len(self.durations)}"
        assert len(self.onsets) == len(self.durations), msg1
        assert len(self.types) == len(self.durations), msg2

    def __getitem__(self, i: int) -> tuple[float, float, str]:
        return self.onsets[i], self.durations[i], self.types[i]

    def __len__(self) -> int:
        return len(self.onsets)

    def get_good_slices(self, sr: float) -> list[slice]:
        """Get slices for periods not annotated as bads"""
        events = []
        # when start and end occur simoultaneosly, first add new bad segment, then remove old
        START = 0
        END = 1
        # mypy doesnt believe __getitem__ makes it iterable
        for onset, duration, type in self:  # type: ignore
            if type.startswith("BAD"):
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
