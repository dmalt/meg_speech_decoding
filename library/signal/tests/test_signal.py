import numpy as np
import pytest

from library.signal import Annotation, Signal
from library.signal.annotations import drop_bad_segments, split_into_good_segments


@pytest.fixture(params=[np.float128, np.float64, np.float32, np.float16])
def signal(request) -> Signal:
    annots = [Annotation(0, 0.1, "test")]
    return Signal(np.random.rand(20000, 5).astype(request.param), sr=100, annotations=annots)


def test_drop_bad_segments_preserves_signal_dtype(signal: Signal) -> None:
    signal_wo_bads = drop_bad_segments(signal)
    assert signal_wo_bads.dtype == signal.dtype


def test_drop_bad_segments_creates_proper_annotations(signal: Signal) -> None:
    signal.annotations = [
        Annotation(0.2, 2, "BAD"),
        Annotation(1.2, 2, "BAD"),
        Annotation(4, 1, "BAD"),
        Annotation(0.1, 5, "GOOD"),
        Annotation(6, 1, "GOOD"),
        Annotation(6.5, 0, "BAD_BOUNDARY")
    ]
    wo_bads = drop_bad_segments(signal)
    assert Annotation(0.2, 0, "BAD_BOUNDARY") in wo_bads.annotations
    assert Annotation(1, 0, "BAD_BOUNDARY") in wo_bads.annotations
    assert Annotation(0.1, 0.1, "GOOD") in wo_bads.annotations
    assert Annotation(0.2, 0.8, "GOOD") in wo_bads.annotations
    assert Annotation(1, 0.1, "GOOD") in wo_bads.annotations
    assert Annotation(2, 0.5, "GOOD") in wo_bads.annotations
    assert Annotation(2.5, 0.5, "GOOD") in wo_bads.annotations
    assert Annotation(2.5, 0, "BAD_BOUNDARY") in wo_bads.annotations


def test_split_into_good_segments(signal: Signal) -> None:
    signal.annotations = [
        Annotation(0.2, 2, "BAD"),
        Annotation(1.2, 2, "BAD"),
        Annotation(4, 1, "BAD"),
        Annotation(0.1, 5, "GOOD"),
        Annotation(6, 1, "GOOD"),
        Annotation(6.5, 0, "BAD_BOUNDARY")
    ]
    segments = split_into_good_segments(signal)
    sr = signal.sr
    assert len(segments[0]) / sr == signal.annotations[0].onset
    assert segments[0].annotations == [Annotation(0.1, 0.1, "GOOD")]

    assert len(segments[1]) / sr == 0.8
    assert segments[1].annotations == [Annotation(0, 0.8, "GOOD")]

    assert len(segments[2]) / sr == 1.5
    assert segments[2].annotations == [Annotation(0, 0.1, "GOOD"), Annotation(1, 0.5, "GOOD")]

    assert len(segments[3]) / sr == len(signal) / sr - 4 - 2.5
    assert segments[3].annotations == [Annotation(0, 0.5, "GOOD")]
