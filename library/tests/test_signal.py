import numpy as np
import pytest

from library.signal import Annotation, Signal, drop_bad_segments

# def test_get_good_slices():
#     chunks = _get_good_slices([Annotation(10, 5, "BAD"), Annotation(11, 5, "BAD")], sr=1)
#     assert chunks == [slice(0, 10), slice(16, None)]


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
    ]
    wo_bads = drop_bad_segments(signal)
    wo_bads.annotations = [
        Annotation(round(a.onset, 4), round(a.duration, 4), a.type) for a in wo_bads.annotations
    ]
    # assert len(wo_bads.annotations) == 2
    assert Annotation(0.2, 0, "BAD_BOUNDARY") in wo_bads.annotations
    assert Annotation(1, 0, "BAD_BOUNDARY") in wo_bads.annotations
    assert Annotation(0.1, 0.1, "GOOD") in wo_bads.annotations
    assert Annotation(0.2, 0.8, "GOOD") in wo_bads.annotations
    assert Annotation(1, 0.1, "GOOD") in wo_bads.annotations
    assert len(wo_bads) == len(signal) - 4 * signal.sr
    # assert wo_bads_annots[0] == Annotation(0, 0.2, "CONTINUOUS")
    # assert wo_bads_annots[1] == Annotation(0.2, len(signal) / signal.sr - 2.2, "CONTINUOUS")
