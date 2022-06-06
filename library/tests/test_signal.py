import numpy as np
import pytest

from library.signal import Annotation, Signal, drop_bad_segments, get_good_slices


def test_get_good_slices():
    chunks = get_good_slices([Annotation(10, 5, "BAD"), Annotation(11, 5, "BAD")], sr=1)
    assert chunks == [slice(0, 10), slice(16, None)]


@pytest.fixture(params=[np.float128, np.float64, np.float32, np.float16])
def signal(request) -> Signal:
    annots = [Annotation(0, 0.1, "test")]
    return Signal(np.random.rand(20000, 5).astype(request.param), sr=100, annotations=annots)


def test_drop_bad_segments_preserves_signal_dtype(signal: Signal) -> None:
    signal_wo_bads = drop_bad_segments(signal)
    assert signal_wo_bads.dtype == signal.dtype


def test_drop_bad_segments_creates_proper_annotations(signal: Signal) -> None:
    signal.annotations = [Annotation(0.2, 2, "BAD")]
    wo_bads_annots = drop_bad_segments(signal).annotations
    assert len(wo_bads_annots) == 2
    assert all(annot.type == "CONTINUOUS" for annot in wo_bads_annots)
    assert wo_bads_annots[0] == Annotation(0, 0.2, "CONTINUOUS")
    assert wo_bads_annots[1] == Annotation(0.2, len(signal) / signal.sr - 2.2, "CONTINUOUS")
