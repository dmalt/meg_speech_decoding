import numpy as np
import pytest

from library.signal import Annotation, Signal
from library.signal_processing.filtering import ButterFiltFilt


@pytest.fixture(params=[np.float128, np.float64, np.float32, np.float16])
def signal(request) -> Signal:
    annots = [Annotation(0, 0.1, "test")]
    return Signal(np.random.rand(20000, 5).astype(request.param), sr=100, annotations=annots)


@pytest.mark.parametrize("l_freq, h_freq", [(1, None), (None, 5), (None, None), (10, 20)])
def test_ButterFiltFilt_preserves_signal_dtype_and_annotations(signal: Signal, l_freq, h_freq):
    filter = ButterFiltFilt(order=5, l_freq=l_freq, h_freq=h_freq)
    filtered = filter(signal)
    assert filtered.dtype == signal.dtype
    assert filtered.annotations == signal.annotations
