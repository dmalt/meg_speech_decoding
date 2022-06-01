from library.io.speech_meg import _get_good_slices


def test_get_good_slices():
    chunks = _get_good_slices(1, [(10, 5), (11, 5)])
    assert chunks == [slice(0, 10), slice(16, None)]
