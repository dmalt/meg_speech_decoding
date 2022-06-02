from library.io.signal_annotations import Annotation, get_good_slices


def test_get_good_slices():
    chunks = get_good_slices([Annotation(10, 5, "BAD"), Annotation(11, 5, "BAD")], sr=1)
    assert chunks == [slice(0, 10), slice(16, None)]
