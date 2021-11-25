import numpy as np
from serde_numpy import parse_float_array, parse_int_array, parse_bool_array, parse_keys


data = """
    {"id": "161fjkjnbf", 
    "stream_1": 
    [[36, 60, 1610034058843.0, 1, 1, -519151558, 49, "", -1, false, false, false, true, 0, "Digit", "1", false, "-1", 20912.165, -1], 
    [36, 61, 1610034058847.0, 2, 1, -519151558, 49, "1", -1, false, false, true, true, 0, "Digit", "1", false, "-1", 20918.465, 0], 
    [36, 63, 1610034058888.0, 0, 1, -519151558, 49, "", -1, false, true, false, true, 0, "Digit", "1", false, "-1", 20986.94, -1]], 
    "stream_2": 
    [[36, 12, 1610034053955.0, 0, 1, 0, 1016, 746, 1016, 817, 0, 0, 1016, 746, 0, 0, 16049.98], 
    [36, 13, 1610034053972.0, 0, 1, 0, 1020, 717, 1020, 788, 0, 0, 1020, 717, 0, 0, 16069.06], 
    [36, 14, 1610034053989.0, 0, 1, 0, 1023, 688, 1023, 759, 0, 0, 1023, 688, 0, 0, 16083.675]]}"""


def test_float():
    out = parse_float_array(data, "stream_1", 2)
    assert out.dtype == np.float64
    assert np.array_equal(out, [1610034058843.0, 1610034058847.0, 1610034058888.0])
    assert isinstance(out, np.ndarray)


def test_int():
    out = parse_int_array(data, "stream_2", 1)
    assert out.dtype == np.int64
    assert np.array_equal(out, [12, 13, 14])
    assert isinstance(out, np.ndarray)


def test_bool():
    out = parse_bool_array(data, "stream_1", 11)
    assert out.dtype == np.bool
    assert np.array_equal(out, [False, True, False])
    assert isinstance(out, np.ndarray)

def test():
    out = parse_keys(data, ["stream_1", "stream_2"], [[0,1,2], [0,2,4]])
    expected = {
        'stream_2': [
            np.array([36, 36, 36]), 
            np.array([1610034053955.0, 1610034053972.0, 1610034053989.0]), 
            np.array([1, 1, 1])], 
        'stream_1': [
            np.array([36, 36, 36]), 
            np.array([60, 61, 63]), 
            np.array([1610034058843.0, 1610034058847.0, 1610034058888.0])]
            }
    for k in out:
        for i in range(len(out[k])):
            assert np.array_equal(out[k][i], expected[k][i])