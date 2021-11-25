import numpy as np
from serde_numpy import parse_float_array


def test_float():
    data = """
        {"sid": "161fjkjnbf", 
        "e_key_events": 
        [[36, 60, 1610034058843.0, 1, 1, -519151558, 49, "", -1, false, false, false, false, 0, "Digit", "1", false, "-1", 20912.165, -1], 
        [36, 61, 1610034058847.0, 2, 1, -519151558, 49, "1", -1, false, false, false, false, 0, "Digit", "1", false, "-1", 20918.465, 0], 
        [36, 63, 1610034058888.0, 0, 1, -519151558, 49, "", -1, false, false, false, false, 0, "Digit", "1", false, "-1", 20986.94, -1]], 
        "e_mouse_events": 
        [[36, 12, 1610034053955.0, 0, 1, 0, 1016, 746, 1016, 817, 0, 0, 1016, 746, 0, 0, 16049.98], 
        [36, 13, 1610034053972.0, 0, 1, 0, 1020, 717, 1020, 788, 0, 0, 1020, 717, 0, 0, 16069.06], 
        [36, 14, 1610034053989.0, 0, 1, 0, 1023, 688, 1023, 759, 0, 0, 1023, 688, 0, 0, 16083.675]]}"""
    out = parse_float_array(data, "e_key_events", 2)
    assert np.array_equal(out, [1610034058843.0, 1610034058847.0, 1610034058888.0])
    assert isinstance(out, np.ndarray)

