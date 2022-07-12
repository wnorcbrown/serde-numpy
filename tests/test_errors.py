import pytest

import numpy as np

from .fixtures import json_str, wonky_json_str
from .utils import deserialize, assert_correct_types, assert_same_structure


def test_deserialize_bool_fail(json_str: bytes):
    with pytest.raises(TypeError):
        structure = {"str": bool}
        deserialize(json_str, structure)


def test_deserialize_float_fail(json_str: bytes):
    with pytest.raises(TypeError):
        structure = {"bool": float}
        deserialize(json_str, structure)


def test_deserialize_str_fail(json_str: bytes):
    with pytest.raises(TypeError):
        structure = {"int": str}
        deserialize(json_str, structure)


def test_extra_column(json_str: bytes):
    with pytest.raises(TypeError) as e:
        structure = {"stream0": [np.float64, np.int64, np.int8, bool]}
        deserialize(json_str, structure)
    assert str(e.value).startswith("Too many columns specified: [np.float64, np.int64, np.int8, bool, ] (4) \nFound: (3)")


def test_extra_column_transpose(json_str: bytes):
    with pytest.raises(TypeError) as e:
        structure = {"stream3": [[np.float64, np.int32, int, str]]}
        deserialize(json_str, structure)
    assert str(e.value).startswith("Too many columns specified: [np.float64, np.int32, int, str, ] (4) \nFound: (3)")


def test_extra_key(json_str: bytes):
    with pytest.raises(TypeError) as e:
        structure = {"extra_key": np.float32, "stream0": [np.float64, np.int64]}
        deserialize(json_str, structure)
    assert str(e.value).startswith(r'Key(s) not found: ["extra_key"]')


def test_extra_key_transpose(json_str: bytes):
    with pytest.raises(TypeError) as e:
        structure = {"stream4": [{"x": np.float64, "y": np.uint8, "extra_key": np.bool_}]}
        deserialize(json_str, structure)
    assert str(e.value).startswith(r'Key(s) not found: ["extra_key"]')


def test_irregular_array(wonky_json_str: bytes):
    with pytest.raises(ValueError) as e:
        structure = {
            "irregular": np.float32,
            }
        deserialize(wonky_json_str, structure)
    assert str(e.value).startswith("Irregular shape found cannot parse as f32 array. Expected shape: [2, 2]  Total elements: 3")



