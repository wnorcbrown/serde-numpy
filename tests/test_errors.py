import pytest

import numpy as np

from .fixtures import json_str, wonky_json_str
from .utils import deserialize, assert_correct_types, assert_same_structure

from serde_numpy import NumpyDeserializer


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


def test_list_of_nested_structure():
    structure = {"stream4": [{"x": np.float64, "y": np.uint8, "x": {"a": np.bool_, "b": np.int32}}]}
    with pytest.raises(NotImplementedError) as e:
        NumpyDeserializer.from_dict(structure)
    assert str(e.value).startswith("""List of nested structures (i.e. [{"a": {"b": Type}}]) currently not implemented.""")


def test_deserialize_list_as_map(json_str: str):
    structure = {"stream0": {"a": np.float64, "b": np.int64, "c": np.int8, "d": bool}}
    with pytest.raises(TypeError) as e:
        deserialize(json_str, structure)
    assert str(e.value).startswith("Cannot deserialize sequence as map of arrays")


def test_deserialize_list_as_type(json_str: str):
    structure = {"stream0": int}
    with pytest.raises(TypeError) as e:
        deserialize(json_str, structure)
    assert str(e.value).startswith("Cannot deserialize sequence as map of arrays")


def test_deserialize_map_as_list(json_str: str):
    structure = [str, int, bool]
    with pytest.raises(TypeError) as e:
        deserialize(json_str, structure)
    assert str(e.value).startswith("Cannot deserialize map as sequence of arrays")


def test_deserialize_map_as_type(json_str: str):
    structure = int
    with pytest.raises(TypeError) as e:
        deserialize(json_str, structure)
    assert str(e.value).startswith("Cannot deserialize map as sequence of arrays")


def test_deserialize_lol_as_lom(json_str: bytes):
    structure = {"stream3": [{"a": np.float64, "b": np.uint8, "c": np.uint8}]}
    deserialized = deserialize(json_str, structure)

def test_deserialize_lom_as_lol(json_str: str):
    structure = {"stream4": [[np.float64, np.uint8, np.uint8]]}
    deserialized = deserialize(json_str, structure)
