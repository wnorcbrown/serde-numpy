import pytest
from typing import Any, Callable, Tuple, Type

import numpy as np

from .fixtures import json_str, msgpack_bytes, wonky_json_str, wonky_msgpack_bytes
from .utils import deserialize_json, deserialize_msgpack, assert_correct_types, assert_same_structure

from serde_numpy import NumpyDeserializer


@pytest.mark.parametrize("name,type_", [("str", bool), ("bool", float), ("int", str), ("int", float)])
@pytest.mark.parametrize("bytes_func", [(json_str, deserialize_json), (msgpack_bytes, deserialize_msgpack)])
def test_deserialize_single_fail(bytes_func: Tuple[bytes, Callable], name: str, type_: Type):
    input_bytes, deserialize_func = bytes_func
    structure = {name: type_}
    with pytest.raises(TypeError):
        structure = {"str": bool}
        deserialize_func(input_bytes, structure)


# second structure is for transpose dtypes
@pytest.mark.parametrize("structure", [{"stream0": [np.float64, np.int64, np.int8, bool]}, {"stream3": [[np.float64, np.int32, int, str]]}])
@pytest.mark.parametrize("bytes_func", [(json_str, deserialize_json), (msgpack_bytes, deserialize_msgpack)])
def test_extra_column(bytes_func: Tuple[bytes, Callable], structure: Any):
    input_bytes, deserialize_func = bytes_func
    with pytest.raises(TypeError) as e:
        deserialize_func(input_bytes, structure)
    assert str(e.value).startswith("Too many columns specified:")


@pytest.mark.parametrize("structure", [{"extra_key": np.float32, "stream0": [np.float64, np.int64]}, {"stream4": [{"x": np.float64, "y": np.uint8, "extra_key_transpose": np.bool_}]}])
@pytest.mark.parametrize("bytes_func", [(json_str, deserialize_json), (msgpack_bytes, deserialize_msgpack)])
def test_extra_key(bytes_func: Tuple[bytes, Callable], structure: Any):
    input_bytes, deserialize_func = bytes_func
    with pytest.raises(TypeError) as e:
        deserialize_func(input_bytes, structure)
    assert str(e.value).startswith(r'Key(s) not found: ["extra_key')


@pytest.mark.parametrize("bytes_func", [(wonky_json_str, deserialize_json), (wonky_msgpack_bytes, deserialize_msgpack)])
def test_irregular_array(bytes_func: Tuple[bytes, Callable]):
    input_bytes, deserialize_func = bytes_func
    with pytest.raises(ValueError) as e:
        structure = {
            "irregular": np.float32,
            }
        deserialize_func(input_bytes, structure)
    assert str(e.value).startswith("Irregular shape found cannot parse as f32 array. Expected shape: [2, 2]  Total elements: 3")


def test_list_of_nested_structure():
    structure = {"stream4": [{"x": np.float64, "y": np.uint8, "x": {"a": np.bool_, "b": np.int32}}]}
    with pytest.raises(TypeError) as e:
        NumpyDeserializer.from_dict(structure)
    assert str(e.value).startswith("""structure unsupported. Currently sequences of nested structures are unsupported e.g. [{\"a\": {\"b\": Type}}])""")


@pytest.mark.parametrize("bytes_func", [(json_str, deserialize_json), (msgpack_bytes, deserialize_msgpack)])
def test_deserialize_list_as_map(bytes_func: Tuple[bytes, Callable]):
    input_bytes, deserialize_func = bytes_func
    structure = {"stream0": {"a": np.float64, "b": np.int64, "c": np.int8, "d": bool}}
    with pytest.raises(TypeError) as e:
        deserialize_func(input_bytes, structure)
    assert str(e.value).startswith("Cannot deserialize sequence as map of arrays")


@pytest.mark.parametrize("bytes_func", [(json_str, deserialize_json), (msgpack_bytes, deserialize_msgpack)])
def test_deserialize_list_as_type(bytes_func: Tuple[bytes, Callable]):
    input_bytes, deserialize_func = bytes_func
    structure = {"stream0": int}
    with pytest.raises(TypeError) as e:
        deserialize_func(input_bytes, structure)
    assert str(e.value).startswith("Could not deserialize as int")


@pytest.mark.parametrize("bytes_func", [(json_str, deserialize_json), (msgpack_bytes, deserialize_msgpack)])
def test_deserialize_map_as_list(bytes_func: Tuple[bytes, Callable]):
    input_bytes, deserialize_func = bytes_func
    structure = [str, int, bool]
    with pytest.raises(TypeError) as e:
        deserialize_func(input_bytes, structure)
    assert str(e.value).startswith("Cannot deserialize map as sequence of arrays")


@pytest.mark.parametrize("bytes_func", [(json_str, deserialize_json), (msgpack_bytes, deserialize_msgpack)])
def test_deserialize_map_as_type(bytes_func: Tuple[bytes, Callable]):
    input_bytes, deserialize_func = bytes_func
    structure = int
    with pytest.raises(TypeError) as e:
        deserialize_func(input_bytes, structure)
    assert str(e.value).startswith("Cannot deserialize map as type: int. Try using a dictionary instead")


@pytest.mark.parametrize("bytes_func", [(json_str, deserialize_json), (msgpack_bytes, deserialize_msgpack)])
def test_deserialize_lol_as_lom(bytes_func: Tuple[bytes, Callable]):
    input_bytes, deserialize_func = bytes_func
    with pytest.raises(TypeError) as e:
        structure = {"stream3": [{"a": np.float64, "b": np.uint8, "c": np.uint8}]}
        deserialized = deserialize_func(input_bytes, structure)
    # currently not maintaining order of input dictionaries
    assert str(e.value).startswith("""invalid type: sequence, expected map with elements: """)


@pytest.mark.parametrize("bytes_func", [(json_str, deserialize_json), (msgpack_bytes, deserialize_msgpack)])
def test_deserialize_lom_as_lol(bytes_func: Tuple[bytes, Callable]):
    input_bytes, deserialize_func = bytes_func
    with pytest.raises(TypeError) as e:
        structure = {"stream4": [[np.float64, np.uint8, np.uint8]]}
        deserialized = deserialize_func(input_bytes, structure)
    assert str(e.value).startswith("invalid type: map, expected sequence with elements: [np.float64, np.uint8, np.uint8, ]")
