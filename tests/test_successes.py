from itertools import product
import pytest
from typing import Any, Callable, Tuple, Type

import numpy as np

from .fixtures import json_str, msgpack_bytes, wonky_json_str
from .utils import deserialize_json, deserialize_msgpack, assert_correct_types, assert_same_structure


def test_parses():
    import orjson
    orjson.loads(json_str)


@pytest.mark.parametrize("name,type_,value", [("str", str, "h"), ("bool", bool, True), ("int", int, 3), ("float", float, 0.34)])
@pytest.mark.parametrize("bytes_func", [(json_str, deserialize_json), (msgpack_bytes, deserialize_msgpack)])
def test_deserialize_single(bytes_func: Tuple[bytes, Callable], name: str, type_: Type, value: Any):
    input_bytes, deserialize_func = bytes_func
    structure = {name: type_}
    deserialized = deserialize_func(input_bytes, structure)
    assert_same_structure(structure, deserialized)
    assert_correct_types(structure, deserialized)
    assert deserialized[name] == value


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("bytes_func", [(json_str, deserialize_json), (msgpack_bytes, deserialize_msgpack)])
def test_deserialize_float_array(bytes_func: Tuple[bytes, Callable], dtype: Type):
    input_bytes, deserialize_func = bytes_func
    structure = {"float_arr": dtype}
    deserialized = deserialize_func(input_bytes, structure)
    assert_same_structure(structure, deserialized)
    assert_correct_types(structure, deserialized)
    assert np.array_equal(deserialized["float_arr"], 
                    np.array([[1.254439975231648,-0.6893827594332794],[-0.2922560025562806,0.5204819306523419]], dtype))


@pytest.mark.parametrize("dtype", [np.int8, np.int16, np.int32, np.int64])
@pytest.mark.parametrize("bytes_func", [(json_str, deserialize_json), (msgpack_bytes, deserialize_msgpack)])
def test_deserialize_int_array(bytes_func: Tuple[bytes, Callable], dtype: Type):
    input_bytes, deserialize_func = bytes_func
    structure = {"int_arr": dtype}
    deserialized = deserialize_func(input_bytes, structure)
    assert_same_structure(structure, deserialized)
    assert_correct_types(structure, deserialized)
    assert np.array_equal(deserialized["int_arr"], 
                        np.array([[-100,-25],[-41,-62]], dtype))


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.uint32, np.uint64])
@pytest.mark.parametrize("bytes_func", [(json_str, deserialize_json), (msgpack_bytes, deserialize_msgpack)])
def test_deserialize_uint_array(bytes_func: Tuple[bytes, Callable], dtype: Type):
    input_bytes, deserialize_func = bytes_func
    structure = {"uint_arr": dtype}
    deserialized = deserialize_func(input_bytes, structure)
    assert_same_structure(structure, deserialized)
    assert_correct_types(structure, deserialized)
    assert np.array_equal(deserialized["uint_arr"], 
                        np.array([[100,25],[41,62]], dtype))


@pytest.mark.parametrize("bytes_func", [(json_str, deserialize_json), (msgpack_bytes, deserialize_msgpack)])
def test_deserialize_bool_array(bytes_func: Tuple[bytes, Callable]):
    input_bytes, deserialize_func = bytes_func
    structure = {"bool_arr": np.bool_}
    deserialized = deserialize_func(input_bytes, structure)
    assert_same_structure(structure, deserialized)
    assert_correct_types(structure, deserialized)
    assert np.array_equal(deserialized["bool_arr"], 
                    np.array([[False, True],[True, False]], np.bool_))


@pytest.mark.parametrize("bytes_func", [(json_str, deserialize_json), (msgpack_bytes, deserialize_msgpack)])
def test_nest(bytes_func: Tuple[bytes, Callable]):
    input_bytes, deserialize_func = bytes_func
    structure = {"str": str, "nest": {"is_nest": bool}}
    deserialized = deserialize_func(input_bytes, structure)
    assert_same_structure(structure, deserialized)
    assert_correct_types(structure, deserialized)
    assert deserialized["nest"]["is_nest"] == True


@pytest.mark.parametrize("bytes_func", [(json_str, deserialize_json), (msgpack_bytes, deserialize_msgpack)])
def test_nest_array(bytes_func: Tuple[bytes, Callable]):
    input_bytes, deserialize_func = bytes_func
    structure = {"str": str, "nest": {"is_nest": bool, "stream1": np.int32}}
    deserialized = deserialize_func(input_bytes, structure)
    assert_same_structure(structure, deserialized)
    assert_correct_types(structure, deserialized)
    assert np.array_equal(deserialized["nest"]["stream1"], [[0, 28], [0, 9]])


@pytest.mark.parametrize("bytes_func", [(json_str, deserialize_json), (msgpack_bytes, deserialize_msgpack)])
def test_list_of_array(bytes_func: Tuple[bytes, Callable]):
    input_bytes, deserialize_func = bytes_func
    structure = {"stream0": [np.float64, np.int16, np.uint8]}
    deserialized = deserialize_func(input_bytes, structure)
    print(type(deserialized), deserialized)
    assert_same_structure(structure, deserialized)
    assert_correct_types(structure, deserialized)
    assert np.array_equal(deserialized["stream0"], [[-1.720294114558863,0.5990469735869592,0.0506514091042812,0.7204746283872987,1.5351637640639662],
                                                    [72,45,-58,-16,-14],
                                                    [1,0,0,1,0]])


@pytest.mark.parametrize("bytes_func", [(json_str, deserialize_json), (msgpack_bytes, deserialize_msgpack)])
def test_transpose_sequence(bytes_func: Tuple[bytes, Callable]):
    input_bytes, deserialize_func = bytes_func
    structure = {"stream3": [[np.float64, np.uint8, np.uint8]],
                 "stream2": [[np.float32, np.bool_, str]],
                 "nest":{"stream0":[[np.float32, np.int8]]}}
    deserialized = deserialize_func(input_bytes, structure)
    assert_same_structure(structure, deserialized)
    assert_correct_types(structure, deserialized)


@pytest.mark.parametrize("bytes_func", [(json_str, deserialize_json), (msgpack_bytes, deserialize_msgpack)])
def test_transpose_map(bytes_func: Tuple[bytes, Callable]):
    input_bytes, deserialize_func = bytes_func
    structure = {"stream4": [{"x": np.float64, "y": np.uint8, "z": np.uint8}]}
    deserialized = deserialize_func(input_bytes, structure)
    assert_same_structure(structure, deserialized)
    assert_correct_types(structure, deserialized)


@pytest.mark.parametrize("bytes_func", [(json_str, deserialize_json), (msgpack_bytes, deserialize_msgpack)])
def test_entire_structure(bytes_func: Tuple[bytes, Callable]):
    input_bytes, deserialize_func = bytes_func
    structure = {
        "str": str,
        "int": int,
        "bool": bool,
        "float": float,
        "float_arr": np.float32,
        "int_arr": np.int16,
        "uint_arr": np.uint32,
        "bool_arr": np.bool_,
        "nest": {"is_nest": bool,
                 "nestiness": float,
                 "stream0": [[np.float32, np.int32]], 
                 "stream1": [[np.int32, np.uint8]]},
        "stream0": [np.float64, np.int64, np.int8],
        "stream1": [float, np.int64, bool],
        "stream2": [[np.float64, bool, str]],
        "stream3": [[np.float64, np.int32, int]],
        "stream4": [{"x": np.float64, "y": np.uint8, "z": np.uint8}],
        }
    deserialized = deserialize_func(input_bytes, structure)
    assert_same_structure(structure, deserialized)
    assert_correct_types(structure, deserialized)


@pytest.mark.parametrize("bytes_func", [(json_str, deserialize_json), (msgpack_bytes, deserialize_msgpack)])
def test_missing_column(bytes_func: Tuple[bytes, Callable]):
    input_bytes, deserialize_func = bytes_func
    structure = {"stream0": [np.float64, np.int16]}
    deserialized = deserialize_func(input_bytes, structure)
    assert_same_structure(structure, deserialized)
    assert_correct_types(structure, deserialized)


@pytest.mark.parametrize("bytes_func", [(json_str, deserialize_json), (msgpack_bytes, deserialize_msgpack)])
def test_missing_column_transpose(bytes_func: Tuple[bytes, Callable]):
    input_bytes, deserialize_func = bytes_func
    structure = {"stream3": [[np.float64, np.uint8]]}
    deserialized = deserialize_func(input_bytes, structure)
    assert_same_structure(structure, deserialized)
    assert_correct_types(structure, deserialized)


@pytest.mark.parametrize("bytes_func", [(json_str, deserialize_json), (msgpack_bytes, deserialize_msgpack)])
def test_missing_key(bytes_func: Tuple[bytes, Callable]):
    input_bytes, deserialize_func = bytes_func
    structure = {"float_arr": np.float32}
    deserialized = deserialize_func(input_bytes, structure)
    assert_same_structure(structure, deserialized)
    assert_correct_types(structure, deserialized)


@pytest.mark.parametrize("bytes_func", [(json_str, deserialize_json), (msgpack_bytes, deserialize_msgpack)])
def test_missing_key_tranpose(bytes_func: Tuple[bytes, Callable]):
    input_bytes, deserialize_func = bytes_func
    structure = {"stream4": [{"x": np.float64, "y": np.uint8}]}
    deserialized = deserialize_func(input_bytes, structure)
    assert_same_structure(structure, deserialized)
    assert_correct_types(structure, deserialized)





