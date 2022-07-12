import pytest

from typing import Any
import numpy as np

from .fixtures import json_str, wonky_json_str
from .utils import deserialize, assert_correct_types, assert_same_structure


def test_parses(json_str: bytes):
    import orjson
    orjson.loads(json_str)


def test_deserialize_str(json_str: bytes):
    structure = {"str": str}
    deserialized = deserialize(json_str, structure)
    assert_same_structure(structure, deserialized)
    assert_correct_types(structure, deserialized)
    assert deserialized["str"] == "h"


def test_deserialize_bool(json_str: bytes):
    structure = {"bool": bool}
    deserialized = deserialize(json_str, structure)
    assert_same_structure(structure, deserialized)
    assert_correct_types(structure, deserialized)
    assert deserialized["bool"] == True


def test_deserialize_int(json_str: bytes):
    structure = {"int": int}
    deserialized = deserialize(json_str, structure)
    assert_same_structure(structure, deserialized)
    assert_correct_types(structure, deserialized)
    assert deserialized["int"] == 3


def test_deserialize_float(json_str: bytes):
    structure = {"float": float}
    deserialized = deserialize(json_str, structure)
    assert_same_structure(structure, deserialized)
    assert_correct_types(structure, deserialized)
    assert deserialized["float"] == 0.34


def test_deserialize_float_array(json_str: bytes):
    for dtype in [np.float32, np.float64]:
        structure = {"float_arr": dtype}
        deserialized = deserialize(json_str, structure)
        assert_same_structure(structure, deserialized)
        assert_correct_types(structure, deserialized)
        assert np.array_equal(deserialized["float_arr"], 
                        np.array([[1.254439975231648,-0.6893827594332794],[-0.2922560025562806,0.5204819306523419]], dtype))


def test_deserialize_int_array(json_str: bytes):
    for dtype in [np.int8, np.int16, np.int32, np.int64]:
        structure = {"int_arr": dtype}
        deserialized = deserialize(json_str, structure)
        assert_same_structure(structure, deserialized)
        assert_correct_types(structure, deserialized)
        assert np.array_equal(deserialized["int_arr"], 
                        np.array([[-100,-25],[-41,-62]], dtype))


def test_deserialize_uint_array(json_str: bytes):
    for dtype in [np.uint8, np.uint16, np.uint32, np.uint64]:
        structure = {"uint_arr": dtype}
        deserialized = deserialize(json_str, structure)
        assert_same_structure(structure, deserialized)
        assert_correct_types(structure, deserialized)
        assert np.array_equal(deserialized["uint_arr"], 
                        np.array([[100,25],[41,62]], dtype))


def test_deserialize_bool_array(json_str: bytes):
    structure = {"bool_arr": np.bool_}
    deserialized = deserialize(json_str, structure)
    assert_same_structure(structure, deserialized)
    assert_correct_types(structure, deserialized)
    assert np.array_equal(deserialized["bool_arr"], 
                    np.array([[False, True],[True, False]], np.bool_))


def test_nest(json_str: bytes):
    structure = {"str": str, "nest": {"is_nest": bool}}
    deserialized = deserialize(json_str, structure)
    assert_same_structure(structure, deserialized)
    assert_correct_types(structure, deserialized)
    assert deserialized["nest"]["is_nest"] == True


def test_nest_array(json_str: bytes):
    structure = {"str": str, "nest": {"is_nest": bool, "stream1": np.int32}}
    deserialized = deserialize(json_str, structure)
    print(type(deserialized), deserialized)
    assert_same_structure(structure, deserialized)
    assert_correct_types(structure, deserialized)
    assert np.array_equal(deserialized["nest"]["stream1"], [[0, 28], [0, 9]])


def test_list_of_array(json_str: bytes):
    structure = {"stream0": [np.float64, np.int16, np.uint8]}
    deserialized = deserialize(json_str, structure)
    print(type(deserialized), deserialized)
    assert_same_structure(structure, deserialized)
    assert_correct_types(structure, deserialized)
    assert np.array_equal(deserialized["stream0"], [[-1.720294114558863,0.5990469735869592,0.0506514091042812,0.7204746283872987,1.5351637640639662],
                                                    [72,45,-58,-16,-14],
                                                    [1,0,0,1,0]])

def test_transpose_sequence(json_str: bytes):
    structure = {"stream3": [[np.float64, np.uint8, np.uint8]],
                 "stream2": [[np.float32, np.bool_, str]],
                 "nest":{"stream0":[[np.float32, np.int8]]}}
    deserialized = deserialize(json_str, structure)
    print(type(deserialized), deserialized)
    assert_same_structure(structure, deserialized)
    assert_correct_types(structure, deserialized)


def test_transpose_map(json_str: bytes):
    structure = {"stream4": [{"x": np.float64, "y": np.uint8, "z": np.uint8}]}
    deserialized = deserialize(json_str, structure)
    print(type(deserialized), deserialized)
    assert_same_structure(structure, deserialized)
    assert_correct_types(structure, deserialized)


def test_entire_structure(json_str: bytes):
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
    deserialized = deserialize(json_str, structure)
    print(type(deserialized), deserialized)
    assert_same_structure(structure, deserialized)
    assert_correct_types(structure, deserialized)


def test_missing_column(json_str: bytes):
    structure = {"stream0": [np.float64, np.int16]}
    deserialized = deserialize(json_str, structure)
    print(type(deserialized), deserialized)
    assert_same_structure(structure, deserialized)
    assert_correct_types(structure, deserialized)


def test_missing_column_transpose(json_str: bytes):
    structure = {"stream3": [[np.float64, np.uint8]]}
    deserialized = deserialize(json_str, structure)
    print(type(deserialized), deserialized)
    assert_same_structure(structure, deserialized)
    assert_correct_types(structure, deserialized)


def test_missing_key(json_str: bytes):
    structure = {"float_arr": np.float32}
    deserialized = deserialize(json_str, structure)
    assert_same_structure(structure, deserialized)
    assert_correct_types(structure, deserialized)


def test_missing_key_tranpose(json_str: bytes):
    structure = {"stream4": [{"x": np.float64, "y": np.uint8}]}
    deserialized = deserialize(json_str, structure)
    print(type(deserialized), deserialized)
    assert_same_structure(structure, deserialized)
    assert_correct_types(structure, deserialized)





