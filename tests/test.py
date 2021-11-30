import pytest

import numpy as np
from serde_numpy import parse_float_array, parse_int_array, parse_bool_array, parse_keys, deserialize


data = b"""
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
    assert out.dtype == bool
    assert np.array_equal(out, [False, True, False])
    assert isinstance(out, np.ndarray)


def test_parse_keys():
    out = parse_keys(data, ["stream_1", "stream_2"], [[0,1,2], [0,2,4]], [["int", "int", "float"], ["int", "float", "int"]])
    expected = {
        'stream_1': [
            np.array([36, 36, 36]), 
            np.array([60, 61, 63]), 
            np.array([1610034058843.0, 1610034058847.0, 1610034058888.0])],
        'stream_2': [
            np.array([36, 36, 36]), 
            np.array([1610034053955.0, 1610034053972.0, 1610034053989.0]), 
            np.array([1, 1, 1])], 
            }
    for k in out:
        for i in range(len(out[k])):
            assert np.array_equal(out[k][i], expected[k][i])


@pytest.fixture
def json_str() -> bytes:
    return b"""{
        "str":"h",
        "int":3,
        "bool":true,
        "float":0.34,
        "float_arr":[[1.254439975231648,-0.6893827594332794],[-0.2922560025562806,0.5204819306523419]],
        "int_arr":[[-100,-25],[-41,-62]],
        "stream1":[[-1.720294114558863,0.5990469735869592,0.0506514091042812,0.7204746283872987,1.5351637640639662],
                   [72,45,-58,-16,-14],[true,false,false,true,false]],
        "stream2":[[-2.1727126743596266,false,"a"],
                   [-0.06389102863189458,true,"b"],
                   [1.3716941547285826,true,"c"]],
        "nest":{
            "is_nest":true,
            "nestiness":0.9999,
            "stream3":[[-0.11954010897451912,28],
                       [0.21599243355210992,9]],
            "arr3":[true,false,true],
            "unused_key":"a"}}"""


def assert_same_structure(dict_1: dict, dict_2: dict):
    assert set(dict_1.keys()) == set(dict_2.keys()), f"Dict 1 keys: {dict.keys()}  Dict 2 keys: {dict_2.keys()}"
    for k, v in dict_1.items():
        if type(v) is dict:
            assert_same_structure(v, dict_2[k])


def assert_correct_types(structure: dict, deserialized: dict):
    for k, v in structure.items():
        if type(v) is not dict:
            if not isinstance(deserialized[k], np.ndarray):
                assert v == type(deserialized[k]), f"Deserialized type: {v}  Actual type: {type(deserialized[k])}"
            else:
                assert v == deserialized[k].dtype


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
    structure = {"float_arr": np.float32}
    deserialized = deserialize(json_str, structure)
    assert_same_structure(structure, deserialized)
    assert_correct_types(structure, deserialized)
    assert np.array_equal(deserialized["float_arr"], 
                    np.array([[1.254439975231648,-0.6893827594332794],[-0.2922560025562806,0.5204819306523419]], np.float32))


def test_deserialize_float_array2(json_str: bytes):
    structure = {"float_arr": np.float64}
    deserialized = deserialize(json_str, structure)
    assert_same_structure(structure, deserialized)
    assert_correct_types(structure, deserialized)
    assert np.array_equal(deserialized["float_arr"], 
                    np.array([[1.254439975231648,-0.6893827594332794],[-0.2922560025562806,0.5204819306523419]], np.float64))


def test_deserialize_int_array(json_str: bytes):
    structure = {"int_arr": np.int32}
    deserialized = deserialize(json_str, structure)
    assert_same_structure(structure, deserialized)
    assert_correct_types(structure, deserialized)
    assert np.array_equal(deserialized["int_arr"], 
                    np.array([[-100,-25],[-41,-62]], np.int32))


def test_deserialize_int_array(json_str: bytes):
    structure = {"int_arr": np.int64}
    deserialized = deserialize(json_str, structure)
    assert_same_structure(structure, deserialized)
    assert_correct_types(structure, deserialized)
    assert np.array_equal(deserialized["int_arr"], 
                    np.array([[-100,-25],[-41,-62]], np.int64))
    