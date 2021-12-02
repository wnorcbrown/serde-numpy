import pytest

import numpy as np
from serde_numpy import deserialize


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


def test_nest(json_str: bytes):
    structure = {"str": str, "nest": {"is_nest": bool}}
    deserialized = deserialize(json_str, structure)
    assert_same_structure(structure, deserialized)
    assert_correct_types(structure, deserialized)
    assert deserialized["nest"]["is_nest"] == True


def test_nest_array(json_str: bytes):
    structure = {"str": str, "nest": {"is_nest": bool, "stream3": np.int32}}
    deserialized = deserialize(json_str, structure)
    print(type(deserialized), deserialized)
    assert_same_structure(structure, deserialized)
    assert_correct_types(structure, deserialized)
    assert np.array_equal(deserialized["nest"]["stream3"], [[0, 28], [0, 9]])


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
            "stream3":[[0,28],
                       [0,9]],
            "arr3":[true,false,true],
            "unused_key":"a"}}"""
    