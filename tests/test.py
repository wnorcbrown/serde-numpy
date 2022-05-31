import pytest

from typing import Any
import numpy as np
from serde_numpy import NumpyDeserializer


def deserialize(json_str: bytes, structure: dict):
    deserializer = NumpyDeserializer.from_dict(structure)
    return deserializer.deserialize_json(json_str)


def assert_same_structure(dict_1: dict, dict_2: dict):
    assert set(dict_1.keys()) == set(dict_2.keys()), f"Dict 1 keys: {dict.keys()}  Dict 2 keys: {dict_2.keys()}"
    for k, v in dict_1.items():
        if type(v) is dict:
            assert_same_structure(v, dict_2[k])


def _get_type(object_: Any) -> type:
    if isinstance(object_, np.ndarray):
        return object_.dtype
    elif isinstance(object_, np.dtype) or isinstance(object_, type):
        return object_
    else:
        return type(object_)


def assert_correct_types(structure: dict, deserialized: dict):
    for k, v in structure.items():
        type_s = _get_type(v)
        type_d = _get_type(deserialized[k]) 
        if type_s is dict:
            assert type_s == type_d
            assert_correct_types(v, deserialized[k])
        elif type_s is list:

            # [[Type, ...]] for parsing list of lists stored row-wise
            if _get_type(v[0]) == list:
                for s, d in zip(v[0], deserialized[k]):
                    if type(d) != list:
                        assert _get_type(d) == _get_type(s)
                    else:
                        assert all(_get_type(d[i]) == _get_type(s) for i in range(len(d)))

            # [{key: Type, ...}] for parsing list of dict stored row-wise
            elif _get_type(v[0]) == dict:
                assert_correct_types(v[0], deserialized[k])

            # [Type, ...] for parsing list of lists stored row-wise
            else:
                for s, d in zip(v, deserialized[k]):
                    if type(d) != list:
                        assert _get_type(d) == _get_type(s)
                    else:
                        assert all(_get_type(d[i]) == _get_type(s) for i in range(len(d)))
            

def test_parses(json_str: bytes):
    import orjson
    orjson.loads(json_str)


def test_deserialize_str(json_str: bytes):
    structure = {"str": str}
    deserialized = deserialize(json_str, structure)
    assert_same_structure(structure, deserialized)
    assert_correct_types(structure, deserialized)
    assert deserialized["str"] == "h"


def test_deserialize_bool_fail(json_str: bytes):
    with pytest.raises(TypeError):
        structure = {"str": bool}
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


def test_deserialize_float_fail(json_str: bytes):
    with pytest.raises(TypeError):
        structure = {"bool": float}
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


def test_deserialize_str_fail(json_str: bytes):
    with pytest.raises(TypeError):
        structure = {"int": str}
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


# def test_deserialize_int_fail(json_str: bytes):
#     with pytest.raises(TypeError):
#         structure = {"float": int}
#         deserialized = deserialize(json_str, structure)
#         assert_same_structure(structure, deserialized)
#         assert_correct_types(structure, deserialized)
#         assert deserialized["float"] == 0.34
# DO WE NEED TO RAISE THIS ERROR? we could allow it to deseriliaze to float silently...


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
    with pytest.raises(TypeError):
        structure = {"str": str, "nest": {"is_nest": bool, "stream0": np.int32}}
        deserialized = deserialize(json_str, structure)
        print(type(deserialized), deserialized)
        assert_same_structure(structure, deserialized)
        assert_correct_types(structure, deserialized)
        assert np.array_equal(deserialized["nest"]["stream0"], [[0, 28], [0, 9]])


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

def test_column_parsing(json_str: bytes):
    structure = {"stream3": [[np.float64, np.uint8, np.uint8]],
                 "stream2": [[np.float32, np.bool, str]],
                 "nest":{"stream0":[[np.float32, np.int8]]}}
    deserialized = deserialize(json_str, structure)
    print(type(deserialized), deserialized)
    assert_same_structure(structure, deserialized)
    assert_correct_types(structure, deserialized)


def test_column_parsing(json_str: bytes):
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


@pytest.fixture
def json_str() -> bytes:
    return b"""{
        "str":"h",
        "int":3,
        "bool":true,
        "float":0.34,
        "float_arr":[[1.254439975231648,-0.6893827594332794],[-0.2922560025562806,0.5204819306523419]],
        "int_arr":[[-100,-25],[-41,-62]],
        "uint_arr":[[100, 25],[41, 62]],
        "bool_arr":[[false, true],[true, false]],
        "stream0":[[-1.720294114558863,0.5990469735869592,0.0506514091042812,0.7204746283872987,1.5351637640639662],
                   [72,45,-58,-16,-14],
                   [1,0,0,1,0]],
        "stream1":[[-1.720294114558863,0.5990469735869592,0.0506514091042812,0.7204746283872987,1.5351637640639662],
                   [72,45,-58,-16,-14],
                   [true,false,false,true,false]],
        "stream2":[[-2.1727126743596266,false,"a"],
                   [-0.06389102863189458,true,"b"],
                   [1.3716941547285826,true,"c"]],
        "stream3":[[-2.1727126743596266,0,1],
                   [-0.06389102863189458,1,2],
                   [1.3716941547285826,1,3]],
        "stream4":[{"x":-2.1727126743596266,"y":0,"z":1},
                   {"x":-0.06389102863189458,"y":1,"z":2},
                   {"x":1.3716941547285826,"y":1,"z":3}],
        "nest":{
            "is_nest":true,
            "nestiness":0.9999,
            "stream0":[[-0.11954010897451912,28],
                       [0.21599243355210992,9]],
            "stream1":[[0, 28],
                       [0, 9]],
            "arr3":[true,false,true],
            "unused_key":"a"}}"""
    