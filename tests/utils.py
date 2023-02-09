from typing import Any
import numpy as np

from serde_numpy import NumpyDeserializer


def deserialize_json(json_str: bytes, structure: dict):
    deserializer = NumpyDeserializer.from_dict(structure)
    return deserializer.deserialize_json(json_str)


def deserialize_msgpack(msgpack_bytes: bytes, structure: dict):
    deserializer = NumpyDeserializer.from_dict(structure)
    return deserializer.deserialize_msgpack(msgpack_bytes)


def assert_same_structure(dict_1: dict, dict_2: dict):
    assert set(dict_1.keys()) == set(dict_2.keys()), f"Dict 1 keys: {dict_1.keys()}  Dict 2 keys: {dict_2.keys()}"
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
            