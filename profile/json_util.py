from typing import Any, List, Mapping, Sequence, Tuple, Type
import string

import numpy as np
import orjson
import json


def make_array(shape: Tuple[int, ...], dtype: Type) -> List[Any]:
    if dtype in [int, np.int16, np.int32, np.int64]:
        return np.random.randint(-(2**15), 2**15, size=shape).tolist()
    elif dtype in [np.uint16, np.uint32, np.uint64]:
        return np.random.randint(0, 2**16, size=shape).tolist()
    elif dtype in [float, np.float16, np.float32, np.float64]:
        return (np.random.randn(np.prod(shape)).reshape(shape) * 2**8).tolist()
    elif dtype in [bool, np.bool_]:
        return (np.random.rand(np.prod(shape)).reshape(shape) < 0.5).tolist()
    elif dtype == str:
        return (
            np.random.choice(
                list(string.ascii_letters),
                np.prod(shape),
                replace=True
                )
            .reshape(shape)
            .tolist()
        )
    else:
        raise ValueError(f"dtype: {dtype} not recognised.")


def make_data(shape: Tuple[int, ...], dtype: Type) -> bytes:
    return str.encode(json.dumps(dict(key=make_array(shape, dtype))))


def tranpose_lol(lol: List[List[Any]]) -> List[List[Any]]:
    return [list(row) for row in zip(*lol)]


def make_transposed_data(n_rows: int, dtypes: Sequence[Type]) -> bytes:
    data = [make_array((n_rows,), dtype) for dtype in dtypes]
    data = tranpose_lol(data)
    return str.encode(json.dumps(dict(key=data)))


def orjson_then_numpy(
    json_str: bytes, dtype: Type
) -> Mapping[str, np.ndarray]:
    out = orjson.loads(json_str)
    out["key"] = np.array(out["key"], dtype)
    return out


def orjson_then_numpy_tranpose(
    json_str: bytes, dtypes: Sequence[Type]
) -> Mapping[str, np.ndarray]:
    out = orjson.loads(json_str)
    out["key"] = tranpose_lol(out["key"])
    out["key"] = [
        np.array(out["key"][i], dtype)
        for i, dtype in enumerate(dtypes)
        ]
    return out
