from typing import Callable, List, Mapping, Sequence, Type, Tuple
import json
import time

import numpy as np
import orjson
import json
import string

from serde_numpy import deserialize, deserialize_new

np.random.seed(33)


def make_data(shape: Tuple[int, ...], dtypes: Sequence[Type]) -> bytes:
    out = {}
    for i, dtype in enumerate(dtypes):
        if dtype in [int, np.int16, np.int32, np.int64]:
            out[f"key_{i}"] = np.random.randint(-2**15, 2**15, size=shape).tolist()
        elif dtype in [float, np.float16, np.float32, np.float64]:
            out[f"key_{i}"] = (np.random.randn(np.prod(shape)).reshape(shape) * 2**8).tolist()
        elif dtype in [bool, np.bool_]:
            out[f"key_{i}"] = (np.random.rand(np.prod(shape)).reshape(shape) < 0.5).tolist()
        elif dtype == str:
            out[f"key_{i}"] = np.random.choice(list(string.ascii_letters), np.prod(shape), replace=True).reshape(shape).tolist()
    return str.encode(json.dumps(out))


def json_numpy_loads(json_str: bytes, keys: Sequence, dtypes: Sequence[Type], load_str_func: Callable) -> Mapping[str, np.ndarray]:
    out = load_str_func(json_str)
    for i, key in enumerate(keys):
        out[key] = np.array(out[key], dtypes[i])
    return out

def serde_numpy_loads(json_str: bytes, keys: Sequence[str], dtypes: Sequence[Type]) -> Mapping[str, List[np.ndarray]]:
    return deserialize(json_str, dict(zip(keys, dtypes)))


def serde_numpy_new_loads(json_str: bytes, keys: Sequence[str], dtypes: Sequence[Type]) -> Mapping[str, List[np.ndarray]]:
    return deserialize_new(json_str, dict(zip(keys, dtypes)))


def _get_dtype(name: str) -> type:
    try:
        return eval(name)
    except NameError:
        return getattr(np, name)


def run_profile(n_rows: int, n_cols: int, dtypes: Sequence[Type] = (np.int32, np.float32), name: str = "", n_iters: int = 100):
    data = make_data((n_rows, n_cols), dtypes)
    keys = [f"key_{i}" for i in range(len(dtypes))]

    times_json = []
    for _ in range(n_iters):
        time0 = time.time()
        _ = json_numpy_loads(data, keys, dtypes, json.loads)
        times_json.append(time.time() - time0)
    
    times_orjson = []
    for _ in range(n_iters):
        time0 = time.time()
        _ = json_numpy_loads(data, keys, dtypes, orjson.loads)
        times_orjson.append(time.time() - time0)
    
    times_serde_numpy = []
    for _ in range(n_iters):
        time0 = time.time()
        _ = serde_numpy_new_loads(data, keys, dtypes)
        times_serde_numpy.append(time.time() - time0)
    
    print("-"*75)
    print(f"{name} data times for json:: Mean: {np.mean(times_json):.3} Std: {np.std(times_json):.3}")
    print(f"{name} data times for orjson:: Mean: {np.mean(times_orjson):.3} Std: {np.std(times_orjson):.3}")
    print(f"{name} data times for serde numpy:: Mean: {np.mean(times_serde_numpy):.3} Std: {np.std(times_serde_numpy):.3}")
    print(f"Speed up serde_numpy vs. orjson: {np.mean(times_orjson)/np.mean(times_serde_numpy):.3}x")
    print(f"Speed up serde_numpy vs. json: {np.mean(times_json)/np.mean(times_serde_numpy):.3}x")
    print("-"*75)
    
if __name__ == "__main__":
    import sys
    dtypes = sys.argv[1:]
    dtypes = [_get_dtype(name) for name in dtypes]
    if len(dtypes) == 0:
        raise ValueError("No dtypes specified!")
    print(f"For dtypes: {dtypes}")
    run_profile(2*1, 2, name="tiny", n_iters=1000, dtypes=dtypes)
    run_profile(2**5, 2, name="small", n_iters=100, dtypes=dtypes)
    run_profile(2**10, 4, name="medium", n_iters=10, dtypes=dtypes)
    run_profile(2**15, 4, name="large", n_iters=5, dtypes=dtypes)
    run_profile(2**20, 6, name="huge", n_iters=2, dtypes=dtypes)
