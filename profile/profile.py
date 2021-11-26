from typing import List, Mapping, Sequence, Type
import json
import time

import numpy as np
import orjson

from serde_numpy import parse_keys

np.random.seed(33)


def make_data(shape = (3, 100), dtypes: Sequence[Type] = (int, float, bool)) -> bytes:
    out = {}
    for i, dtype in enumerate(dtypes):
        if dtype == int:
            out[f"key_{i}"] = np.random.randint(-2**15, 2**15, size=shape).tolist()
        elif dtype == float:
            out[f"key_{i}"] = (np.random.randn(shape[0]*shape[1]).reshape(shape) * 2**8).tolist()
        elif dtype == bool:
            out[f"key_{i}"] = (np.random.rand(shape[0]*shape[1]).reshape(shape) < 0.5).tolist()
    return str.encode(json.dumps(out))


def orjson_numpy_loads(json_str: bytes, n_keys: int) -> Mapping[str, np.ndarray]:
    out = orjson.loads(json_str)
    for i in range(n_keys):
        data = out[f"key_{i}"]
        data = list(map(list, zip(*data)))
        out[f"key_{i}"] = np.array(data)
    return out


def serde_numpy_loads(json_str: bytes, n_keys: int, n_cols: int, dtypes: Sequence[Type]) -> Mapping[str, List[np.ndarray]]:
    keys = [f"key_{i}" for i in range(n_keys)]
    indexes = [list(range(n_cols)) for _ in keys]
    dtypes = [[dtype.__name__]*n_cols for dtype in dtypes]
    return parse_keys(json_str, keys, indexes, dtypes)


def run_profile(n_rows: int, n_cols: int, dtypes: Sequence[Type] = (int, float, bool), name: str = "", n_iters: int = 100):
    data = make_data((n_rows, n_cols), dtypes)
    
    times_orjson = []
    for _ in range(n_iters):
        time0 = time.time()
        _ = orjson_numpy_loads(data, len(dtypes))
        times_orjson.append(time.time() - time0)
    
    times_serde_numpy = []
    for _ in range(n_iters):
        time0 = time.time()
        _ = serde_numpy_loads(data, len(dtypes), n_cols, dtypes)
        times_serde_numpy.append(time.time() - time0)
    
    print("-"*75)
    print(f"{name} data times for orjson:: Mean: {np.mean(times_orjson):.3} Std: {np.std(times_orjson):.3}")
    print(f"{name} data times for serde numpy:: Mean: {np.mean(times_serde_numpy):.3} Std: {np.std(times_serde_numpy):.3}")
    print("-"*75)
    
if __name__ == "__main__":
    run_profile(2*1, 4, name="tiny", n_iters=1000)
    run_profile(2**5, 4, name="small", n_iters=100)
    run_profile(2**10, 4, name="medium", n_iters=10)
    run_profile(2**15, 4, name="large", n_iters=5)
    run_profile(2**20, 4, name="huge", n_iters=2)
