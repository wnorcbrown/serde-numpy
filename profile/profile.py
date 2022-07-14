from typing import Callable, List, Mapping, Sequence, Type, Tuple, Any
import json
import time

import numpy as np
import orjson
import json
import string
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from serde_numpy import NumpyDeserializer

PLOT_COLS = 3
np.random.seed(33)


def make_array(shape: Tuple[int, ...], dtype: Type) -> List[Any]:
    if dtype in [int, np.int16, np.int32, np.int64]:
        return np.random.randint(-2**15, 2**15, size=shape).tolist()
    elif dtype in [np.uint16, np.uint32, np.uint64]:
        return np.random.randint(0, 2**16, size=shape).tolist()
    elif dtype in [float, np.float16, np.float32, np.float64]:
        return (np.random.randn(np.prod(shape)).reshape(shape) * 2**8).tolist()
    elif dtype in [bool, np.bool_]:
        return (np.random.rand(np.prod(shape)).reshape(shape) < 0.5).tolist()
    elif dtype == str:
        return np.random.choice(list(string.ascii_letters), np.prod(shape), replace=True).reshape(shape).tolist()
    else:
        raise ValueError(f"dtype: {dtype} not recognised.")


def make_data(shape: Tuple[int, ...], dtype: Type) -> bytes:
    return str.encode(json.dumps(dict(key=make_array(shape, dtype))))


def tranpose_lol(lol: List[List[Any]]) -> List[List[Any]]:
    return [list(row) for row in zip(*lol)]


def make_transposed_data(n_rows: int, dtypes: Sequence[Type]) -> bytes:
    data = [ make_array((n_rows,), dtype) for dtype in dtypes]
    data = tranpose_lol(data)
    return str.encode(json.dumps(dict(key=data)))


def orjson_then_numpy(json_str: bytes, dtype: Type) -> Mapping[str, np.ndarray]:
    out = orjson.loads(json_str)
    out["key"] = np.array(out["key"], dtype)
    return out


def orjson_then_numpy_tranpose(json_str: bytes, dtypes: Sequence[Type]) -> Mapping[str, np.ndarray]:
    out = orjson.loads(json_str)
    out["key"] = tranpose_lol(out["key"])
    out["key"] = [np.array(out["key"][i], dtype) for i, dtype in enumerate(dtypes)]
    return out


def serde_numpy(json_str: bytes, deserialize: Callable) -> Mapping[str, List[np.ndarray]]:
    return deserialize(json_str, )


def _get_dtype(name: str) -> type:
    try:
        return eval(name)
    except NameError:
        return getattr(np, name)


def get_times_2d_array(n_rows: Sequence[int], n_cols: Sequence[int], dtype: Type, n_iters: int = 10) -> Mapping[str, np.ndarray]:
    times_orjson = []
    times_serde_numpy = []
    n_rows_ = []
    n_cols_ = []
    deserializer = NumpyDeserializer.from_dict(dict(key=dtype))
    for rows in n_rows:
        for cols in n_cols:
            data = make_data((rows, cols), dtype)
            n_rows_.append(rows)
            n_cols_.append(cols)
            orjson_times_ = []
            serde_np_times_ = []
            for _ in range(n_iters):
                
                time0 = time.time()
                _ = orjson_then_numpy(data, dtype)
                orjson_times_.append(time.time() - time0)

                time0 = time.time()
                _ = deserializer.deserialize_json(data)
                serde_np_times_.append(time.time() - time0)
                
            times_orjson.append(np.median(orjson_times_))
            times_serde_numpy.append(np.median(serde_np_times_))

    return dict(n_rows=np.array(n_rows_), 
                n_cols=np.array(n_cols_), 
                orjson_plus_numpy_time=np.array(times_orjson), 
                serde_numpy_time=np.array(times_serde_numpy))


def get_times_transposed_arrays(n_rows: Sequence[int], dtypes: Sequence[Type], n_iters: int = 10) -> Mapping[str, np.ndarray]:
    times_orjson = []
    times_serde_numpy = []
    n_rows_ = []
    deserializer = NumpyDeserializer.from_dict(dict(key=[dtypes]))
    for rows in n_rows:
        data = make_transposed_data(rows, dtypes)
        n_rows_.append(rows)
        orjson_times_ = []
        serde_np_times_ = []
        for _ in range(n_iters):
            
            time0 = time.time()
            _ = orjson_then_numpy_tranpose(data, dtypes)
            orjson_times_.append(time.time() - time0)

            time0 = time.time()
            _ = deserializer.deserialize_json(data)
            serde_np_times_.append(time.time() - time0)
            
        times_orjson.append(np.median(orjson_times_))
        times_serde_numpy.append(np.median(serde_np_times_))

    return dict(n_rows=np.array(n_rows_),
                orjson_plus_numpy_time=np.array(times_orjson), 
                serde_numpy_time=np.array(times_serde_numpy))


def plot_times_2d_array(times_rows: Mapping[str, np.ndarray], times_cols: Mapping[str, np.ndarray], ax: plt.Axes, title: str):
    ax.plot(times_rows["n_rows"], times_rows["orjson_plus_numpy_time"] / times_rows["serde_numpy_time"], alpha=0.5)
    ax.plot(times_cols["n_cols"], times_cols["orjson_plus_numpy_time"] / times_cols["serde_numpy_time"], alpha=0.5)
    ax.grid()
    ax.set_xscale("log")
    ax.legend(["n_rows", "n_cols"])
    ax.set_title(title)
    ax.set_ylabel("speed up")


def run_profile_2d_array(dtypes: Sequence[type]):
    print("Profiling 2D array deserialization...")
    n_rows = (2**np.arange(0, 20, 2)).astype(int)
    n_cols = (2**np.arange(0, 20, 2)).astype(int)
    plot_rows = int(np.ceil(len(dtypes) / PLOT_COLS))
    fig = plt.figure(figsize=(6*PLOT_COLS, 4*plot_rows))
    gs = gridspec.GridSpec(plot_rows, PLOT_COLS)
    for i, dtype in enumerate(dtypes):
        times_rows = get_times_2d_array(n_rows, [10], dtype)
        times_cols = get_times_2d_array([10], n_cols, dtype)
        ax = fig.add_subplot(gs[i // PLOT_COLS, i % PLOT_COLS])
        plot_times_2d_array(times_rows, times_cols, ax, dtype.__name__)
    plt.savefig("profile/2darr_profile.png", pad_inches = 0) 


def run_profile_transposed_arrays(dtypes: Sequence[type]):
    print("Profiling tranposed arrays deserialization...")
    n_rows = (2**np.arange(0, 22, 2)).astype(int)
    times = get_times_transposed_arrays(n_rows, dtypes)
    fig = plt.figure(figsize=(6, 4))
    plt.plot(times["n_rows"], times["orjson_plus_numpy_time"] / times["serde_numpy_time"], alpha=0.5)
    plt.grid()
    plt.xscale("log")
    plt.legend(["n_rows"])
    plt.title([dtype.__name__ for dtype in dtypes], fontsize=10)
    plt.ylabel("speed up")
    plt.savefig("profile/transpose_profile.png", pad_inches = 0)

        
if __name__ == "__main__":
    import sys
    dtypes = [_get_dtype(s) for s in sys.argv[1:]] or [np.int16, np.int32, np.int64, np.uint16, np.uint32, np.uint64, np.float32, np.float64, np.bool_]
    run_profile_2d_array(dtypes)
    run_profile_transposed_arrays(dtypes)
    print("Done.")



