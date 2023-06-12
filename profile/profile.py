from typing import Mapping, Sequence, Type
import time

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from serde_numpy import NumpyDeserializer, decode_jpeg, decode_png
from img_util import (
    get_random_img,
    get_jpeg_bytes,
    get_png_bytes,
    decode_pillow
)
from json_util import (
    make_data,
    make_transposed_data,
    orjson_then_numpy,
    orjson_then_numpy_tranpose,
)

PLOT_COLS = 3
np.random.seed(33)


def _get_dtype(name: str) -> type:
    try:
        return eval(name)
    except NameError:
        return getattr(np, name)


def get_times_2d_array(
    n_rows: Sequence[int], n_cols: Sequence[int], dtype: Type, n_iters: int = 10
) -> Mapping[str, np.ndarray]:
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

    return dict(
        n_rows=np.array(n_rows_),
        n_cols=np.array(n_cols_),
        orjson_plus_numpy_time=np.array(times_orjson),
        serde_numpy_time=np.array(times_serde_numpy),
    )


def get_times_transposed_arrays(
    n_rows: Sequence[int], dtypes: Sequence[Type], n_iters: int = 10
) -> Mapping[str, np.ndarray]:
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

    return dict(
        n_rows=np.array(n_rows_),
        orjson_plus_numpy_time=np.array(times_orjson),
        serde_numpy_time=np.array(times_serde_numpy),
    )


def get_times_jpeg() -> Mapping[str, np.ndarray]:
    sizes = [2**i for i in range(2, 14)]
    times_pillow = []
    times_serde_numpy = []
    for size in sizes:
        img = get_random_img(size)
        jpeg_bytes = get_jpeg_bytes(img)
        time0 = time.time()
        _ = decode_pillow(jpeg_bytes)
        times_pillow.append(time.time() - time0)

        time0 = time.time()
        _ = decode_jpeg(
            jpeg_bytes
        )
        times_serde_numpy.append(time.time() - time0)
    return dict(
        sizes=np.array(sizes),
        pillow_time=np.array(times_pillow),
        serde_numpy_time=np.array(times_serde_numpy),
    )


def get_times_png() -> Mapping[str, np.ndarray]:
    sizes = [2**i for i in range(2, 14)]
    times_pillow = []
    times_serde_numpy = []
    for size in sizes:
        img = get_random_img(size)
        png_bytes = get_png_bytes(img)
        time0 = time.time()
        _ = decode_pillow(png_bytes)
        times_pillow.append(time.time() - time0)

        time0 = time.time()
        _ = decode_png(
            png_bytes
        )
        times_serde_numpy.append(time.time() - time0)
    return dict(
        sizes=np.array(sizes),
        pillow_time=np.array(times_pillow),
        serde_numpy_time=np.array(times_serde_numpy),
    )


def plot_times_2d_array(
    times_rows: Mapping[str, np.ndarray],
    times_cols: Mapping[str, np.ndarray],
    ax: plt.Axes,
    title: str,
):
    ax.plot(
        times_rows["n_rows"],
        times_rows["orjson_plus_numpy_time"] / times_rows["serde_numpy_time"],
        alpha=0.5,
    )
    ax.plot(
        times_cols["n_cols"],
        times_cols["orjson_plus_numpy_time"] / times_cols["serde_numpy_time"],
        alpha=0.5,
    )
    ax.grid()
    ax.set_xscale("log")
    ax.legend(["n_rows", "n_cols"])
    ax.set_title(title)
    ax.set_ylabel("speed up")


def run_profile_2d_array(dtypes: Sequence[type]):
    print("Profiling 2D array deserialization...")
    n_rows = (2 ** np.arange(0, 20, 2)).astype(int)
    n_cols = (2 ** np.arange(0, 20, 2)).astype(int)
    plot_rows = int(np.ceil(len(dtypes) / PLOT_COLS))
    fig = plt.figure(figsize=(6 * PLOT_COLS, 4 * plot_rows))
    gs = gridspec.GridSpec(plot_rows, PLOT_COLS)
    for i, dtype in enumerate(dtypes):
        times_rows = get_times_2d_array(n_rows, [10], dtype)
        times_cols = get_times_2d_array([10], n_cols, dtype)
        ax = fig.add_subplot(gs[i // PLOT_COLS, i % PLOT_COLS])
        plot_times_2d_array(times_rows, times_cols, ax, dtype.__name__)
    plt.savefig("./2darr_profile.png", pad_inches=0, bbox_inches='tight')


def run_profile_transposed_arrays(dtypes: Sequence[type]):
    print("Profiling tranposed arrays deserialization...")
    n_rows = (2 ** np.arange(0, 22, 2)).astype(int)
    times = get_times_transposed_arrays(n_rows, dtypes)
    fig = plt.figure(figsize=(6, 4))
    plt.plot(
        times["n_rows"],
        times["orjson_plus_numpy_time"] / times["serde_numpy_time"],
        alpha=0.5,
    )
    plt.grid()
    plt.xscale("log")
    plt.legend(["n_rows"])
    plt.title([dtype.__name__ for dtype in dtypes], fontsize=10)
    plt.ylabel("speed up")
    plt.savefig("./transpose_profile.png", pad_inches=0, bbox_inches='tight')


def run_profile_jpeg():
    print("Profiling JPEG decoding...")
    times = get_times_jpeg()
    fig = plt.figure(figsize=(6, 4))
    plt.plot(
        times["sizes"],
        times["pillow_time"] / times["serde_numpy_time"],
        alpha=0.5,
    )
    plt.grid()
    plt.xscale("log")
    plt.xlabel(["image size (n x n)"])
    plt.title("JPEG decoding", fontsize=10)
    plt.ylabel("speed up")
    plt.savefig("./jpeg_profile.png", pad_inches=0, bbox_inches='tight')


def run_profile_png():
    print("Profiling PNG decoding...")
    times = get_times_png()
    fig = plt.figure(figsize=(6, 4))
    plt.plot(
        times["sizes"],
        times["pillow_time"] / times["serde_numpy_time"],
        alpha=0.5,
    )
    plt.grid()
    plt.xscale("log")
    plt.xlabel(["image size (n x n)"])
    plt.title("PNG decoding", fontsize=10)
    plt.ylabel("speed up")
    plt.savefig("./png_profile.png", pad_inches=0, bbox_inches='tight')


if __name__ == "__main__":
    import sys

    dtypes = [_get_dtype(s) for s in sys.argv[1:]] or [
        np.int16,
        np.int32,
        np.int64,
        np.uint16,
        np.uint32,
        np.uint64,
        np.float32,
        np.float64,
        np.bool_,
    ]
    run_profile_2d_array(dtypes)
    run_profile_transposed_arrays(dtypes)
    run_profile_jpeg()
    run_profile_png()
    print("Done.")
