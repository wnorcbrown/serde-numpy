# serde-numpy

serde-numpy is a library for efficient deserializing of various file formats directly into numpy arrays. 

See how it works for:
- [images](#image-formats)
- [json formats](#json-formats)


## Installation
Currently only available for linux, python >= 3.7

```bash
pip install --upgrade pip
pip install serde-numpy
```

# Image formats

## Motivation

Speed up your neural net training loops!

Reading images quickly off disk into an array format is a common problem in training computer vision models. By using serde numpy we get access to the fast [zune](https://github.com/etemesi254/zune-image) library in rust, which we use to write directly to `numpy` arrays - no intermediate `Image` formats, therefore less copying and less python.

Here's a demonstration of the speed up:
```python
# a lot of libraries do this:
def pillow_plus_np(path: str) -> np.ndarray:
    img = Image.open(path)
    return np.array(img)

# this is a bit faster:
def torch_read_png(path: str) -> np.ndarray:
    byte_array = torchvision.io.read_file(path)
    return torchvision.io.decode_png(byte_array)

BATCH_SIZE = 128
NUM_WORKERS = 4
functions = (
    "serde_numpy.read_png",
    "torch_read_png",
    "torchvision.io.read_image",
    "pillow_plus_npasarray",
)
for name in functions:
    exec(f"f = {name}")
    ds = ImageFolder(base.as_posix(), loader=f)
    loader = DataLoader(
        ds, 
        batch_size=BATCH_SIZE, 
        num_workers=NUM_WORKERS, 
        drop_last=True)
    for batch in tqdm(loader, total=len(ds) // BATCH_SIZE, desc=name, ncols=96):
        x, y = batch
```
```
serde_numpy.read_png: 100%|████████████████████████████████| 3906/3906 [00:05<00:00, 760.66it/s]
torch_read_png: 100%|██████████████████████████████████████| 3906/3906 [00:08<00:00, 467.31it/s]
torchvision.io.read_image: 100%|███████████████████████████| 3906/3906 [00:08<00:00, 463.77it/s]
pillow_plus_npasarray: 100%|███████████████████████████████| 3906/3906 [00:13<00:00, 294.49it/s]
```
```python
# serde_numpy is nearly twice the speed
```
## Example usage

```python
>>> from serde_numpy import decode_jpeg, read_jpeg, decode_png, read_png
>>> 
>>> img = read_jpeg("test.jpg")
>>> img
array([[[ 75,  29,  82],
        [ 96,  56, 133],
        [ 72,  47, 168],
        [ 63,  56, 179]],

       [[216, 176, 203],
        [173, 139, 190],
        [111,  93, 188],
        [129, 128, 225]],

       [[ 75,  46,  21],
        [ 73,  51,  48],
        [ 81,  73, 115],
        [157, 167, 209]],

       [[165, 142,  99],
        [181, 165, 144],
        [169, 169, 188],
        [185, 203, 222]]], dtype=uint8)
>>> 
>>> byte_array = open("test.png", "rb").read()
>>> img = decode_png(byte_array)
>>> img
array([[[ 33,  47, 146],
        [206,  19, 120],
        [185,   8,  55],
        [ 33,  54, 176]],

       [[252, 156, 169],
        [169, 139, 100],
        [ 24, 128, 222],
        [136, 146, 213]],

       [[ 28,  24, 192],
        [184,  51,  58],
        [ 39,  61, 252],
        [237, 165, 113]],

       [[239, 111,  72],
        [ 30, 242,  38],
        [165, 161, 223],
        [ 91, 246, 217]]], dtype=uint8)

```

## Benchmarks

All benchmarks were performed on an AMD Ryzen 9 3950X (Python 3.8.12, numpy 1.23.2, orjson 3.6.4). We compare serde_numpy's `decode_png` and `decode_jpeg` versus pillow's `Image.open` + `np.asarray` (which is the de facto standard for libraries than do a lot of image loading e.g. pytorch's `torchvision`).

### JPEG
JPEG decoding for square images:

![alt text](profile/jpeg_profile.png "Jpeg profiling")

### PNG
PNG decoding for square images:

![alt text](profile/png_profile.png "Png profiling")

# JSON Formats

## Motivation
If you've ever done something like this in your code:

```python
data = json.load(open("data.json"))

arr = np.array(data["x"])
```
then this library does it faster by using minimal array allocations and less python.

Speed ups are 1.5x - 8x times faster, depending on array sizes (and CPU), when compared to orjson + numpy.


## Usage


The user specifies the numpy dtypes within a `structure` corresponding to the data that they want to deserialize.

### N-dimensional array

A subset of the json's (or msgpack) keys are specified in the `structure` which is used to initialize the `NumpyDeserializer` and then that subset of keys are deserialized accordingly:


```python
>>> from serde_numpy import NumpyDeserializer
>>> 
>>> json_str = b"""
... {
...     "name": "coordinates",
...     "version": "0.1.0",
...     "arr": [[1.254439975231648, -0.6893827594332794],
...             [-0.2922560025562806, 0.5204819306523419]]
... }
... """
>>> 
>>> structure = {
...     'name': str,
...     'arr': np.float32
... }
>>> 
>>> deserializer = NumpyDeserializer.from_dict(structure)
>>> 
>>> deserializer.deserialize_json(json_str)
{'arr': array([[ 1.25444   , -0.68938273],
               [-0.292256  ,  0.52048194]], dtype=float32), 
 'name': 'coordinates'}
```

### Transposed arrays

Sometimes people store data in jsons in a row-wise fashion as opposed to column-wise. Therefore each row can contain multiple dtypes. serde-numpy allows you to specify the types of each row and then deserializes into columns. To tell the numpy deserializer that you want to transpose the columns put square brackets outside either a dictionary `[{key: Type, ...}]` like this example:

```python
>>> json_str = b"""
... {
...     "df": [{"a": 3, "b": 4.23},
...            {"a": 4, "b": 5.12}]
... }
... """
>>> 
>>> structure = {"df": [{"a": np.uint16, "b": np.float64}]}
>>> 
>>> deserializer = NumpyDeserializer.from_dict(structure)
>>> 
>>> deserializer.deserialize_json(json_str)
{'df': {'b': array([4.23, 5.12]), 'a': array([3, 4], dtype=uint16)}}
```
**or** put square brackets outside a list `[[Type, ...]]` of types:

```python
>>> json_str = b"""
... {
...     "df": [["i", true],
...            ["j", false],
...            ["k", true]]
... }
... """
>>> 
>>> structure = {"df": [[str, np.bool_]]}
>>> 
>>> deserializer = NumpyDeserializer.from_dict(structure)
>>> 
>>> deserializer.deserialize_json(json_str)
{'df': [['i', 'j', 'k'], array([ True, False,  True])]}
```

### Currently supported data formats:
- `JSON` :: `NumpyDeserializer.deserialize_json`
- `MessagePack` :: `NumpyDeserializer.deserialize_msgpack`

### Currently supported types:
Numpy types:
- `np.int8`
- `np.int16`
- `np.int32`
- `np.int64`
- `np.uint8`
- `np.uint16`
- `np.uint32`
- `np.uint64`
- `np.float32`
- `np.float64`
- `np.bool_`

Python types:
- `int`
- `float`
- `str`
- `dict`
- `list`

## Benchmarks

All benchmarks were performed on an AMD Ryzen 9 3950X (Python 3.8.12, numpy 1.23.2, orjson 3.6.4). Orjson was selected as the comparison as it is the fastest on python json benchmarks and we have also found it to be fastest in practice.

### 2D Array deserialization

Two tests are performed. The number of rows are kept constant at 10 while varying the number of columns **and** the number of columns are kept constant at 10 while varying the number of rows. We compare against `orjson.loads` + `np.array` with the desired data type. Results are presented below for deserializing arrays of various data types:

![alt text](profile/2darr_profile.png "2D Array profiling")

### Transposed arrays deserialization

For this test we test the speed of deserializing multiple data types which have been serialized in a row-wise fashion and converting it to column-wise arrays during deserializition.

![alt text](profile/transpose_profile.png "Transpose columns profiling")