# serde-numpy

Deserialize a subset of json keys directly into numpy arrays.

```bash
poetry install
poetry run maturin develop
```

Example usage (eventually):

```python
import serde_numpy

json_str = b"""
{'name': 'vill',
 'version': 3000,
 'arr1': [[1.254439975231648, -0.6893827594332794],
          [-0.2922560025562806, 0.5204819306523419]],
 'arr2': [[-100, -25], [-41, -62]],
 'stream1': [[-1.720294114558863, 0.5990469735869592, 0.0506514091042812, 0.7204746283872987, 1.5351637640639662],
             [72, 45, -58, -16, -14],
             [true, false, false, true, false]],
 'stream2': [[-2.1727126743596266, false, 'a'],
             [-0.06389102863189458, true, 'b'],
             [1.3716941547285826, true, 'c']],
 'nest': {'is_nest': True,
          'nestiness': 0.9999,
          'stream3': [[-0.11954010897451912, 28], 
                      [0.21599243355210992, 9]],
          'arr3': [true, false, true],
          'unused_key': 'a'}}"""

structure = {
    "name": str, 
    "version": int,
    "arr1": np.float32, # equivalent to calling np.array(x["arr1"], dtype=np.float32)
    "arr2": np.int64,
    "stream1": [np.float64, np.int32, np.bool], # for column-wise list of lists
    "stream2": [[np.float32, np.bool, np.str]], # for row-wise list of lists
    "nest": {
        "is_nest": bool,
        "nestiness": float,
        "stream3": [[np.float128, np.uint64]],
        "arr3": np.bool,
    }
}

deserialized = serde_numpy.deserialize(json_str, structure)
deserialized

{'name': 'vill',
 'version': 3000,
 'arr1': np.array([[1.254439975231648, -0.6893827594332794],
                   [-0.2922560025562806, 0.5204819306523419]]),
 'arr2': np.array([[-100, -25], [-41, -62]]),
 'stream1': [np.array([-1.720294114558863, 0.5990469735869592, 0.0506514091042812, 0.7204746283872987, 1.5351637640639662]),
             np.array([72, 45, -58, -16, -14]),
             np.array([True, False, False, True, False])],
 'stream2': [np.array([-2.1727126743596266, -0.06389102863189458, 1.3716941547285826]),
             np.array([False, True, True]),
             np.array(['a', 'b', 'c']),
 'nest': {'is_nest': True,
          'nestiness': 0.9999,
          'stream3': [np.array([-0.11954010897451912, 0.21599243355210992]), 
                      np.array([28, 9])],
          'arr3': np.array([True, False, True])}}"""



```

[maturin]: https://github.com/PyO3/maturin
[poetry]: https://python-poetry.org/
