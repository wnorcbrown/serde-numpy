# serde-numpy

Here, we use [`maturin`][maturin] for building Python wheel and
[`poetry`][poetry] for managing Python dependencies and virtualenvs.
Following commands creates a virtualenv, install Python-side
dependencies, and install the extension to the virtualenv.

```bash
poetry install
poetry run maturin develop
```

Once the extension installed, you can run the extension from
Python REPL started by `poetry run python`:

```python
>>> import numpy as np
>>> import serde_numpy
>>> 
```

[maturin]: https://github.com/PyO3/maturin
[poetry]: https://python-poetry.org/
