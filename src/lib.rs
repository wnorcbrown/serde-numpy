use ::pyo3::exceptions::PyIOError;
use ::pyo3::PyErr;
use pyo3::prelude::{pymodule, PyModule, PyResult, Python, PyObject, IntoPy};

use serde::de::DeserializeSeed;
mod parsing;
use parsing::{StructureDescriptor, Structure};

#[pymodule]
fn serde_numpy(_py: Python<'_>, m: &PyModule) -> PyResult<()> {

    #[pyfn(m)]
    #[pyo3(name = "deserialize")]
    fn deserialize<'py>(
        py: Python<'py>,
        json_str: &[u8],
        structure: Structure,
    ) -> PyResult<PyObject> {
        let structure_descriptor = StructureDescriptor{data: structure};
        let result = structure_descriptor
                    .deserialize(&mut serde_json::Deserializer::from_slice(json_str));

        match result {
            Ok(value) => Ok(value.into_py(py)),
            Err(_err) => return Err(PyErr::new::<PyIOError, _>("Invalid JSON")),
        }
    }

    Ok(())
}
