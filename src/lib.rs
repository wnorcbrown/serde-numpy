use std::collections::HashMap;

use ::pyo3::exceptions::PyIOError;
use ::pyo3::PyErr;
use pyo3::prelude::{pymodule, PyModule, PyResult, Python, PyObject, IntoPy};
use pyo3::types::PyDict;

use serde::de::DeserializeSeed;
mod parsing;
use parsing::PyStructure;

use parsing::{StructureDescriptor};

#[pymodule]
fn serde_numpy(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    #[pyfn(m)]
    #[pyo3(name = "deserialize")]
    fn deserialize<'py>(
        py: Python<'py>,
        json_str: &[u8],
        structure: HashMap<&'py str, PyStructure>,
    ) -> PyResult<&'py PyDict> {
        let result = serde_json::from_slice(json_str);

        match result {
            Ok(value) => parsing::deserialize(py, &value, structure),
            Err(_err) => return Err(PyErr::new::<PyIOError, _>("Invalid JSON")),
        }
    }


    #[pyfn(m)]
    #[pyo3(name = "deserialize_new")]
    fn deserialize_new<'py>(
        py: Python<'py>,
        json_str: &[u8],
        structure: &[u8],
    ) -> PyResult<PyObject> {
        let structure_descriptor: StructureDescriptor = serde_json::from_slice(structure).unwrap();
        let result = structure_descriptor
                    .deserialize(&mut serde_json::Deserializer::from_slice(json_str));

        match result {
            Ok(value) => Ok(value.into_py(py)),
            Err(_err) => return Err(PyErr::new::<PyIOError, _>("Invalid JSON")),
        }
    }

    Ok(())
}
