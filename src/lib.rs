use std::convert::TryFrom;

use num_traits::Num;
use ::pyo3::exceptions::PyIOError;
use ::pyo3::PyErr;
use pyo3::prelude::{pymodule, IntoPy, PyModule, PyObject, PyResult, Python, pyclass, pymethods};
use pyo3::types::PyType;

use serde::de::DeserializeSeed;
mod parsing;
use parsing::{Structure, StructureDescriptor};

#[pyclass]
struct NumpyDeserializer {
    structure_descriptor: StructureDescriptor,
}

#[pymethods]
impl NumpyDeserializer {
    #[classmethod]
    fn from_dict(_cls: &PyType, _py: Python, structure: Structure) -> PyResult<Self> {
        Ok(NumpyDeserializer{structure_descriptor: StructureDescriptor { data: structure }})
    }

    #[classmethod]
    fn from_json_bytes(_cls: &PyType, _py: Python, bytes: &[u8]) -> PyResult<Self> {
        let result = serde_json::from_slice(bytes);
        match result {
            Ok(structure_descriptor) => Ok(NumpyDeserializer{structure_descriptor}),
            Err(_) => Err(PyErr::new::<PyIOError, _>("Invalid JSON in descriptor")) // better handling needed
        }

    }

    fn deserialize_json(&self, py: Python, json_str: &[u8]) -> PyResult<PyObject> {
        // need to get rid of clone
        let result =
            self.structure_descriptor.clone().deserialize(&mut serde_json::Deserializer::from_slice(json_str));
        match result {
                Ok(value) => Ok(value.into_py(py)),
                Err(_err) => return Err(PyErr::new::<PyIOError, _>("Invalid JSON")),
            }
    }
}

#[pymodule]
fn serde_numpy(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<NumpyDeserializer>()?;
    Ok(())
}
