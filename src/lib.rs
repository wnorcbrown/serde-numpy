use pyo3;
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::{pyclass, pymethods, pymodule, IntoPy, PyModule, PyObject, PyResult, Python};
use pyo3::types::PyType;

use serde::de::DeserializeSeed;
mod parsing;
use parsing::StructureDescriptor;

#[pyclass]
struct NumpyDeserializer {
    structure_descriptor: StructureDescriptor,
}

#[pymethods]
impl NumpyDeserializer {
    #[classmethod]
    fn from_dict(_cls: &PyType, py: Python, structure: PyObject) -> PyResult<Self> {
        match structure.extract(py) {
            Ok(data) => Ok(NumpyDeserializer { structure_descriptor: StructureDescriptor { data } }),
            Err(_) => Err(PyTypeError::new_err("structure unsupported. Currently sequences of nested structures are unsupported e.g. [{\"a\": {\"b\": Type}}])"))
        }
    }

    #[classmethod]
    fn from_json_bytes(_cls: &PyType, _py: Python, bytes: &[u8]) -> PyResult<Self> {
        let result = serde_json::from_slice(bytes);
        match result {
            Ok(data) => Ok(NumpyDeserializer {
                structure_descriptor: StructureDescriptor { data },
            }),
            Err(err) => Err(PyValueError::new_err(format!(
                "Error parsing structure bytes {}",
                err
            ))), // better handling needed
        }
    }

    fn deserialize_json(&self, py: Python, json_str: &[u8]) -> PyResult<PyObject> {
        // need to get rid of clone
        let result = self
            .structure_descriptor
            .clone()
            .deserialize(&mut serde_json::Deserializer::from_slice(json_str));
        match result {
            Ok(value) => value.into_py(py),
            Err(err) => Err(PyTypeError::new_err(err.to_string())),
        }
    }
}

#[pymodule]
fn serde_numpy(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<NumpyDeserializer>()?;
    Ok(())
}
