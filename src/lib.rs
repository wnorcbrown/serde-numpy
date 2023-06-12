use std::fs::read;

use pyo3::{self, wrap_pyfunction, pyfunction};
use pyo3::exceptions::{PyTypeError, PyValueError, PyIOError};
use pyo3::prelude::{pyclass, pymethods, pymodule, IntoPy, PyModule, PyObject, PyResult, Python};
use pyo3::types::PyType;

use serde::de::DeserializeSeed;
mod parsing;
use parsing::StructureDescriptor;

mod img;
use img::{decode_jpeg_bytes, decode_png_bytes};
    

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

    fn deserialize_msgpack(&self, py: Python, msgpack_bytes: &[u8]) -> PyResult<PyObject> {
        // need to get rid of clone
        let md = &mut rmp_serde::decode::Deserializer::new(msgpack_bytes);
        let result = self.structure_descriptor.clone().deserialize(md);
        match result {
            Ok(value) => value.into_py(py),
            Err(err) => Err(PyTypeError::new_err(err.to_string())),
        }
    }
}

#[pyfunction]
fn decode_jpeg(py: Python, jpeg_bytes: &[u8]) -> PyResult<PyObject> {
    match decode_jpeg_bytes(jpeg_bytes) {
        Ok(output) => output.into_py(py),
        Err(err) => Err(PyIOError::new_err(err.to_string())),
    }
}

#[pyfunction]
fn read_jpeg(py: Python, path: &str) -> PyResult<PyObject> {
    match read(path) {
        Ok(jpeg_bytes) => decode_jpeg(py, &jpeg_bytes),
        Err(err) => Err(PyIOError::new_err(err.to_string())),
    }
}


#[pyfunction]
fn decode_png(py: Python, png_bytes: &[u8]) -> PyResult<PyObject> {
    match decode_png_bytes(png_bytes) {
        Ok(output) => output.into_py(py),
        Err(err) => Err(PyIOError::new_err(format!("{:?}", err))),
    }
}


#[pyfunction]
fn read_png(py: Python, path: &str) -> PyResult<PyObject> {
    match read(path) {
        Ok(png_bytes) => decode_png(py, &png_bytes),
        Err(err) => Err(PyIOError::new_err(err.to_string())),
    }
}


#[pymodule]
fn serde_numpy(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<NumpyDeserializer>()?;

    m.add_function(wrap_pyfunction!(decode_jpeg, m)?)?;
    m.add_function(wrap_pyfunction!(read_jpeg, m)?)?;
    m.add_function(wrap_pyfunction!(decode_png, m)?)?;
    m.add_function(wrap_pyfunction!(read_png, m)?)?;

    Ok(())
}
