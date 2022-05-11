use std::collections::HashMap;

use serde;
use serde::Deserialize;
use serde_json::Value;

use pyo3::prelude::*;

#[derive(Debug, PartialEq, Deserialize)]
pub struct PythonType(pub Value);

impl IntoPy<PyObject> for PythonType {
    fn into_py(self, py: Python) -> PyObject {
        match self.0 {
            Value::Null => ().into_py(py),
            Value::Bool(val) => val.to_object(py),
            Value::Number(number) => {
                if let Some(val) = number.as_u64() {
                    val.into_py(py)
                } else if let Some(val) = number.as_i64() {
                    val.into_py(py)
                } else if let Some(val) = number.as_f64() {
                    val.into_py(py)
                } else {
                    ().into_py(py)
                }
            }
            Value::String(string) => string.to_object(py),
            Value::Array(arr) => arr
                .into_iter()
                .map(|x| PythonType(x).into_py(py))
                .collect::<Vec<PyObject>>()
                .into_py(py),
            Value::Object(map) => map
                .into_iter()
                .map(|(k, v)| (k, PythonType(v).into_py(py)))
                .collect::<HashMap<String, PyObject>>()
                .into_py(py),
        }
    }
}
