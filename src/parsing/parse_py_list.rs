use pyo3::{IntoPy, PyObject, Python};
use serde_json::Value;

use pyo3::prelude::{PyResult, PyErr};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::conversion::AsPyPointer;

use crate::parsing::parse_utils::get_shape;


fn parse_0d<U: IntoPy<PyObject>>(py: Python, 
                                 value: &Value, 
                                 as_type: &dyn Fn(&Value) -> Option<U>
                                ) -> PyResult<PyObject> {
    if let Some(out) = as_type(value) {
        Ok(out.into_py(py))
    }
    else {
        Err(PyErr::new::<PyTypeError, _>(format!("Unable to parse value: {:?} as type int", value)))
    }
}


fn parse_1d<U: IntoPy<PyObject>>(py: Python, 
                                 value: &Value, 
                                 shape: Vec<usize>, 
                                 as_type: &dyn Fn(&Value) -> Option<U>, 
                                ) -> PyResult<PyObject> {
    let mut out = Vec::new();
    for i in 0..shape[0] {
        if let Some(v) = as_type(&value[i]) {
            out.push(v);
        }
        else {
            return Err(PyErr::new::<PyTypeError, _>(format!("Found {:?} in json list, cannot convert to np.float", value[i])))
        }
    }
    Ok(out.into_py(py))
}


fn parse_2d<U: IntoPy<PyObject>>(py: Python, 
                                  value: &Value, 
                                  shape: Vec<usize>, 
                                  as_type: &dyn Fn(&Value) -> Option<U>, 
                                 ) -> PyResult<PyObject> {
    let mut out = Vec::new();
    for i in 0..shape[0] {
        let mut row = Vec::new();
        for j in 0..shape[1] {
            if let Some(v) = as_type(&value[i][j]){
                row.push(v);
            }
            else {
                return Err(PyErr::new::<PyTypeError, _>(format!("Found {:?} in json list, cannot convert to np.float", value[i][j])))
            }
        }
        out.push(row);
    }
    Ok(out.into_py(py))
}


pub fn parse_list<U: IntoPy<PyObject>>(py: Python, value: &Value, as_type: &dyn Fn(&Value) -> Option<U>) -> PyResult<PyObject> {
    let shape = get_shape(value);
    match shape.len() {
        0 => parse_0d(py, value, &as_type),
        1 => parse_1d(py, value, shape, &as_type),
        2 => parse_2d(py, value, shape, &as_type),
        _ => Err(PyErr::new::<PyValueError, _>(format!("{}-d list not currently supported", shape.len())))
    }
}