use pyo3::{IntoPy, Python, PyObject};
use serde_json::Value;
use pyo3::prelude::{PyResult, PyErr};
use pyo3::exceptions::{PyValueError, PyIOError};
use ndarray::{Array2};
use num_traits::identities::Zero;
use numpy::{Element, IntoPyArray};
use pyo3::conversion::AsPyPointer;

use crate::parsing::parse_utils::get_shape;


pub fn to_f32(i: f64) -> f32 {
    i as f32
}


pub fn to_f64(i: f64) -> f64 {
    i
}


fn parse_float_0d<T>(py: Python, value: &Value, converter: &dyn Fn(f64) -> T) -> PyResult<PyObject> {
    if let Some(float) = value.as_f64() {
        Ok(float.into_py(py))
    }
    else {
        Err(PyErr::new::<PyIOError, _>(format!("Unable to parse value: {:?} as type np.float", value)))
    }
}


fn parse_float_1d<T: Element>(py: Python, value: &Value, shape: Vec<usize>, converter: &dyn Fn(f64) -> T) -> PyResult<PyObject> {
    let mut out: Vec<T> = Vec::new();
    for i in 0..shape[0] {
        if let Some(v) = value[i].as_f64() {
            out.push(converter(v));
        }
        else {
            return Err(PyErr::new::<PyValueError, _>(format!("Found {:?} in json list, cannot convert to np.float", value[i])))
        }
    }
    Ok(unsafe { PyObject::from_borrowed_ptr(py, out.into_pyarray(py).as_ptr())})
}


fn parse_float_2d<T: Element + Zero>(py:Python, value: &Value, shape: Vec<usize>, converter: &dyn Fn(f64) -> T) -> PyResult<PyObject> {
    let mut out = Array2::<T>::zeros((shape[0], shape[1]));
    for i in 0..shape[0] {
        for j in 0..shape[1] {
            if let Some(v) = value[i][j].as_f64(){
                out[[i, j]] = converter(v);
            }
            else {
                return Err(PyErr::new::<PyValueError, _>(format!("Found {:?} in json list, cannot convert to np.float", value[i][j])))
            }
        }
    }
    Ok(unsafe { PyObject::from_borrowed_ptr(py, out.into_pyarray(py).as_ptr())})
}


pub fn parse_float_array<T: Element + Zero>(py: Python, value: &Value, converter: &dyn Fn(f64) -> T) -> PyResult<PyObject> {
    let shape = get_shape(value);
    match shape.len() {
        0 => parse_float_0d(py, value, &converter),
        1 => parse_float_1d(py, value, shape, &converter),
        2 => parse_float_2d(py, value, shape, &converter),
        _ => Err(PyErr::new::<PyValueError, _>(format!("{}-d array not currently supported", shape.len())))
    }
}