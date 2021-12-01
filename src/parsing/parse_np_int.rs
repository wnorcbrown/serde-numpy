use pyo3::{IntoPy, Python, PyObject};
use serde_json::Value;
use pyo3::prelude::{PyResult, PyErr};
use pyo3::exceptions::{PyValueError, PyIOError};
use ndarray::{Array2};
use num_traits::identities::Zero;
use numpy::{Element, IntoPyArray};
use pyo3::conversion::AsPyPointer;

use crate::parsing::parse_utils::get_shape;


pub fn to_i32(i: i64) -> i32 {
    i as i32
}

pub fn to_i64(i: i64) -> i64 {
    i
}

fn parse_int_0d<T>(py: Python, value: &Value, converter: &dyn Fn(i64) -> T) -> PyResult<PyObject> {
    if let Some(int) = value.as_i64() {
        Ok(int.into_py(py))
    }
    else {
        Err(PyErr::new::<PyIOError, _>(format!("Unable to parse value: {:?} as type np.int", value)))
    }
}


fn parse_int_1d<T: Element>(py: Python, value: &Value, shape: Vec<usize>, converter: &dyn Fn(i64) -> T) -> PyResult<PyObject> {
    let mut out: Vec<T> = Vec::new();
    for i in 0..shape[0] {
        if let Some(v) = value[i].as_i64() {
            out.push(converter(v));
        }
        else {
            return Err(PyErr::new::<PyValueError, _>(format!("Found {:?} in json list, cannot convert to np.int", value[i])))
        }
    }
    Ok(unsafe { PyObject::from_borrowed_ptr(py, out.into_pyarray(py).as_ptr())})
}


fn parse_int_2d<T: Element + Zero>(py:Python, value: &Value, shape: Vec<usize>, converter: &dyn Fn(i64) -> T) -> PyResult<PyObject> {
    let mut out = Array2::<T>::zeros((shape[0], shape[1]));
    for i in 0..shape[0] {
        for j in 0..shape[1] {
            if let Some(v) = value[i][j].as_i64(){
                out[[i, j]] = converter(v);
            }
            else {
                return Err(PyErr::new::<PyValueError, _>(format!("Found {:?} in json list, cannot convert to np.int", value[i][j])))
            }
        }
    }
    Ok(unsafe { PyObject::from_borrowed_ptr(py, out.into_pyarray(py).as_ptr())})
}


pub fn parse_int_array<T: Element + Zero>(py: Python, value: &Value, converter: &dyn Fn(i64) -> T) -> PyResult<PyObject> {
    let shape = get_shape(value);
    match shape.len() {
        0 => parse_int_0d(py, value, &converter),
        1 => parse_int_1d(py, value, shape, &converter),
        2 => parse_int_2d(py, value, shape, &converter),
        _ => Err(PyErr::new::<PyValueError, _>(format!("{}-d array not currently supported", shape.len())))
    }
}