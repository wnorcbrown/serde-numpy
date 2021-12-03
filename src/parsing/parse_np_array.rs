use pyo3::{Python, PyObject};
use serde_json::Value;

use num_traits::identities::Zero;

use numpy::{Element, IntoPyArray};
use ndarray::{Array1, Array2};

use pyo3::prelude::{PyResult, PyErr};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::conversion::AsPyPointer;



use crate::parsing::parse_utils::get_shape;


fn parse_1d<U, T: Element + Zero>(py:Python, value: &Value, shape: Vec<usize>, as_type: &dyn Fn(&Value) -> Option<U>, converter: &dyn Fn(U) -> T) -> PyResult<PyObject>
{
    let mut out = Array1::<T>::zeros(shape[0]);
    for i in 0..shape[0] {
        if let Some(v) = as_type(&value[i]) {
            out[i] = converter(v);
        }
        else {
            return Err(PyErr::new::<PyTypeError, _>(format!("Found {:?} in json list, cannot convert to np.float", value[i])))
        }
    }
    Ok(unsafe { PyObject::from_borrowed_ptr(py, out.into_pyarray(py).as_ptr())})
}


fn parse_2d<U, T: Element + Zero>(py:Python, 
                                  value: &Value, 
                                  shape: Vec<usize>, 
                                     as_type: &dyn Fn(&Value) -> Option<U>, 
                                     converter: &dyn Fn(U) -> T) -> PyResult<PyObject>
{
    let mut out = Array2::<T>::zeros((shape[0], shape[1]));
    for i in 0..shape[0] {
        for j in 0..shape[1] {
            if let Some(v) = as_type(&value[i][j]){
                out[[i, j]] = converter(v);
            }
            else {
                return Err(PyErr::new::<PyTypeError, _>(format!("Found {:?} in json list, cannot convert to np.float", value[i][j])))
            }
        }
    }
    Ok(unsafe { PyObject::from_borrowed_ptr(py, out.into_pyarray(py).as_ptr())})
}


pub fn parse_array<U, T: Element + Zero>(py: Python, value: &Value, as_type: &dyn Fn(&Value) -> Option<U>, converter: &dyn Fn(U) -> T) -> PyResult<PyObject> {
    let shape = get_shape(value);
    match shape.len() {
        // 0 => parse_0d(py, value, as_type, &converter),
        1 => parse_1d(py, value, shape, &as_type, &converter),
        2 => parse_2d(py, value, shape, &as_type, &converter),
        _ => Err(PyErr::new::<PyValueError, _>(format!("{}-d array not currently supported", shape.len())))
    }
}