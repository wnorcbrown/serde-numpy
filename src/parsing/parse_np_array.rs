use pyo3::{IntoPy, PyObject, Python};
use serde_json::Value;
use serde_json::value::Index;

use num_traits::identities::Zero;

use numpy::{Element, IntoPyArray};
use ndarray::{Array1, Array2};

use pyo3::prelude::{PyResult, PyErr};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::conversion::AsPyPointer;

use crate::parsing::parse_utils::get_shape;


fn parse_0d<U: IntoPy<PyObject>, I: Index>(py: Python, 
                                 value: &Value, 
                                 as_type: &dyn Fn(&Value) -> Option<U>,
                                 opt_column_selector: Option<I>,
                                ) -> PyResult<PyObject> {
    let element;
    match opt_column_selector {
        None => {element = value;},
        Some(ref column_selector) => {element = &value[column_selector];}
    }
    if let Some(out) = as_type(element) {
        Ok(out.into_py(py))
    }
    else {
        Err(PyErr::new::<PyTypeError, _>(format!("Unable to parse value: {:?} as type int", value)))
    }
}


fn parse_1d<U, T: Element + Zero, I: Index>(py: Python, 
                                  value: &Value, 
                                  shape: Vec<usize>, 
                                  as_type: &dyn Fn(&Value) -> Option<U>, 
                                  converter: &dyn Fn(U) -> T,
                                  opt_column_selector: Option<I>,
                                 ) -> PyResult<PyObject> {
    let mut out = Array1::<T>::zeros(shape[0]);
    for i in 0..shape[0] {
        let element;
        
        match opt_column_selector {
            None => {element = &value[i];},
            Some(ref column_selector) => {element = &value[i][column_selector];}
        }
        if let Some(v) = as_type(element) {
            out[i] = converter(v);
        }
        else {
            return Err(PyErr::new::<PyTypeError, _>(format!("Found {:?} in json list, cannot convert to np.float", value[i])))
        }
    }
    Ok(unsafe { PyObject::from_borrowed_ptr(py, out.into_pyarray(py).as_ptr())})
}


fn parse_2d<U, T: Element + Zero>(py: Python, 
                                  value: &Value, 
                                  shape: Vec<usize>, 
                                  as_type: &dyn Fn(&Value) -> Option<U>, 
                                  converter: &dyn Fn(U) -> T
                                 ) -> PyResult<PyObject> {
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


pub fn parse_array<U: IntoPy<PyObject>, T: Element + Zero, I: Index>(py: Python, 
                                                           value: &Value, 
                                                           as_type: &dyn Fn(&Value) -> Option<U>, 
                                                           converter: &dyn Fn(U) -> T,
                                                           opt_column_selector: Option<I>,
                                                          ) -> PyResult<PyObject> {
    let shape = get_shape(value, &opt_column_selector);
    match shape.len() {
        0 => parse_0d(py, value, &as_type, opt_column_selector),
        1 => parse_1d(py, value, shape, &as_type, &converter, opt_column_selector),
        2 => parse_2d(py, value, shape, &as_type, &converter),
        _ => Err(PyErr::new::<PyValueError, _>(format!("{}-d array not currently supported", shape.len())))
    }
}