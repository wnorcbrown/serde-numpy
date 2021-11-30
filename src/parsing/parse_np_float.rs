use serde_json::Value;
use pyo3::prelude::{PyResult, PyErr};
use pyo3::exceptions::{PyValueError, PyIOError};
use ndarray::{Array2};


use crate::parsing::parse_utils::{ReturnTypes, get_shape};


fn parse_float32_0d(value: &Value) -> PyResult<ReturnTypes> {
    if let Some(float) = value.as_f64() {
        Ok(ReturnTypes::Float(float))
    }
    else {
        Err(PyErr::new::<PyIOError, _>(format!("Unable to parse value: {:?} as type np.float32", value)))
    }
}


fn parse_float32_1d(value: &Value, shape: Vec<usize>) -> PyResult<ReturnTypes> {
    let mut out = Vec::new();
    for i in 0..shape[0] {
        if let Some(v) = value[i].as_f64() {
            out.push(v as f32);
        }
        else {
            return Err(PyErr::new::<PyValueError, _>(format!("Found {:?} in json list, cannot convert to np.float32", value[i])))
        }
    }
    Ok(ReturnTypes::Float32Array1D(out))
}


fn parse_float32_2d(value: &Value, shape: Vec<usize>) -> PyResult<ReturnTypes> {
    let mut out = Array2::<f32>::zeros((shape[0], shape[1]));
    for i in 0..shape[0] {
        for j in 0..shape[1] {
            if let Some(v) = value[i][j].as_f64(){
                out[[i, j]] = v as f32;
            }
            else {
                return Err(PyErr::new::<PyValueError, _>(format!("Found {:?} in json list, cannot convert to np.float32", value[i][j])))
            }
        }
    }
    Ok(ReturnTypes::Float32Array2D(out))
}


fn parse_float64_0d(value: &Value) -> PyResult<ReturnTypes> {
    if let Some(float) = value.as_f64() {
        Ok(ReturnTypes::Float(float))
    }
    else {
        Err(PyErr::new::<PyIOError, _>(format!("Unable to parse value: {:?} as type np.float64", value)))
    }
}


fn parse_float64_1d(value: &Value, shape: Vec<usize>) -> PyResult<ReturnTypes> {
    let mut out = Vec::new();
    for i in 0..shape[0] {
        if let Some(v) = value[i].as_f64() {
            out.push(v);
        }
        else {
            return Err(PyErr::new::<PyValueError, _>(format!("Found {:?} in json list, cannot convert to np.float64", value[i])))
        }
    }
    Ok(ReturnTypes::Float64Array1D(out))
}


fn parse_float64_2d(value: &Value, shape: Vec<usize>) -> PyResult<ReturnTypes> {
    let mut out = Array2::<f64>::zeros((shape[0], shape[1]));
    for i in 0..shape[0] {
        for j in 0..shape[1] {
            if let Some(v) = value[i][j].as_f64(){
                out[[i, j]] = v;
            }
            else {
                return Err(PyErr::new::<PyValueError, _>(format!("Found {:?} in json list, cannot convert to np.float64", value[i][j])))
            }
        }
    }
    Ok(ReturnTypes::Float64Array2D(out))
}


pub fn parse_float32(value: &Value) -> PyResult<ReturnTypes> {
    let shape = get_shape(value);
    match shape.len() {
        0 => parse_float32_0d(value),
        1 => parse_float32_1d(value, shape),
        2 => parse_float32_2d(value, shape),
        _ => Err(PyErr::new::<PyValueError, _>(format!("{}-d array not currently supported", shape.len())))
    }
}


pub fn parse_float64(value: &Value) -> PyResult<ReturnTypes> {
    let shape = get_shape(value);
    match shape.len() {
        0 => parse_float64_0d(value),
        1 => parse_float64_1d(value, shape),
        2 => parse_float64_2d(value, shape),
        _ => Err(PyErr::new::<PyValueError, _>(format!("{}-d array not currently supported", shape.len())))
    }
}