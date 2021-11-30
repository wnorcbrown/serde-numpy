use serde_json::Value;
use pyo3::prelude::{PyResult, PyErr};
use pyo3::exceptions::{PyValueError, PyIOError};
use ndarray::{Array2};


use crate::parsing::parse_utils::{ReturnTypes, get_shape};


fn parse_int32_0d(value: &Value) -> PyResult<ReturnTypes> {
    if let Some(int) = value.as_i64() {
        Ok(ReturnTypes::Int(int))
    }
    else {
        Err(PyErr::new::<PyIOError, _>(format!("Unable to parse value: {:?} as type np.int32", value)))
    }
}


fn parse_int32_1d(value: &Value, shape: Vec<usize>) -> PyResult<ReturnTypes> {
    let mut out = Vec::new();
    for i in 0..shape[0] {
        if let Some(v) = value[i].as_i64() {
            out.push(v as i32);
        }
        else {
            return Err(PyErr::new::<PyValueError, _>(format!("Found {:?} in json list, cannot convert to np.int32", value[i])))
        }
    }
    Ok(ReturnTypes::Int32Array1D(out))
}


fn parse_int32_2d(value: &Value, shape: Vec<usize>) -> PyResult<ReturnTypes> {
    let mut out = Array2::<i32>::zeros((shape[0], shape[1]));
    for i in 0..shape[0] {
        for j in 0..shape[1] {
            if let Some(v) = value[i][j].as_i64(){
                out[[i, j]] = v as i32;
            }
            else {
                return Err(PyErr::new::<PyValueError, _>(format!("Found {:?} in json list, cannot convert to np.int32", value[i][j])))
            }
        }
    }
    Ok(ReturnTypes::Int32Array2D(out))
}


fn parse_int64_0d(value: &Value) -> PyResult<ReturnTypes> {
    if let Some(int) = value.as_i64() {
        Ok(ReturnTypes::Int(int))
    }
    else {
        Err(PyErr::new::<PyIOError, _>(format!("Unable to parse value: {:?} as type np.int64", value)))
    }
}


fn parse_int64_1d(value: &Value, shape: Vec<usize>) -> PyResult<ReturnTypes> {
    let mut out = Vec::new();
    for i in 0..shape[0] {
        if let Some(v) = value[i].as_i64() {
            out.push(v);
        }
        else {
            return Err(PyErr::new::<PyValueError, _>(format!("Found {:?} in json list, cannot convert to np.int64", value[i])))
        }
    }
    Ok(ReturnTypes::Int64Array1D(out))
}


fn parse_int64_2d(value: &Value, shape: Vec<usize>) -> PyResult<ReturnTypes> {
    let mut out = Array2::<i64>::zeros((shape[0], shape[1]));
    for i in 0..shape[0] {
        for j in 0..shape[1] {
            if let Some(v) = value[i][j].as_i64(){
                out[[i, j]] = v;
            }
            else {
                return Err(PyErr::new::<PyValueError, _>(format!("Found {:?} in json list, cannot convert to np.int64", value[i][j])))
            }
        }
    }
    Ok(ReturnTypes::Int64Array2D(out))
}


pub fn parse_int32(value: &Value) -> PyResult<ReturnTypes> {
    let shape = get_shape(value);
    match shape.len() {
        0 => parse_int32_0d(value),
        1 => parse_int32_1d(value, shape),
        2 => parse_int32_2d(value, shape),
        _ => Err(PyErr::new::<PyValueError, _>(format!("{}-d array not currently supported", shape.len())))
    }
}


pub fn parse_int64(value: &Value) -> PyResult<ReturnTypes> {
    let shape = get_shape(value);
    match shape.len() {
        0 => parse_int64_0d(value),
        1 => parse_int64_1d(value, shape),
        2 => parse_int64_2d(value, shape),
        _ => Err(PyErr::new::<PyValueError, _>(format!("{}-d array not currently supported", shape.len())))
    }
}