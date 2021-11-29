use std::collections::HashMap;
use itertools::izip;

use serde_json::Value;

use numpy::IntoPyArray;
use numpy::PyArray;
use numpy::ndarray::Dim;
use ndarray::{ArrayBase, Array2, Array3, OwnedRepr};

use pyo3::exceptions::{PyValueError, PyIOError};
use pyo3::prelude::{Python, PyErr, PyResult, IntoPy, PyObject};
use pyo3::conversion::AsPyPointer;
use pyo3::types::PyType;



pub fn parse_int_column<'py>(py: Python<'py>, value: &Value, key: &str, index: usize) -> PyResult<NumpyTypes<'py>> {
    if let Some(stream) = value[key].as_array() {
        let mut out: Vec<i64> = Vec::new();
        for i in 0..stream.len() {
            if let Some(v) = stream[i][index].as_i64() {
                out.push(v);
            };
        }
        Ok(NumpyTypes::IntArray(out.into_pyarray(py)))
    }
    else {
        Err(PyErr::new::<PyValueError, _>(format!("json key {} does not exist", &key)))
    }
}


pub fn parse_float_column<'py>(py: Python<'py>, value: &Value, key: &str, index: usize) -> PyResult<NumpyTypes<'py>> {
    if let Some(stream) = value[key].as_array() {
        let mut out: Vec<f64> = Vec::new();
        for i in 0..stream.len() {
            if let Some(v) = stream[i][index].as_f64() {
                out.push(v);
            };
        }
        Ok(NumpyTypes::FloatArray(out.into_pyarray(py)))
    }
    else {
        Err(PyErr::new::<PyValueError, _>(format!("json key {} does not exist", &key)))
    }
}


pub fn parse_bool_column<'py>(py: Python<'py>, value: &Value, key: &str, index: usize) -> PyResult<NumpyTypes<'py>> {
    if let Some(stream) = value[key].as_array() {
        let mut out: Vec<bool> = Vec::new();
        for i in 0..stream.len() {
            if let Some(v) = stream[i][index].as_bool() {
                out.push(v);
            };
        }
        Ok(NumpyTypes::BoolArray(out.into_pyarray(py)))
    }
    else {
        Err(PyErr::new::<PyValueError, _>(format!("json key {} does not exist", &key)))
    }
}


// pub fn parse_str_column<'py>(py: Python<'py>, value: &Value, key: &str, index: usize) -> Result<NumpyTypes<'py>, String> {
//     if let Some(stream) = value[key].as_array() {
//         let mut out: Vec<String> = vec!["".to_string(); stream.len()];
//         for i in 0..stream.len() {
//             if let Some(v) = stream[i][index].as_str() {
//                 out[i] = v.to_string();
//             };
//         }
//         Ok(NumpyTypes::StringArray(out.into_pyarray(py)))
//     }
//     else {
//         Err(format!("json key {} does not exist", &key))
//     }
// }


pub fn parse_column_untyped<'py>(py: Python<'py>, value: &Value, key: &str, index: usize) -> PyResult<NumpyTypes<'py>> {
    let initial_value = &value[key][0][index];
    if initial_value.is_f64() {
        return parse_float_column(py, value, key, index)
    }
    else if initial_value.is_i64() {
        return parse_int_column(py, value, key, index)
    }
    else if initial_value.is_boolean() {
        return parse_bool_column(py, value, key, index)
    }
    // else if initial_value.is_string() {
    //     return parse_str_column(py, value, key, index)
    // }
    Err(PyErr::new::<PyIOError, _>(format!("cannot parse column with initial value {:?}", initial_value)))
}


pub fn parse_column_typed<'py>(py: Python<'py>, value: &Value, key: &str, index: usize, dtype: &str) -> PyResult<NumpyTypes<'py>> {
    if dtype == "float" {
        return parse_float_column(py, value, key, index)
    }
    else if dtype == "int" {
        return parse_int_column(py, value, key, index)
    }
    else if dtype == "bool" {
        return parse_bool_column(py, value, key, index)
    }
    // else if initial_value.is_string() {
    //     return parse_str_column(py, value, key, index)
    // }
    Err(PyErr::new::<PyIOError, _>(format!("cannot parse column with dtype {:?}", dtype)))
}


pub fn parse_columns<'py>(py: Python<'py>, value: &Value, key: &str, indexes: Vec<usize>, dtypes: Vec<&str>) -> PyResult<Vec<NumpyTypes<'py>>> {
    let mut out = Vec::new();
    for (index, dtype) in izip!(indexes, dtypes) {
        if let Ok(vector) = parse_column_typed(py, &value, &key, index, dtype) {
            out.push(vector);
        }
        else {
            return Err(PyErr::new::<PyIOError, _>(format!("cannot parse column at index {} for key {}", index, key)))
        }
    }
    Ok(out)
}


pub fn parse_keys<'py>(py: Python<'py>, value: &Value, keys: Vec<&str>, indexes: Vec<Vec<usize>>, types: Vec<Vec<&str>>) -> PyResult<HashMap<String, Vec<NumpyTypes<'py>>>> {
    assert_eq!(keys.len(), indexes.len(), "Number of keys: {} is not the same as number of indexes: {}", keys.len(), indexes.len());
    let mut out = HashMap::new();
    for (key, indexes_, types_) in izip!(keys, indexes, types) {
        let result = parse_columns(py, &value, key, indexes_, types_);
        match result  {
            Ok(vector) => {out.insert(String::from(key), vector);},
            Err(err) => return Err(err)
        }
    }
    Ok(out)
}



#[derive(Debug)]
pub enum NumpyTypes <'py> {
    IntArray(&'py PyArray<i64, Dim<[usize; 1]>>),
    FloatArray(&'py PyArray<f64, Dim<[usize; 1]>>),
    BoolArray(&'py PyArray<bool, Dim<[usize; 1]>>),
    // StringArray(&'py PyArray<String, Dim<[usize; 1]>>),
}

impl IntoPy<PyObject> for NumpyTypes<'_> {
    fn into_py(self, py: Python<'_>) -> PyObject {
        match self {
            NumpyTypes::IntArray(array) => unsafe { PyObject::from_borrowed_ptr(py, array.as_ptr())},
            NumpyTypes::FloatArray(array) => unsafe { PyObject::from_borrowed_ptr(py, array.as_ptr())},
            NumpyTypes::BoolArray(array) => unsafe { PyObject::from_borrowed_ptr(py, array.as_ptr())},
        }
    }
}




pub enum ReturnTypes {
    String(String),
    Bool(bool),
    Int(i64),
    Float(f64),
    FloatArray1D(Vec<f32>),
    FloatArray2D(ArrayBase<OwnedRepr<f32>, Dim<[usize; 2]>>),
}

impl IntoPy<PyObject> for ReturnTypes {
    fn into_py(self, py: Python<'_>) -> PyObject {
        match self {
            ReturnTypes::String(value) => value.into_py(py),
            ReturnTypes::Bool(value) => value.into_py(py),
            ReturnTypes::Int(value) => value.into_py(py),
            ReturnTypes::Float(value) => value.into_py(py),
            ReturnTypes::FloatArray1D(value) => unsafe { PyObject::from_borrowed_ptr(py, value.into_pyarray(py).as_ptr())},
            ReturnTypes::FloatArray2D(value) => unsafe { PyObject::from_borrowed_ptr(py, value.into_pyarray(py).as_ptr())},
        }
    }
}



fn parse_str(value: &Value) -> PyResult<ReturnTypes> {
    if let Some(string) = value.as_str() {
        Ok(ReturnTypes::String(String::from(string)))
    }
    else {
        Err(PyErr::new::<PyIOError, _>(format!("Unable to parse value: {:?} as type str", value)))
    }
}

fn parse_bool(value: &Value) -> PyResult<ReturnTypes> {
    if let Some(bool) = value.as_bool() {
        Ok(ReturnTypes::Bool(bool))
    }
    else {
        Err(PyErr::new::<PyIOError, _>(format!("Unable to parse value: {:?} as type bool", value)))
    }
}


fn parse_int(value: &Value) -> PyResult<ReturnTypes> {
    if let Some(int) = value.as_i64() {
        Ok(ReturnTypes::Int(int))
    }
    else {
        Err(PyErr::new::<PyIOError, _>(format!("Unable to parse value: {:?} as type int", value)))
    }
}


fn parse_float(value: &Value) -> PyResult<ReturnTypes> {
    if let Some(float) = value.as_f64() {
        Ok(ReturnTypes::Float(float))
    }
    else {
        Err(PyErr::new::<PyIOError, _>(format!("Unable to parse value: {:?} as type float", value)))
    }
}


fn _get_shape(value: &Value, mut shape: Vec<usize>) -> Vec<usize> {
    if let Some(stream) = value.as_array() {
        shape.push(stream.len());
        _get_shape(&value[0], shape)
    }
    else {
        shape
    }
}

fn get_shape(value: &Value) -> Vec<usize> {
    _get_shape(value, Vec::new())
}


fn parse_float32_0d(value: &Value) -> PyResult<ReturnTypes> {
    if let Some(float) = value.as_f64() {
        Ok(ReturnTypes::Float(float))
    }
    else {
        Err(PyErr::new::<PyIOError, _>(format!("Unable to parse value: {:?} as type np.float32", value)))
    }
}


// pub fn parse_float_column<'py>(py: Python<'py>, value: &Value, key: &str, index: usize) -> PyResult<NumpyTypes<'py>> {
//     if let Some(stream) = value[key].as_array() {
//         let mut out: Vec<f64> = Vec::new();
//         for i in 0..stream.len() {
//             if let Some(v) = stream[i][index].as_f64() {
//                 out.push(v);
//             };
//         }
//         Ok(NumpyTypes::FloatArray(out.into_pyarray(py)))
//     }
//     else {
//         Err(PyErr::new::<PyValueError, _>(format!("json key {} does not exist", &key)))
//     }
// }


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
    Ok(ReturnTypes::FloatArray1D(out))
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
    Ok(ReturnTypes::FloatArray2D(out))
}



fn parse_float32(value: &Value) -> PyResult<ReturnTypes> {
    let shape = get_shape(value);
    match shape.len() {
        0 => parse_float32_0d(value),
        1 => parse_float32_1d(value, shape),
        2 => parse_float32_2d(value, shape),
        _ => Err(PyErr::new::<PyValueError, _>(format!("{}-d array not currently supported", shape.len())))
    }
}



pub fn deserialize<'py>(py: Python<'py>, value: Value, structure: HashMap<&'py str, &'py PyType>) -> PyResult<HashMap<&'py str, ReturnTypes>> {
    let mut out = HashMap::new();
    for (k, v) in structure {
        if let Ok(type_name) = v.name() {
            match type_name {
                "str" => {out.insert(k, parse_str(&value[k])?);},
                "bool" => {out.insert(k, parse_bool(&value[k])?);},
                "int" => {out.insert(k, parse_int(&value[k])?);},
                "float" => {out.insert(k, parse_float(&value[k])?);},
                "float32" => {out.insert(k, parse_float32(&value[k])?);},
                _ => {return Err(PyErr::new::<PyValueError, _>(format!("{:?} type not supported", v)))}
            }
        }
    }
    Ok(out)
}