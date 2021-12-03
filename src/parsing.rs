use std::collections::HashMap;
use itertools::izip;

use pyo3::FromPyObject;
use serde_json::Value;

use numpy::IntoPyArray;
use numpy::PyArray;
use numpy::ndarray::Dim;

use pyo3::exceptions::{PyValueError, PyTypeError};
use pyo3::prelude::{Python, PyErr, PyResult, IntoPy, PyObject, ToPyObject};
use pyo3::conversion::AsPyPointer;
use pyo3::types::PyType;

pub mod parse_utils;
mod parse_np_array;
use parse_np_array::{parse_array, to_f32, to_f64, to_i8, to_i16, to_i32, to_i64};



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
    Err(PyErr::new::<PyTypeError, _>(format!("cannot parse column with dtype {:?}", dtype)))
}


pub fn parse_columns<'py>(py: Python<'py>, value: &Value, key: &str, indexes: Vec<usize>, dtypes: Vec<&str>) -> PyResult<Vec<NumpyTypes<'py>>> {
    let mut out = Vec::new();
    for (index, dtype) in izip!(indexes, dtypes) {
        if let Ok(vector) = parse_column_typed(py, &value, &key, index, dtype) {
            out.push(vector);
        }
        else {
            return Err(PyErr::new::<PyTypeError, _>(format!("cannot parse column at index {} for key {}", index, key)))
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

















fn parse_str(py: Python, value: &Value) -> PyResult<PyObject> {
    if let Some(string) = value.as_str() {
        Ok(string.into_py(py))
    }
    else {
        Err(PyErr::new::<PyTypeError, _>(format!("Unable to parse value: {:?} as type str", value)))
    }
}

fn parse_bool(py: Python, value: &Value) -> PyResult<PyObject> {
    if let Some(bool) = value.as_bool() {
        Ok(bool.into_py(py))
    }
    else {
        Err(PyErr::new::<PyTypeError, _>(format!("Unable to parse value: {:?} as type bool", value)))
    }
}


fn parse_int(py: Python, value: &Value) -> PyResult<PyObject> {
    if let Some(int) = value.as_i64() {
        Ok(int.into_py(py))
    }
    else {
        Err(PyErr::new::<PyTypeError, _>(format!("Unable to parse value: {:?} as type int", value)))
    }
}


fn parse_float(py: Python, value: &Value) -> PyResult<PyObject> {
    if let Some(float) = value.as_f64() {
        Ok(float.into_py(py))
    }
    else {
        Err(PyErr::new::<PyTypeError, _>(format!("Unable to parse value: {:?} as type float", value)))
    }
}

#[derive(FromPyObject)]
#[derive(Debug)]
pub enum Structure <'py> {
    Type(&'py PyType),
    Map(HashMap<String, Structure<'py>>)
}

#[derive(Debug)]
pub enum OutStructure {
    Value(PyObject),
    Map(HashMap<String, OutStructure>)
}

impl IntoPy<PyObject> for OutStructure {
    fn into_py(self, py: Python) -> PyObject {
        match self {
            OutStructure::Value(v) => v.into_py(py),
            OutStructure::Map(v) => v.into_py(py),
        }
    }
}



fn value_as_f64(value: &Value) -> Option<f64> {
    value.as_f64()
}


fn value_as_i64(value: &Value) -> Option<i64> {
    value.as_i64()
}



pub fn deserialize<'py>(py: Python<'py>, value: &Value, structure: HashMap<String, Structure>) -> PyResult<HashMap<String, OutStructure>> {
    let mut out: HashMap<String, OutStructure> = HashMap::new();
    
    for (k, v) in structure {
        if let Structure::Type(t) = v {
            let result;
            if let Ok(type_name) = t.name() {
                match type_name {
                    "str" => {result = parse_str(py, &value[&k]);},
                    "bool" => {result = parse_bool(py, &value[&k]);},
                    "int" => {result = parse_int(py, &value[&k]);},
                    "float" => {result = parse_float(py, &value[&k]);},
                    "float32" => {result = parse_array(py, &value[&k], &value_as_f64, &to_f32);},
                    "float64" => {result = parse_array(py, &value[&k], &value_as_f64,&to_f64);},
                    "int8" => {result = parse_array(py, &value[&k], &value_as_i64,&to_i8);},
                    "int16" => {result = parse_array(py, &value[&k], &value_as_i64,&to_i16);},
                    "int32" => {result = parse_array(py, &value[&k], &value_as_i64,&to_i32);},
                    "int64" => {result = parse_array(py, &value[&k], &value_as_i64,&to_i64);},
                    _ => return Err(PyErr::new::<PyValueError, _>(format!("{:?} type not supported", v)))
                }
                match result {
                    Ok(obj) => {out.insert(k, OutStructure::Value(obj));},
                    Err(err) => return Err(err)

                }
            }
        }
        else if let Structure::Map(m) = v {
            let result =  deserialize(py, &value[&k], m);
            match result {
                Ok(mapping) => {out.insert(k, OutStructure::Map(mapping));},
                Err(err) => return Err(err)
            }
        }
        else {
            return Err(PyErr::new::<PyValueError, _>(format!("{:?} type not supported", v)))
        }
        println!("{:?}", out);
    }
    Ok(out)
}