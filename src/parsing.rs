use std::collections::HashMap;

use numpy::IntoPyArray;
use numpy::PyArray;
use numpy::ndarray::Dim;

use pyo3::exceptions::{PyValueError, PyIOError};
use pyo3::prelude::{Python, PyErr, PyResult, IntoPy, PyObject,};
use pyo3::conversion::AsPyPointer;
use serde_json::Value;


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



pub fn parse_int_column<'py>(py: Python<'py>, value: &Value, key: &str, index: usize) -> PyResult<NumpyTypes<'py>> {
    if let Some(stream) = value[key].as_array() {
        let mut out: Vec<i64> = vec![0; stream.len()];
        for i in 0..stream.len() {
            if let Some(v) = value[key][i][index].as_i64() {
                out[i] = v;
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
        let mut out: Vec<f64> = vec![0.0; stream.len()];
        for i in 0..stream.len() {
            if let Some(v) = value[key][i][index].as_f64() {
                out[i] = v;
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
        let mut out: Vec<bool> = vec![false; stream.len()];
        for i in 0..stream.len() {
            if let Some(v) = value[key][i][index].as_bool() {
                out[i] = v;
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
//             if let Some(v) = value[key][i][index].as_str() {
//                 out[i] = v.to_string();
//             };
//         }
//         Ok(NumpyTypes::StringArray(out.into_pyarray(py)))
//     }
//     else {
//         Err(format!("json key {} does not exist", &key))
//     }
// }


pub fn parse_column<'py>(py: Python<'py>, value: &Value, key: &str, index: usize) -> PyResult<NumpyTypes<'py>> {
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


pub fn parse_columns<'py>(py: Python<'py>, value: &Value, key: &str, indexes: Vec<usize>) -> PyResult<Vec<NumpyTypes<'py>>> {
    let mut out = Vec::new();
    for index in indexes {
        if let Ok(vector) = parse_column(py, &value, &key, index) {
            out.push(vector);
        }
        else {
            return Err(PyErr::new::<PyIOError, _>(format!("cannot parse column at index {} for key {}", index, key)))
        }
    }
    Ok(out)
}


// pub struct ArrayMap {
//     data: HashMap<String, Vec<NumpyTypes>>
// }

// impl ArrayMap {
//     fn new() -> ArrayMap {
//         ArrayMap{data: HashMap::<String, Vec<NumpyTypes>>::new()}
//     }

// }

// impl ArrayMap {
//     fn insert(&mut self, key: String, value: Vec<VectorTypes>) {
//         self.data.insert(key, value);
//     }
// }


pub fn parse_keys<'py>(py: Python<'py>, value: &Value, keys: Vec<&str>, indexes: Vec<Vec<usize>>) -> PyResult<HashMap<String, Vec<NumpyTypes<'py>>>> {
    assert_eq!(keys.len(), indexes.len(), "Number of keys: {} is not the same as number of indexes: {}", keys.len(), indexes.len());
    let mut out = HashMap::new();
    for (&key, indexes_) in keys.iter().zip(indexes) {
        let result = parse_columns(py, &value, key, indexes_);
        match result  {
            Ok(vector) => {out.insert(String::from(key), vector);},
            Err(err) => return Err(err)
        }
    }
    Ok(out)
}