use std::collections::HashMap;


use serde_json::Value;
use serde_json::value::Index;

use pyo3::FromPyObject;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::{Python, PyErr, PyResult, PyObject};
use pyo3::types::{PyType, PyDict, PyList};

mod parse_utils;
mod parse_np_array;
mod parse_py_list;
use parse_np_array::parse_array;
use parse_py_list::parse_list;


#[derive(FromPyObject)]
#[derive(Debug)]
pub enum Structure <'py> {
    Type(&'py PyType),
    List(Vec<&'py PyType>),
    ListofList(Vec<Vec<&'py PyType>>),
    ListofMap(Vec<HashMap<String, &'py PyType>>),
    Map(HashMap<String, Structure<'py>>)
}


fn get_py_object<I: Index>(py: Python, value: &Value, type_name: &str, opt_column_selector: Option<I>) -> PyResult<PyObject> {
    match type_name {
        "str" => parse_list(py, &value, &value_as_str, opt_column_selector),
        "bool" => parse_list(py, &value, &value_as_bool, opt_column_selector),
        "int" => parse_list(py, &value, &value_as_i64, opt_column_selector),
        "float" => parse_list(py, &value, &value_as_f64, opt_column_selector),

        "float32" => parse_array(py, &value, &value_as_f64, &to_f32, opt_column_selector),
        "float64" => parse_array(py, &value, &value_as_f64,&identity, opt_column_selector),

        "int8" => parse_array(py, &value, &value_as_i64,&to_i8, opt_column_selector),
        "int16" => parse_array(py, &value, &value_as_i64,&to_i16, opt_column_selector),
        "int32" => parse_array(py, &value, &value_as_i64,&to_i32, opt_column_selector),
        "int64" => parse_array(py, &value, &value_as_i64,&identity, opt_column_selector),

        "uint8" => parse_array(py, &value, &value_as_u64,&to_u8, opt_column_selector),
        "uint16" => parse_array(py, &value, &value_as_u64,&to_u16, opt_column_selector),
        "uint32" => parse_array(py, &value, &value_as_u64,&to_u32, opt_column_selector),
        "uint64" => parse_array(py, &value, &value_as_u64,&identity, opt_column_selector),

        "bool_" => parse_array(py, &value, &value_as_bool,&identity, opt_column_selector),
        _ => return Err(PyErr::new::<PyValueError, _>(format!("{:?} type not supported", type_name)))
    }
}


pub fn deserialize<'py>(py: Python<'py>, value: &Value, structure: HashMap<String, Structure>) -> PyResult<&'py PyDict> {
    let out = PyDict::new(py);
    
    for (k, v) in structure {
        if let Structure::Type(t) = v {
            if let Ok(type_name) = t.name() {
               let result = get_py_object(py, &value[&k], type_name, None::<usize>);
                match result {
                    Ok(obj) => {out.set_item(k, obj)?;},
                    Err(err) => return Err(err)

                }
            }
        }
        else if let Structure::List(m) = v {
            let mut objects = Vec::new();
            for (i, &t) in m.iter().enumerate() {
                if let Ok(type_name) = t.name() {
                    match get_py_object(py, &value[&k][i], type_name, None::<usize>) {
                        Ok(obj) => {objects.push(obj);}
                        Err(err) => return Err(err)
                    }
                }
            }
            out.set_item(k, PyList::new(py, objects))?;
        }
        else if let Structure::ListofList(m) = v {
            let mut objects = Vec::new();
            for (i, &t) in m[0].iter().enumerate() {
                if let Ok(type_name) = t.name() {
                    match get_py_object(py, &value[&k], type_name, Some(i)) {
                        Ok(obj) => {objects.push(obj);}
                        Err(err) => return Err(err)
                    }
                }
            }
            out.set_item(k, objects)?;
        }
        else if let Structure::ListofMap(m) = v {
            let objects = PyDict::new(py);
            for (i, &t) in &m[0] {
                if let Ok(type_name) = t.name() {
                    match get_py_object(py, &value[&k], type_name, Some(&i)) {
                        Ok(obj) => {objects.set_item(i, obj)?;}
                        Err(err) => return Err(err)
                    }
                }
            }
            out.set_item(k, objects)?;
        }
        else if let Structure::Map(m) = v {
            let result =  deserialize(py, &value[&k], m);
            match result {
                Ok(mapping) => {out.set_item(k, mapping)?;},
                Err(err) => return Err(err)
            }
        }
        else {
            return Err(PyErr::new::<PyValueError, _>(format!("{:?} type not supported", v)))
        }
    }
    Ok(out)
}



/// As Types:

fn value_as_f64(value: &Value) -> Option<f64> {
    value.as_f64()
}

fn value_as_i64(value: &Value) -> Option<i64> {
    value.as_i64()
}

fn value_as_u64(value: &Value) -> Option<u64> {
    value.as_u64()
}

fn value_as_bool(value: &Value) -> Option<bool> {
    value.as_bool()
}

fn value_as_str(value: &Value) -> Option<String> {
    if let Some(slice) = value.as_str() {
        return Some(slice.to_owned())
    }
    None
}



/// Converters:

pub fn identity<T>(i: T) -> T {
    i
}

pub fn to_f32(i: f64) -> f32 {
    i as f32
}


pub fn to_i8(i: i64) -> i8 {
    i as i8
}

pub fn to_i16(i: i64) -> i16 {
    i as i16
}

pub fn to_i32(i: i64) -> i32 {
    i as i32
}

pub fn to_u8(i: u64) -> u8 {
    i as u8
}

pub fn to_u16(i: u64) -> u16 {
    i as u16
}

pub fn to_u32(i: u64) -> u32 {
    i as u32
}