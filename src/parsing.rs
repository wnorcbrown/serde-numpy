use std::collections::HashMap;
use std::process::Output;

use ndarray::Array1;
use numpy::IntoPyArray;
use serde_json::value::Index;
use serde_json::Value;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::{PyErr, PyObject, PyResult, Python};
use pyo3::types::{PyDict, PyList, PyType};
use pyo3::FromPyObject;

mod parse_np_array;
mod parse_py_list;
mod parse_utils;
use parse_np_array::parse_array;
use parse_py_list::parse_list;

fn get_py_object<I: Index>(
    py: Python,
    value: &Value,
    type_name: &str,
    opt_column_selector: Option<I>,
) -> PyResult<PyObject> {
    match type_name {
        "str" => parse_list(py, &value, &value_as_str, opt_column_selector),
        "bool" => parse_list(py, &value, &value_as_bool, opt_column_selector),
        "int" => parse_list(py, &value, &value_as_i64, opt_column_selector),
        "float" => parse_list(py, &value, &value_as_f64, opt_column_selector),

        "float32" => parse_array(py, &value, &value_as_f64, &to_f32, opt_column_selector),
        "float64" => parse_array(py, &value, &value_as_f64, &identity, opt_column_selector),

        "int8" => parse_array(py, &value, &value_as_i64, &to_i8, opt_column_selector),
        "int16" => parse_array(py, &value, &value_as_i64, &to_i16, opt_column_selector),
        "int32" => parse_array(py, &value, &value_as_i64, &to_i32, opt_column_selector),
        "int64" => parse_array(py, &value, &value_as_i64, &identity, opt_column_selector),

        "uint8" => parse_array(py, &value, &value_as_u64, &to_u8, opt_column_selector),
        "uint16" => parse_array(py, &value, &value_as_u64, &to_u16, opt_column_selector),
        "uint32" => parse_array(py, &value, &value_as_u64, &to_u32, opt_column_selector),
        "uint64" => parse_array(py, &value, &value_as_u64, &identity, opt_column_selector),

        "bool_" => parse_array(py, &value, &value_as_bool, &identity, opt_column_selector),
        _ => {
            return Err(PyErr::new::<PyValueError, _>(format!(
                "{:?} type not supported",
                type_name
            )))
        }
    }
}

pub fn deserialize<'py>(
    py: Python<'py>,
    value: &Value,
    structure: HashMap<&'py str, Structure>,
) -> PyResult<&'py PyDict> {
    let out = PyDict::new(py);

    for (k, v) in structure {
        if let Structure::Type(t) = v {
            if let Ok(type_name) = t.name() {
                let result = get_py_object(py, &value[&k], type_name, None::<usize>);
                match result {
                    Ok(obj) => {
                        out.set_item(k, obj)?;
                    }
                    Err(err) => return Err(err),
                }
            }
        } else if let Structure::List(m) = v {
            let mut objects = Vec::new();
            for (i, &t) in m.iter().enumerate() {
                if let Ok(type_name) = t.name() {
                    match get_py_object(py, &value[&k][i], type_name, None::<usize>) {
                        Ok(obj) => {
                            objects.push(obj);
                        }
                        Err(err) => return Err(err),
                    }
                }
            }
            out.set_item(k, PyList::new(py, objects))?;
        } else if let Structure::ListofList(m) = v {
            let mut objects = Vec::new();
            for (i, &t) in m[0].iter().enumerate() {
                if let Ok(type_name) = t.name() {
                    match get_py_object(py, &value[&k], type_name, Some(i)) {
                        Ok(obj) => {
                            objects.push(obj);
                        }
                        Err(err) => return Err(err),
                    }
                }
            }
            out.set_item(k, objects)?;
        } else if let Structure::ListofMap(m) = v {
            let objects = PyDict::new(py);
            for (i, &t) in &m[0] {
                if let Ok(type_name) = t.name() {
                    match get_py_object(py, &value[&k], type_name, Some(&i)) {
                        Ok(obj) => {
                            objects.set_item(i, obj)?;
                        }
                        Err(err) => return Err(err),
                    }
                }
            }
            out.set_item(k, objects)?;
        } else if let Structure::Map(m) = v {
            let result = deserialize(py, &value[&k], m);
            match result {
                Ok(mapping) => {
                    out.set_item(k, mapping)?;
                }
                Err(err) => return Err(err),
            }
        } else {
            return Err(PyErr::new::<PyValueError, _>(format!(
                "{:?} type not supported",
                v
            )));
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
        return Some(slice.to_owned());
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

use pyo3::{IntoPy, PyAny, ToPyObject};
use serde;
use serde::de::{DeserializeSeed, Deserializer, MapAccess, SeqAccess, Visitor};
use serde::{Deserialize, Serialize};

#[derive(FromPyObject, Debug)]
pub enum Structure<'py> {
    Type(&'py PyType),
    List(Vec<&'py PyType>),
    ListofList(Vec<Vec<&'py PyType>>),
    ListofMap(Vec<HashMap<&'py str, &'py PyType>>),
    Map(HashMap<&'py str, Structure<'py>>),
}

// #[derive(Debug, PartialEq)]
// enum NumpyTypes {
//     Int32(PyObject),
//     Float32(PyObject),
//     Null,
// }

// impl From<&str> for NumpyTypes {
//     fn from(numpy_type: &str) -> NumpyTypes {
//         match numpy_type {
//             "int32" => NumpyTypes::
//         }
//     }
// }


#[derive(Debug, PartialEq)]
enum I32 {
    Scalar(i32),
    Array(Array1<i32>),
}

impl IntoPy<PyObject> for I32 {
    fn into_py(self, py: Python) -> PyObject {
        match self {
            I32::Scalar(val) => val.into_py(py),
            I32::Array(arr) => arr.into_pyarray(py).into_py(py)
        }
    }
}


impl<'de> Deserialize<'de> for I32 {
    fn deserialize<D>(deserializer: D) -> Result<I32, D::Error>
    where 
        D: serde::Deserializer<'de>,
    {
        struct I32Visitor;

        impl<'de> Visitor<'de> for I32Visitor {
            type Value = I32;

            fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
                formatter.write_str("int or array of ints")
            }

            fn visit_i32<E>(self, value: i32) -> Result<I32, E> {
                Ok(I32::Scalar(value))
            }

            fn visit_i64<E>(self, value: i64) -> Result<I32, E> {
                Ok(I32::Scalar(value as i32))
            }

            fn visit_u64<E>(self, value: u64) -> Result<I32, E> {
                Ok(I32::Scalar(value as i32))
            }

            fn visit_seq<S>(self, mut seq: S) -> Result<I32, S::Error>
            where
                S: SeqAccess<'de>,
            {
                let mut vec = Vec::<i32>::new();
                while let Some(elem) = seq.next_element::<i32>()? {
                    vec.push(elem)
                }
                Ok(I32::Array(vec.into()))
            }
        }
        deserializer.deserialize_any(I32Visitor)
    }
}



#[derive(Debug, PartialEq)]
enum F32 {
    Scalar(f32),
    Array(Array1<f32>),
}

impl IntoPy<PyObject> for F32 {
    fn into_py(self, py: Python) -> PyObject {
        match self {
            F32::Scalar(val) => val.into_py(py),
            F32::Array(arr) => arr.into_pyarray(py).into_py(py)
        }
    }
}


impl<'de> Deserialize<'de> for F32 {
    fn deserialize<D>(deserializer: D) -> Result<F32, D::Error>
    where 
        D: serde::Deserializer<'de>,
    {
        struct F32Visitor;

        impl<'de> Visitor<'de> for F32Visitor {
            type Value = F32;

            fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
                formatter.write_str("float or array of floats")
            }

            fn visit_f32<E>(self, value: f32) -> Result<F32, E> {
                Ok(F32::Scalar(value))
            }

            fn visit_f64<E>(self, value: f64) -> Result<F32, E> {
                Ok(F32::Scalar(value as f32))
            }

            fn visit_seq<S>(self, mut seq: S) -> Result<F32, S::Error>
            where
                S: SeqAccess<'de>,
            {
                let mut vec = Vec::<f32>::new();
                while let Some(elem) = seq.next_element::<f32>()? {
                    vec.push(elem)
                }
                Ok(F32::Array(vec.into()))
            }
        }
        deserializer.deserialize_any(F32Visitor)
    }
}


#[derive(Debug, PartialEq)]
enum OutputTypes {
    I32(I32),
    F32(F32)
}

impl IntoPy<PyObject> for OutputTypes {
    fn into_py(self, py: Python) -> PyObject {
        match self {
            OutputTypes::I32(v) => v.into_py(py),
            OutputTypes::F32(v) => v.into_py(py)
        }
    }
}


#[derive(Debug, Serialize, Deserialize)]
#[serde(transparent)]
struct StructureDescriptor {
    data: HashMap<String, String>,
}

impl<'de> DeserializeSeed<'de> for StructureDescriptor {
    type Value = HashMap<String, OutputTypes>;

    fn deserialize<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_map(StructureVisitor(self))
    }
}

struct StructureVisitor(StructureDescriptor);

impl<'de> Visitor<'de> for StructureVisitor {
    type Value = HashMap<String, OutputTypes>;

    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "TODO")
    }

    fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
    where
        A: MapAccess<'de>,
    {  
        let mut out = HashMap::new();
        while let Some(key) = map.next_key::<String>()? {
            let value = match self.0.data.get(&key).map_or("", |v| v.as_str()) {
                "int32" => OutputTypes::I32(map.next_value::<I32>()?),
                "float32" => OutputTypes::F32(map.next_value::<F32>()?),
                _ => panic!("")
            };
            out.insert(key, value);
        }
        Ok(out)
    }
}



#[test]
fn test() {
    let flat_structure = r#"{
        "int": "int32",
        "int_arr": "int32",
        "float": "float32",
        "float_arr": "float32"
    }"#;

    let structure_descriptor: StructureDescriptor = serde_json::from_str(flat_structure).unwrap();

    let json = r#"{
        "int": 5,
        "int_arr": [-1, 3],
        "float": -1.8,
        "float_arr": [6.7, 7.8]
    }"#;

    let _: Value = serde_json::from_str(json).unwrap(); // test valid json

    let out = structure_descriptor
        .deserialize(&mut serde_json::Deserializer::from_str(json))
        .unwrap();

    assert_eq!(out.get("int").unwrap(), &OutputTypes::I32(I32::Scalar(5)));
    assert_eq!(out.get("int_arr").unwrap(), &OutputTypes::I32(I32::Array(ndarray::array![-1, 3])));
    assert_eq!(out.get("float").unwrap(), &OutputTypes::F32(F32::Scalar(-1.8)));
    assert_eq!(out.get("float_arr").unwrap(), &OutputTypes::F32(F32::Array(ndarray::array![6.7, 7.8])))
}
