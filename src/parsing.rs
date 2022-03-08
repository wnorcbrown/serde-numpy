use serde_json::value::Index;
use serde_json::Value;
use std::collections::HashMap;

use serde;
use serde::de::{DeserializeSeed, Deserializer, MapAccess, SeqAccess, Visitor};
use serde::Deserialize;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::{PyErr, PyObject, PyResult, Python, IntoPy};
use pyo3::types::{PyDict, PyList, PyType};
use pyo3::FromPyObject;

mod array_types;
mod parse_np_array;
mod parse_py_list;
mod parse_utils;
use array_types::{F32Array, I32Array, F32, I32};
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
    structure: HashMap<&'py str, PyStructure>,
) -> PyResult<&'py PyDict> {
    let out = PyDict::new(py);

    for (k, v) in structure {
        if let PyStructure::Type(t) = v {
            if let Ok(type_name) = t.name() {
                let result = get_py_object(py, &value[&k], type_name, None::<usize>);
                match result {
                    Ok(obj) => {
                        out.set_item(k, obj)?;
                    }
                    Err(err) => return Err(err),
                }
            }
        } else if let PyStructure::List(m) = v {
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
        } else if let PyStructure::ListofList(m) = v {
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
        } else if let PyStructure::ListofMap(m) = v {
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
        } else if let PyStructure::Map(m) = v {
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

#[derive(FromPyObject, Debug)]
pub enum PyStructure<'py> {
    Type(&'py PyType),
    List(Vec<&'py PyType>),
    ListofList(Vec<Vec<&'py PyType>>),
    ListofMap(Vec<HashMap<&'py str, &'py PyType>>),
    Map(HashMap<&'py str, PyStructure<'py>>),
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

// impl IntoPy<PyObject> for OutputTypes {
//     fn into_py(self, py: Python) -> PyObject {
//         match self {
//             OutputTypes::I32(v) => v.into_py(py),
//             OutputTypes::F32(v) => v.into_py(py),
//         }
//     }
// }

#[derive(Clone, Debug, Deserialize)]
pub enum InputTypes {
    #[allow(non_camel_case_types)]
    int32,
    #[allow(non_camel_case_types)]
    float32,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(untagged)]
pub enum Structure {
    Type(InputTypes),
    List(Vec<InputTypes>),
    ListofList(Vec<Vec<InputTypes>>),
    ListofMap(Vec<HashMap<String, InputTypes>>),
    Map(HashMap<String, Structure>),
}

#[derive(Debug, PartialEq)]
pub enum OutputTypes {
    I32(I32Array),
    F32(F32Array),
    List(Vec<OutputTypes>),
    Map(HashMap<String, OutputTypes>),
}

impl IntoPy<PyObject> for OutputTypes {
    fn into_py(self, py: Python) -> PyObject {
        match self {
            OutputTypes::I32(v) => v.into_py(py),
            OutputTypes::F32(v) => v.into_py(py),
            OutputTypes::List(v) => v.into_py(py),
            OutputTypes::Map(v) => v.into_py(py),
        }
    }
}


struct TransposeSeq<'s>(&'s mut Vec<OutputTypes>);

impl<'de, 's> DeserializeSeed<'de> for TransposeSeq<'s> {
    type Value = TransposeSeq<'s>;

    fn deserialize<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_seq(TransposeSeqVisitor(self))
    }
}


struct TransposeSeqVisitor<'s>(TransposeSeq<'s>);

impl<'de, 's> Visitor<'de> for TransposeSeqVisitor<'s> {
    type Value = TransposeSeq<'s>;

    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "TODO")
    }

    fn visit_seq<S>(self, mut seq: S) -> Result<Self::Value, S::Error>
    where
        S: SeqAccess<'de>,
    {
        let out: &mut Vec<OutputTypes> = self.0.0;
        for output_type in out.iter_mut() {
            match output_type {
                OutputTypes::I32(arr) => arr.push(seq.next_element::<I32Array>()?.unwrap()),
                OutputTypes::F32(arr) => arr.push(seq.next_element::<F32Array>()?.unwrap()),
                _ => panic!(
                    "other variants shoudn't be able to occur because of logic in StructureVisitor"
                ),
            }
        }
        Ok(self.0)
    }
}

struct TransposeMap<'s>(&'s mut HashMap<String, OutputTypes>);

impl<'de, 's> DeserializeSeed<'de> for TransposeMap<'s> {
    type Value = TransposeMap<'s>;

    fn deserialize<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_map(TransposeMapVisitor(self))
    }
}


struct TransposeMapVisitor<'s>(TransposeMap<'s>);

impl<'de, 's> Visitor<'de> for TransposeMapVisitor<'s> {
    type Value = TransposeMap<'s>;

    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "TODO")
    }

    fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
    where
        A: MapAccess<'de>,
    {
        let out: &mut HashMap<String, OutputTypes> = self.0.0;
        while let Some(key) = map.next_key::<String>()? {
            if let Some(output_type) = out.get_mut(&key) {
                match output_type {
                    OutputTypes::I32(arr) => arr.push(map.next_value::<I32Array>()?),
                    OutputTypes::F32(arr) => arr.push(map.next_value::<F32Array>()?),
                    _ => panic!(
                        "other variants shoudn't be able to occur because of logic in StructureVisitor"
                    ),
                }
            }
        }
        Ok(self.0)
    }
}

#[derive(Debug, Deserialize)]
#[serde(transparent)]
pub struct StructureDescriptor {
    data: Structure,
}

impl<'de> DeserializeSeed<'de> for StructureDescriptor {
    type Value = OutputTypes;

    fn deserialize<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_any(StructureVisitor(self))
    }
}

struct StructureVisitor(StructureDescriptor);

impl<'de> Visitor<'de> for StructureVisitor {
    type Value = OutputTypes;

    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "TODO")
    }

    fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
    where
        A: MapAccess<'de>,
    {
        let mut out = HashMap::new();
        let structure = self.0.data;
        if let Structure::Map(structure_map) = structure {
            while let Some(key) = map.next_key::<String>()? {
                let value = match structure_map.get(&key) {
                    Some(Structure::Type(InputTypes::int32)) => {
                        OutputTypes::I32(map.next_value::<I32Array>()?)
                    }
                    Some(Structure::Type(InputTypes::float32)) => {
                        OutputTypes::F32(map.next_value::<F32Array>()?)
                    }

                    Some(Structure::List(sub_structure_list)) => {
                        let sub_structure = StructureDescriptor {
                            data: Structure::List(sub_structure_list.clone()),
                        };
                        map.next_value_seed(sub_structure)?
                    }
                    Some(Structure::ListofList(sub_structure_lol)) => {
                        let sub_structure = StructureDescriptor {
                            data: Structure::ListofList(sub_structure_lol.clone()),
                        };
                        map.next_value_seed(sub_structure)?
                    }
                    Some(Structure::ListofMap(sub_structure_lom)) => {
                        let sub_structure = StructureDescriptor {
                            data: Structure::ListofMap(sub_structure_lom.clone()),
                        };
                        map.next_value_seed(sub_structure)?
                    }
                    Some(Structure::Map(sub_structure_map)) => {
                        // TODO get rid of clone and pass as reference
                        let sub_structure = StructureDescriptor {
                            data: Structure::Map(sub_structure_map.clone()),
                        };
                        map.next_value_seed(sub_structure)?
                    }
                    _ => panic!(""),
                };
                out.insert(key, value);
            }
            Ok(OutputTypes::Map(out))
        } else {
            panic!(""); // add correct error here
        }
    }

    fn visit_seq<S>(self, mut seq: S) -> Result<Self::Value, S::Error>
    where
        S: SeqAccess<'de>,
    {
        let structure = self.0.data;
        if let Structure::List(structure_list) = structure {
            let mut out = Vec::<OutputTypes>::new();
            for input_type in structure_list {
                match input_type {
                    InputTypes::int32 => {
                        out.push(OutputTypes::I32(seq.next_element::<I32Array>()?.unwrap()))
                    }
                    InputTypes::float32 => {
                        out.push(OutputTypes::F32(seq.next_element::<F32Array>()?.unwrap()))
                    }
                }
            }
            Ok(OutputTypes::List(out))
        } else if let Structure::ListofList(structure_lol) = structure {
            let mut out: Vec<OutputTypes> = structure_lol[0]
                .iter()
                .map(|input_type| -> OutputTypes {
                    match input_type {
                        InputTypes::int32 => OutputTypes::I32(I32Array::new()),
                        InputTypes::float32 => OutputTypes::F32(F32Array::new()),
                    }
                })
                .collect();
            let mut transpose_vecs = TransposeSeq(&mut out);
            loop {
                let next = seq.next_element_seed::<TransposeSeq>(transpose_vecs)?;
                match next {
                    Some(next) => transpose_vecs = next,
                    None => break,
                }
            }
            Ok(OutputTypes::List(out))
        } else if let Structure::ListofMap(structure_lom) = structure {
            let mut out: HashMap<String, OutputTypes> = structure_lom[0]
                .iter()
                .map(|(key, input_type)| -> (String, OutputTypes) {
                    match input_type {
                        InputTypes::int32 => (key.clone(), OutputTypes::I32(I32Array::new())),
                        InputTypes::float32 => (key.clone(), OutputTypes::F32(F32Array::new())),
                    }
                })
                .collect();
            let mut transpose_map = TransposeMap(&mut out);
            loop {
                let next = seq.next_element_seed::<TransposeMap>(transpose_map)?;
                match next {
                    Some(next) => transpose_map = next,
                    None => break,
                }
            }
            Ok(OutputTypes::Map(out))
        } else {
            panic!()
        }
    }
}

#[test]
fn test_flat() {
    let flat_structure = r#"{
        "int": "int32",
        "int_arr": "int32",
        "int_arr2D": "int32",
        "int_arr3D": "int32",
        "float": "float32",
        "float_arr": "float32"
    }"#;

    let structure_descriptor: StructureDescriptor = serde_json::from_str(flat_structure).unwrap();

    let json = r#"{
        "int": 5,
        "int_arr": [-1, 3],
        "int_arr2D": [[1, 2], [3, 4], [5, 6]],
        "int_arr3D": [[[1, 2], [3, 4], [5, 6]]],
        "float": -1.8,
        "float_arr": [6.7, 7.8]
    }"#;

    let _: Value = serde_json::from_str(json).unwrap(); // test valid json

    let out = structure_descriptor
        .deserialize(&mut serde_json::Deserializer::from_str(json))
        .unwrap();

    let expected = OutputTypes::Map(HashMap::from([
        (
            "int".to_string(),
            OutputTypes::I32(I32Array(I32::Scalar(5), None)),
        ),
        (
            "int_arr".to_string(),
            OutputTypes::I32(I32Array(I32::Array(vec![-1, 3]), Some(vec![2]))),
        ),
        (
            "int_arr2D".to_string(),
            OutputTypes::I32(I32Array(
                I32::Array(vec![1, 2, 3, 4, 5, 6]),
                Some(vec![3, 2]),
            )),
        ),
        (
            "int_arr3D".to_string(),
            OutputTypes::I32(I32Array(
                I32::Array(vec![1, 2, 3, 4, 5, 6]),
                Some(vec![1, 3, 2]),
            )),
        ),
        (
            "float".to_string(),
            OutputTypes::F32(F32Array(F32::Scalar(-1.8), None)),
        ),
        (
            "float_arr".to_string(),
            OutputTypes::F32(F32Array(F32::Array(vec![6.7, 7.8]), Some(vec![2]))),
        ),
    ]));

    assert_eq!(out, expected);
}

#[test]
fn test_nested() {
    let nested_structure = r#"{
        "int": {"int_arr2D": "int32"},
        "float": {"float_arr2D": "float32"}
    }"#;

    let structure_descriptor: StructureDescriptor = serde_json::from_str(nested_structure).unwrap();

    let json = r#"{
        "int": {"int_arr2D": [[1, 2], [3, 4], [5, 6]]},
        "float": {"float_arr2D": [6.7, 7.8]}
    }"#;

    let _: Value = serde_json::from_str(json).unwrap(); // test valid json

    let out = structure_descriptor
        .deserialize(&mut serde_json::Deserializer::from_str(json))
        .unwrap();

    let int_arr = OutputTypes::I32(I32Array(
        I32::Array(vec![1, 2, 3, 4, 5, 6]),
        Some(vec![3, 2]),
    ));

    let float_arr = OutputTypes::F32(F32Array(F32::Array(vec![6.7, 7.8]), Some(vec![2])));

    let expected = OutputTypes::Map(HashMap::from([
        (
            "int".to_string(),
            OutputTypes::Map(HashMap::from([("int_arr2D".to_string(), int_arr)])),
        ),
        (
            "float".to_string(),
            OutputTypes::Map(HashMap::from([("float_arr2D".to_string(), float_arr)])),
        ),
    ]));

    assert_eq!(out, expected);
}

#[test]
fn test_list() {
    let structure = r#"{
        "arr1": ["int32", "float32"],
        "arr2": ["float32", "float32"]
    }"#;

    let structure_descriptor: StructureDescriptor = serde_json::from_str(structure).unwrap();

    let json = r#"{
        "arr1": [[[1, 2], [3, 4], [5, 6]], [6.7, 7.8]],
        "arr2": [[3.4, 4.5], [6.7, 7.8]]
    }"#;

    let _: Value = serde_json::from_str(json).unwrap(); // test valid json

    let out = structure_descriptor
        .deserialize(&mut serde_json::Deserializer::from_str(json))
        .unwrap();

    let expected = OutputTypes::Map(HashMap::from([
        (
            "arr1".to_string(),
            OutputTypes::List(vec![
                OutputTypes::I32(I32Array(
                    I32::Array(vec![1, 2, 3, 4, 5, 6]),
                    Some(vec![3, 2]),
                )),
                OutputTypes::F32(F32Array(F32::Array(vec![6.7, 7.8]), Some(vec![2]))),
            ]),
        ),
        (
            "arr2".to_string(),
            OutputTypes::List(vec![
                OutputTypes::F32(F32Array(F32::Array(vec![3.4, 4.5]), Some(vec![2]))),
                OutputTypes::F32(F32Array(F32::Array(vec![6.7, 7.8]), Some(vec![2]))),
            ]),
        ),
    ]));

    assert_eq!(out, expected);
}

#[test]
fn test_list_of_lists() {
    let structure = r#"{
        "arr1": [["int32", "float32"]]
    }"#;

    let structure_descriptor: StructureDescriptor = serde_json::from_str(structure).unwrap();

    let json = r#"{
        "arr1": [[1, 2.1], 
                 [3, 4.3], 
                 [5, 6.5], 
                 [6, 7.8]]
    }"#;

    let _: Value = serde_json::from_str(json).unwrap(); // test valid json

    let out = structure_descriptor
        .deserialize(&mut serde_json::Deserializer::from_str(json))
        .unwrap();

    let expected = OutputTypes::Map(HashMap::from([
        (
            "arr1".to_string(),
            OutputTypes::List(vec![
                OutputTypes::I32(I32Array(
                    I32::Array(vec![1, 3, 5, 6]), 
                    Some(vec![4]))),
                OutputTypes::F32(F32Array(
                    F32::Array(vec![2.1, 4.3, 6.5, 7.8]),
                    Some(vec![4]),
                )),
            ]),
        ),
    ]));

    assert_eq!(out, expected);
}


#[test]
fn test_list_of_maps() {
    let structure = r#"{
        "arr1": [{"a": "int32", "b": "float32"}]
    }"#;

    let structure_descriptor: StructureDescriptor = serde_json::from_str(structure).unwrap();

    let json = r#"{
        "arr1": [{"a": 1, "b": 2.1}, 
                 {"a": 3, "b": 4.3}, 
                 {"a": 5, "b": 6.5}, 
                 {"a": 6, "b": 7.8}]
    }"#;

    let _: Value = serde_json::from_str(json).unwrap(); // test valid json

    let out = structure_descriptor
        .deserialize(&mut serde_json::Deserializer::from_str(json))
        .unwrap();

    let expected = OutputTypes::Map(HashMap::from([
        (
            "arr1".to_string(),
            OutputTypes::Map(
                HashMap::from([
                    ("a".to_string(), OutputTypes::I32(I32Array(I32::Array(vec![1, 3, 5, 6]), Some(vec![4])))),
                    ("b".to_string(), OutputTypes::F32(F32Array(F32::Array(vec![2.1, 4.3, 6.5, 7.8]), Some(vec![4]))))
                ])
            ),
        ),
    ]));

    assert_eq!(out, expected);
}
