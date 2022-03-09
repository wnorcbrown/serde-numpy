use std::collections::HashMap;

use serde;
use serde::de::{DeserializeSeed, Deserializer, MapAccess, SeqAccess, Visitor};
use serde::Deserialize;

use pyo3::prelude::{IntoPy, PyErr, PyObject, Python};
use pyo3::types::PyType;
use pyo3::FromPyObject;

mod array_types;
mod transpose_types;
use array_types::Array;
use transpose_types::{TransposeMap, TransposeSeq};

#[derive(Clone, Debug, Deserialize)]
pub enum InputTypes {
    #[allow(non_camel_case_types)]
    int32,
    #[allow(non_camel_case_types)]
    float32,
}

use pyo3::PyAny;

impl<'source> FromPyObject<'source> for InputTypes {
    fn extract(object: &'source PyAny) -> Result<Self, PyErr> {
        if let Ok(pytype) = object.extract::<&'source PyType>() {
            match pytype.name() {
                Ok("int32") => Ok(InputTypes::int32),
                Ok("float32") => Ok(InputTypes::float32),
                _ => panic!("unrecognised type"),
            }
        } else if let Ok(string) = object.extract::<&'source str>() {
            match string {
                "int32" => Ok(InputTypes::int32),
                "float32" => Ok(InputTypes::float32),
                _ => panic!("unrecognised type"),
            }
        } else {
            panic!("cannot parse as InputType {:?}", object)
        }
    }
}

#[derive(Clone, FromPyObject, Debug, Deserialize)]
#[serde(untagged)]
pub enum Structure {
    ListofList(Vec<Vec<InputTypes>>),
    ListofMap(Vec<HashMap<String, InputTypes>>),
    List(Vec<InputTypes>),
    Map(HashMap<String, Structure>),
    Type(InputTypes),
}

#[derive(Debug, PartialEq)]
pub enum OutputTypes {
    I32(Array<i32>),
    F32(Array<f32>),
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

#[derive(Debug, Deserialize)]
#[serde(transparent)]
pub struct StructureDescriptor {
    pub data: Structure,
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
                        OutputTypes::I32(map.next_value()?)
                    }
                    Some(Structure::Type(InputTypes::float32)) => {
                        OutputTypes::F32(map.next_value()?)
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
                        out.push(OutputTypes::I32(seq.next_element()?.unwrap()))
                    }
                    InputTypes::float32 => {
                        out.push(OutputTypes::F32(seq.next_element()?.unwrap()))
                    }
                }
            }
            Ok(OutputTypes::List(out))
        } else if let Structure::ListofList(structure_lol) = structure {
            let mut out: Vec<OutputTypes> = structure_lol[0]
                .iter()
                .map(|input_type| -> OutputTypes {
                    match input_type {
                        InputTypes::int32 => OutputTypes::I32(Array::new()),
                        InputTypes::float32 => OutputTypes::F32(Array::new()),
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
                        InputTypes::int32 => (key.clone(), OutputTypes::I32(Array::new())),
                        InputTypes::float32 => (key.clone(), OutputTypes::F32(Array::new())),
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

#[cfg(test)]
mod tests {

    use super::*;
    use array_types::Base;
    use serde_json::Value;

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

        let structure_descriptor: StructureDescriptor =
            serde_json::from_str(flat_structure).unwrap();

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
                OutputTypes::I32(Array(Base::Scalar(5), None)),
            ),
            (
                "int_arr".to_string(),
                OutputTypes::I32(Array(Base::Array(vec![-1, 3]), Some(vec![2]))),
            ),
            (
                "int_arr2D".to_string(),
                OutputTypes::I32(Array(
                    Base::Array(vec![1, 2, 3, 4, 5, 6]),
                    Some(vec![3, 2]),
                )),
            ),
            (
                "int_arr3D".to_string(),
                OutputTypes::I32(Array(
                    Base::Array(vec![1, 2, 3, 4, 5, 6]),
                    Some(vec![1, 3, 2]),
                )),
            ),
            (
                "float".to_string(),
                OutputTypes::F32(Array(Base::Scalar(-1.8), None)),
            ),
            (
                "float_arr".to_string(),
                OutputTypes::F32(Array(Base::Array(vec![6.7, 7.8]), Some(vec![2]))),
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

        let structure_descriptor: StructureDescriptor =
            serde_json::from_str(nested_structure).unwrap();

        let json = r#"{
        "int": {"int_arr2D": [[1, 2], [3, 4], [5, 6]]},
        "float": {"float_arr2D": [6.7, 7.8]}
    }"#;

        let _: Value = serde_json::from_str(json).unwrap(); // test valid json

        let out = structure_descriptor
            .deserialize(&mut serde_json::Deserializer::from_str(json))
            .unwrap();

        let int_arr = OutputTypes::I32(Array(
            Base::Array(vec![1, 2, 3, 4, 5, 6]),
            Some(vec![3, 2]),
        ));

        let float_arr = OutputTypes::F32(Array(Base::Array(vec![6.7, 7.8]), Some(vec![2])));

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
                    OutputTypes::I32(Array(
                        Base::Array(vec![1, 2, 3, 4, 5, 6]),
                        Some(vec![3, 2]),
                    )),
                    OutputTypes::F32(Array(Base::Array(vec![6.7, 7.8]), Some(vec![2]))),
                ]),
            ),
            (
                "arr2".to_string(),
                OutputTypes::List(vec![
                    OutputTypes::F32(Array(Base::Array(vec![3.4, 4.5]), Some(vec![2]))),
                    OutputTypes::F32(Array(Base::Array(vec![6.7, 7.8]), Some(vec![2]))),
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

        let expected = OutputTypes::Map(HashMap::from([(
            "arr1".to_string(),
            OutputTypes::List(vec![
                OutputTypes::I32(Array(Base::Array(vec![1, 3, 5, 6]), Some(vec![4]))),
                OutputTypes::F32(Array(
                    Base::Array(vec![2.1, 4.3, 6.5, 7.8]),
                    Some(vec![4]),
                )),
            ]),
        )]));

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

        let expected = OutputTypes::Map(HashMap::from([(
            "arr1".to_string(),
            OutputTypes::Map(HashMap::from([
                (
                    "a".to_string(),
                    OutputTypes::I32(Array(Base::Array(vec![1, 3, 5, 6]), Some(vec![4]))),
                ),
                (
                    "b".to_string(),
                    OutputTypes::F32(Array(
                        Base::Array(vec![2.1, 4.3, 6.5, 7.8]),
                        Some(vec![4]),
                    )),
                ),
            ])),
        )]));

        assert_eq!(out, expected);
    }
}
