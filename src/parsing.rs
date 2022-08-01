use std::collections::{HashMap, HashSet};
use std::fmt::{self, Display, Formatter};
use std::str::FromStr;

use itertools::Itertools;

use serde;
use serde::de;
use serde::de::{DeserializeSeed, Deserializer, IgnoredAny, MapAccess, SeqAccess, Visitor};
use serde::Deserialize;
use serde_json::Value;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::{IntoPy, PyAny, PyErr, PyObject, PyResult, Python};
use pyo3::types::PyType;
use pyo3::FromPyObject;

mod array_types;
mod python_types;
mod transpose_types;
use array_types::{Array, BoolArray};
use python_types::PythonType;
use transpose_types::{TransposeMap, TransposeSeq};

#[derive(Clone, Debug, Deserialize)]
#[allow(non_camel_case_types)]
pub enum InputTypes {
    int8,
    int16,
    int32,
    int64,

    uint8,
    uint16,
    uint32,
    uint64,

    float32,
    float64,

    bool_,

    int,
    float,
    str,
    bool,

    list,
    dict,
    any,
}

impl Display for InputTypes {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::int8 => write!(f, "np.int8"),
            Self::int16 => write!(f, "np.int16"),
            Self::int32 => write!(f, "np.int32"),
            Self::int64 => write!(f, "np.int64"),

            Self::uint8 => write!(f, "np.uint8"),
            Self::uint16 => write!(f, "np.uint16"),
            Self::uint32 => write!(f, "np.uint32"),
            Self::uint64 => write!(f, "np.uint64"),

            Self::float32 => write!(f, "np.float32"),
            Self::float64 => write!(f, "np.float64"),

            Self::bool_ => write!(f, "np.bool_"),

            Self::int => write!(f, "int"),
            Self::float => write!(f, "float"),
            Self::str => write!(f, "str"),
            Self::bool => write!(f, "bool"),

            Self::list => write!(f, "list"),
            Self::dict => write!(f, "dict"),
            Self::any => write!(f, "any"),
        }
    }
}

impl InputTypes {
    fn get_transpose_output_type(&self) -> OutputTypes {
        match self {
            &InputTypes::int8 => OutputTypes::I8(Array::new()),
            &InputTypes::int16 => OutputTypes::I16(Array::new()),
            &InputTypes::int32 => OutputTypes::I32(Array::new()),
            &InputTypes::int64 => OutputTypes::I64(Array::new()),

            &InputTypes::uint8 => OutputTypes::U8(Array::new()),
            &InputTypes::uint16 => OutputTypes::U16(Array::new()),
            &InputTypes::uint32 => OutputTypes::U32(Array::new()),
            &InputTypes::uint64 => OutputTypes::U64(Array::new()),

            &InputTypes::float32 => OutputTypes::F32(Array::new()),
            &InputTypes::float64 => OutputTypes::F64(Array::new()),

            &InputTypes::bool_ => OutputTypes::Bool(BoolArray::new()),

            &InputTypes::int => OutputTypes::PyList(Vec::new()),
            &InputTypes::float => OutputTypes::PyList(Vec::new()),
            &InputTypes::str => OutputTypes::PyList(Vec::new()),
            &InputTypes::bool => OutputTypes::PyList(Vec::new()),

            &InputTypes::list => OutputTypes::PyList(Vec::new()),
            &InputTypes::dict => OutputTypes::PyList(Vec::new()),
            &InputTypes::any => OutputTypes::PyList(Vec::new()),
        }
    }
}

impl FromStr for InputTypes {
    type Err = PyErr;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "int8" => Ok(InputTypes::int8),
            "int16" => Ok(InputTypes::int16),
            "int32" => Ok(InputTypes::int32),
            "int64" => Ok(InputTypes::int64),

            "uint8" => Ok(InputTypes::uint8),
            "uint16" => Ok(InputTypes::uint16),
            "uint32" => Ok(InputTypes::uint32),
            "uint64" => Ok(InputTypes::uint16),

            "float32" => Ok(InputTypes::float32),
            "float64" => Ok(InputTypes::float64),

            "bool_" => Ok(InputTypes::bool_),

            "int" => Ok(InputTypes::int),
            "float" => Ok(InputTypes::float),
            "str" => Ok(InputTypes::str),
            "bool" => Ok(InputTypes::bool),

            "list" => Ok(InputTypes::list),
            "dict" => Ok(InputTypes::dict),
            "any" => Ok(InputTypes::any),
            _ => Err(PyValueError::new_err(format!("unrecognised type {}", s))),
        }
    }
}

impl<'source> FromPyObject<'source> for InputTypes {
    fn extract(object: &'source PyAny) -> Result<Self, PyErr> {
        if let Ok(pytype) = object.extract::<&'source PyType>() {
            match pytype.name() {
                Ok(string) => InputTypes::from_str(string),
                Err(err) => Err(err),
            }
        } else if let Ok(string) = object.extract::<&'source str>() {
            InputTypes::from_str(string)
        } else {
            Err(PyValueError::new_err(format!(
                "cannot parse {} as numpy type",
                object
            )))
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
    I8(Array<i8>),
    I16(Array<i16>),
    I32(Array<i32>),
    I64(Array<i64>),

    U8(Array<u8>),
    U16(Array<u16>),
    U32(Array<u32>),
    U64(Array<u64>),

    F32(Array<f32>),
    F64(Array<f64>),

    Bool(BoolArray),

    PythonType(PythonType),
    PyList(Vec<PythonType>),

    List(Vec<OutputTypes>),
    Map(HashMap<String, OutputTypes>),
}

impl Display for OutputTypes {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::I8(_) => write!(f, "np.int8"),
            Self::I16(_) => write!(f, "np.int16"),
            Self::I32(_) => write!(f, "np.int32"),
            Self::I64(_) => write!(f, "np.int64"),

            Self::U8(_) => write!(f, "np.uint8"),
            Self::U16(_) => write!(f, "np.uint16"),
            Self::U32(_) => write!(f, "np.uint32"),
            Self::U64(_) => write!(f, "np.uint64"),

            Self::F32(_) => write!(f, "np.float32"),
            Self::F64(_) => write!(f, "np.float64"),

            Self::Bool(_) => write!(f, "np.bool_"),

            Self::PythonType(_) => write!(f, "Any"),
            Self::PyList(_) => write!(f, "List"),

            Self::List(vec) => write!(
                f,
                "[{}]",
                vec.iter().fold(String::new(), |agg, var| agg
                    + var.to_string().as_str()
                    + ", ")
            ),
            Self::Map(_) => write!(f, "Dict"),
        }
    }
}

impl IntoPy<PyResult<PyObject>> for OutputTypes {
    fn into_py(self, py: Python) -> PyResult<PyObject> {
        match self {
            OutputTypes::I8(v) => v.into_py(py),
            OutputTypes::I16(v) => v.into_py(py),
            OutputTypes::I32(v) => v.into_py(py),
            OutputTypes::I64(v) => v.into_py(py),

            OutputTypes::U8(v) => v.into_py(py),
            OutputTypes::U16(v) => v.into_py(py),
            OutputTypes::U32(v) => v.into_py(py),
            OutputTypes::U64(v) => v.into_py(py),

            OutputTypes::F32(v) => v.into_py(py),
            OutputTypes::F64(v) => v.into_py(py),

            OutputTypes::Bool(v) => v.into_py(py),

            OutputTypes::PythonType(v) => Ok(v.into_py(py)),
            OutputTypes::PyList(v) => Ok(v.into_py(py)),

            OutputTypes::List(v) => {
                // consider using try_collect here instead of this:
                let mut out = Vec::with_capacity(v.len());
                for x in v {
                    out.push(x.into_py(py)?)
                }
                Ok(out.into_py(py))
            }
            OutputTypes::Map(v) => {
                let mut out = HashMap::with_capacity(v.len());
                for (k, x) in v {
                    out.insert(k, x.into_py(py)?);
                }
                Ok(out.into_py(py))
            }
        }
    }
}

#[derive(Debug, Deserialize, Clone)]
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
        match structure {
            Structure::Map(structure_map) => {
                let n_keys = structure_map.len();
                let mut seen_keys = HashSet::with_capacity(n_keys);
                while let Some(key) = map.next_key::<String>()? {
                    let value = match structure_map.get(&key) {
                        Some(Structure::Type(input_type)) => match input_type {
                            InputTypes::int8 => OutputTypes::I8(map.next_value()?),
                            InputTypes::int16 => OutputTypes::I16(map.next_value()?),
                            InputTypes::int32 => OutputTypes::I32(map.next_value()?),
                            InputTypes::int64 => OutputTypes::I64(map.next_value()?),

                            InputTypes::uint8 => OutputTypes::U8(map.next_value()?),
                            InputTypes::uint16 => OutputTypes::U16(map.next_value()?),
                            InputTypes::uint32 => OutputTypes::U32(map.next_value()?),
                            InputTypes::uint64 => OutputTypes::U64(map.next_value()?),

                            InputTypes::float32 => OutputTypes::F32(map.next_value()?),
                            InputTypes::float64 => OutputTypes::F64(map.next_value()?),

                            InputTypes::bool_ => OutputTypes::Bool(map.next_value()?),

                            InputTypes::int => {
                                OutputTypes::PythonType(PythonType(Value::Number(map.next_value().map_err(|_| serde::de::Error::custom("Could not deserialize as int"))?)))
                            }
                            InputTypes::float => {
                                OutputTypes::PythonType(PythonType(Value::Number(map.next_value().map_err(|_| serde::de::Error::custom("Could not deserialize as float"))?)))
                            }
                            InputTypes::bool => {
                                OutputTypes::PythonType(PythonType(Value::Bool(map.next_value().map_err(|_| serde::de::Error::custom("Could not deserialize as bool"))?)))
                            }
                            InputTypes::str => {
                                OutputTypes::PythonType(PythonType(Value::String(map.next_value().map_err(|_| serde::de::Error::custom("Could not deserialize as str"))?)))
                            }
                            _ => OutputTypes::PythonType(map.next_value()?),
                        },
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
                        None => {
                            map.next_value::<serde::de::IgnoredAny>()?;
                            continue;
                        }
                    };

                    seen_keys.insert(key.clone());
                    out.insert(key, value);
                }
                if n_keys > seen_keys.len() {
                    let not_seen_keys = structure_map
                        .iter()
                        .filter(|(k, _)| !seen_keys.contains(k.clone()))
                        .map(|(k, _)| k.clone())
                        .collect_vec();
                    return Err(de::Error::custom(format!(
                        "Key(s) not found: {not_seen_keys:?}"
                    )));
                }
                Ok(OutputTypes::Map(out))
            }
            Structure::List(list) => Err(de::Error::custom(format!(
                "Cannot deserialize map as sequence of arrays: {:?}. Try using a dictionary instead",
                list
            ))),
            Structure::ListofList(lol) => Err(de::Error::custom(format!(
                "Cannot deserialize map as transposed sequence of arrays: {:?}. Try using a dictionary instead",
                lol
            ))),
            Structure::ListofMap(lom) => Err(de::Error::custom(format!(
                "Cannot deserialize map as transposed sequence of maps: {:?}. Try using a dictionary instead",
                lom
            ))),
            Structure::Type(t) => Err(de::Error::custom(format!(
                "Cannot deserialize map as type: {:?}. Try using a dictionary instead",
                t
            )))
        }
    }

    fn visit_seq<S>(self, mut seq: S) -> Result<Self::Value, S::Error>
    where
        S: SeqAccess<'de>,
    {
        use serde::de::Error;
        let structure = self.0.data;
        match structure {
            Structure::List(structure_list) => {
                let mut out = Vec::<OutputTypes>::new();
                for (i, input_type) in structure_list.iter().enumerate() {
                    let output_type = match input_type {
                        InputTypes::int8 => seq.next_element()?.map(|arr| OutputTypes::I8(arr)),
                        InputTypes::int16 => seq.next_element()?.map(|arr| OutputTypes::I16(arr)),
                        InputTypes::int32 => seq.next_element()?.map(|arr| OutputTypes::I32(arr)),
                        InputTypes::int64 => seq.next_element()?.map(|arr| OutputTypes::I64(arr)),

                        InputTypes::uint8 => seq.next_element()?.map(|arr| OutputTypes::U8(arr)),
                        InputTypes::uint16 => seq.next_element()?.map(|arr| OutputTypes::U16(arr)),
                        InputTypes::uint32 => seq.next_element()?.map(|arr| OutputTypes::U32(arr)),
                        InputTypes::uint64 => seq.next_element()?.map(|arr| OutputTypes::U64(arr)),

                        InputTypes::float32 => seq.next_element()?.map(|arr| OutputTypes::F32(arr)),
                        InputTypes::float64 => seq.next_element()?.map(|arr| OutputTypes::F64(arr)),

                        InputTypes::bool_ => seq.next_element()?.map(|arr| OutputTypes::Bool(arr)),

                        _ => seq.next_element()?.map(|arr| OutputTypes::PythonType(arr)),
                    };
                    match output_type {
                        Some(output_type) => out.push(output_type),
                        None => {
                            return Err(S::Error::custom(format!(
                                "Too many columns specified: [{}] ({}) \nFound: ({})",
                                structure_list.iter().fold(String::new(), |agg, var| agg
                                    + var.to_string().as_str()
                                    + ", "),
                                structure_list.len(),
                                i
                            )))
                        }
                    };
                }
                while let Some(_) = seq.next_element::<IgnoredAny>()? {
                    // empty any remaining items from the list with unspecified types
                }
                Ok(OutputTypes::List(out))
            }
            Structure::ListofList(structure_lol) => {
                let mut out: Vec<OutputTypes> = structure_lol[0]
                    .iter()
                    .map(|input_type| -> OutputTypes { input_type.get_transpose_output_type() })
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
            }
            Structure::ListofMap(structure_lom) => {
                let mut out: HashMap<String, OutputTypes> = structure_lom[0]
                    .iter()
                    .map(|(key, input_type)| -> (String, OutputTypes) {
                        (key.clone(), input_type.get_transpose_output_type())
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
            }
            Structure::Map(map) => Err(de::Error::custom(format!(
                "Cannot deserialize sequence as map of arrays: {:?}. Try using a list instead",
                map
            ))),
            Structure::Type(t) => Err(de::Error::custom(format!(
                "Cannot deserialize sequence as type: {:?}. Try using a list instead",
                t
            ))),
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
                OutputTypes::I32(Array(Base::Array(vec![1, 2, 3, 4, 5, 6]), Some(vec![3, 2]))),
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

        let int_arr =
            OutputTypes::I32(Array(Base::Array(vec![1, 2, 3, 4, 5, 6]), Some(vec![3, 2])));

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
                    OutputTypes::I32(Array(Base::Array(vec![1, 2, 3, 4, 5, 6]), Some(vec![3, 2]))),
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
                OutputTypes::F32(Array(Base::Array(vec![2.1, 4.3, 6.5, 7.8]), Some(vec![4]))),
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
                    OutputTypes::F32(Array(Base::Array(vec![2.1, 4.3, 6.5, 7.8]), Some(vec![4]))),
                ),
            ])),
        )]));

        assert_eq!(out, expected);
    }
}
