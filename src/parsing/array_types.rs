use std::collections::HashMap;

use serde::de::{SeqAccess, Visitor};
use serde::Deserialize;

use ndarray::ShapeBuilder;
use numpy::IntoPyArray;
use pyo3::prelude::*;

#[derive(Debug, PartialEq)]
pub enum I32 {
    Scalar(i32),
    Array(Vec<i32>),
}

#[derive(Debug, PartialEq)]
pub struct I32Array(pub I32, pub Option<Vec<usize>>);

impl I32Array {
    pub fn new() -> I32Array {
        I32Array(I32::Array(vec![]), Some(vec![0]))
    }

    pub fn push(&mut self, arr: I32Array) {
        match arr.0 {
            I32::Scalar(value) => match self {
                I32Array(I32::Array(ref mut vec), Some(ref mut shape)) => {
                    vec.push(value);
                    if shape.len() != 1 {
                        panic!("not working for non 1D arrays")
                    };
                    shape[0] += 1
                }
                _ => panic!("not implemented"),
            },
            _ => panic!("not implemented"),
        }
    }
}

impl IntoPy<PyObject> for I32Array {
    fn into_py(self, py: Python) -> PyObject {
        match self {
            I32Array(I32::Scalar(val), _) => val.into_py(py),
            I32Array(I32::Array(arr), shape) => {
                ndarray::ArrayBase::from_shape_vec(shape.unwrap().into_shape(), arr)
                    .unwrap()
                    .into_pyarray(py)
                    .into_py(py)
            }
        }
    }
}

impl<'de> Deserialize<'de> for I32Array {
    fn deserialize<D>(deserializer: D) -> Result<I32Array, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        struct I32Visitor;

        impl<'de> Visitor<'de> for I32Visitor {
            type Value = I32Array;

            fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
                formatter.write_str("int or array of ints")
            }

            fn visit_i8<E>(self, value: i8) -> Result<Self::Value, E> {
                Ok(I32Array(I32::Scalar(value as i32), None))
            }

            fn visit_i16<E>(self, value: i16) -> Result<Self::Value, E> {
                Ok(I32Array(I32::Scalar(value as i32), None))
            }

            fn visit_i32<E>(self, value: i32) -> Result<Self::Value, E> {
                Ok(I32Array(I32::Scalar(value as i32), None))
            }

            fn visit_i64<E>(self, value: i64) -> Result<Self::Value, E> {
                Ok(I32Array(I32::Scalar(value as i32), None))
            }

            fn visit_u8<E>(self, value: u8) -> Result<Self::Value, E> {
                Ok(I32Array(I32::Scalar(value as i32), None))
            }

            fn visit_u16<E>(self, value: u16) -> Result<Self::Value, E> {
                Ok(I32Array(I32::Scalar(value as i32), None))
            }

            fn visit_u32<E>(self, value: u32) -> Result<Self::Value, E> {
                Ok(I32Array(I32::Scalar(value as i32), None))
            }

            fn visit_u64<E>(self, value: u64) -> Result<Self::Value, E> {
                Ok(I32Array(I32::Scalar(value as i32), None))
            }

            fn visit_seq<S>(self, mut seq: S) -> Result<Self::Value, S::Error>
            where
                S: SeqAccess<'de>,
            {
                let mut vec = Vec::<i32>::new();
                let mut dim = None;
                let mut length = 0;
                while let Some(elem) = seq.next_element::<I32Array>()? {
                    length += 1;
                    match elem.0 {
                        I32::Scalar(val) => vec.push(val),
                        I32::Array(arr) => {
                            vec.extend(arr.iter());
                            if dim.is_none() {
                                dim = elem.1;
                            }
                        }
                    }
                }
                let mut shape = vec![length];
                match dim {
                    Some(dim) => {
                        shape.extend(dim);
                        Ok(I32Array(I32::Array(vec), Some(shape)))
                    }
                    None => Ok(I32Array(I32::Array(vec), Some(shape))),
                }
            }
        }
        deserializer.deserialize_any(I32Visitor)
    }
}

#[derive(Debug, PartialEq)]
pub enum F32 {
    Scalar(f32),
    Array(Vec<f32>),
}

#[derive(Debug, PartialEq)]
pub struct F32Array(pub F32, pub Option<Vec<usize>>);

impl IntoPy<PyObject> for F32Array {
    fn into_py(self, py: Python) -> PyObject {
        match self {
            F32Array(F32::Scalar(val), _) => val.into_py(py),
            F32Array(F32::Array(arr), shape) => {
                ndarray::ArrayBase::from_shape_vec(shape.unwrap().into_shape(), arr)
                    .unwrap()
                    .into_pyarray(py)
                    .into_py(py)
            }
        }
    }
}

impl F32Array {
    pub fn new() -> F32Array {
        F32Array(F32::Array(vec![]), Some(vec![0]))
    }

    pub fn push(&mut self, arr: F32Array) {
        match arr.0 {
            F32::Scalar(value) => match self {
                F32Array(F32::Array(ref mut vec), Some(ref mut shape)) => {
                    vec.push(value);
                    if shape.len() != 1 {
                        panic!("not working for non 1D arrays")
                    };
                    shape[0] += 1
                }
                _ => panic!("not implemented"),
            },
            _ => panic!("not implemented"),
        }
    }
}

impl<'de> Deserialize<'de> for F32Array {
    fn deserialize<D>(deserializer: D) -> Result<F32Array, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        struct F32Visitor;

        impl<'de> Visitor<'de> for F32Visitor {
            type Value = F32Array;

            fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
                formatter.write_str("float or array of floats")
            }

            fn visit_f32<E>(self, value: f32) -> Result<Self::Value, E> {
                Ok(F32Array(F32::Scalar(value), None))
            }

            fn visit_f64<E>(self, value: f64) -> Result<Self::Value, E> {
                Ok(F32Array(F32::Scalar(value as f32), None))
            }

            fn visit_seq<S>(self, mut seq: S) -> Result<Self::Value, S::Error>
            where
                S: SeqAccess<'de>,
            {
                let mut vec = Vec::<f32>::new();
                let mut dim = None;
                let mut length = 0;
                while let Some(elem) = seq.next_element::<F32Array>()? {
                    length += 1;
                    match elem.0 {
                        F32::Scalar(val) => vec.push(val),
                        F32::Array(arr) => {
                            vec.extend(arr.iter());
                            if dim.is_none() {
                                dim = elem.1;
                            }
                        }
                    }
                }
                let mut shape = vec![length];
                match dim {
                    Some(dim) => {
                        shape.extend(dim);
                        Ok(F32Array(F32::Array(vec), Some(shape)))
                    }
                    None => Ok(F32Array(F32::Array(vec), Some(shape))),
                }
            }
        }
        deserializer.deserialize_any(F32Visitor)
    }
}

#[derive(Debug, PartialEq)]
pub enum OutputTypes {
    I32(I32Array),
    F32(F32Array),
    List(Vec<OutputTypes>),
    Map(HashMap<String, OutputTypes>),
}
