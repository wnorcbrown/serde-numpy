use serde::de;
use serde::de::{DeserializeSeed, Deserializer, SeqAccess, Visitor};
use serde::Deserialize;

use ndarray::ShapeBuilder;
use num_traits::cast::FromPrimitive;
use numpy::IntoPyArray;
use pyo3::prelude::*;

#[derive(Debug, PartialEq)]
pub enum Base<T> {
    Scalar(T),
    Array(Vec<T>),
}

#[derive(Debug, PartialEq)]
pub struct Array<T>(pub Base<T>, pub Option<Vec<usize>>);

impl<T: IntoPy<PyObject> + numpy::Element> IntoPy<PyObject> for Array<T> {
    fn into_py(self, py: Python) -> PyObject {
        match self {
            Array(Base::Scalar(val), _) => val.into_py(py),
            Array(Base::Array(arr), shape) => {
                ndarray::ArrayBase::from_shape_vec(shape.unwrap().into_shape(), arr)
                    .unwrap()
                    .into_pyarray(py)
                    .into_py(py)
            }
        }
    }
}

impl<T> Array<T> {
    pub fn new() -> Array<T> {
        Array(Base::Array(vec![]), Some(vec![0]))
    }

    pub fn push(&mut self, arr: Array<T>) {
        match arr.0 {
            Base::Scalar(value) => match self {
                Array(Base::Array(ref mut vec), Some(ref mut shape)) => {
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

impl<'de, T: FromPrimitive + Clone + Deserialize<'de>> Deserialize<'de> for Array<T> {
    fn deserialize<D>(deserializer: D) -> Result<Array<T>, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let mut values = Vec::<T>::new();
        let mut shape = Vec::new();
        let builder = ArrayBuilder {
            values: &mut values,
            shape: &mut shape,
            compute_shape: true,
        };
        let visitor = ExtendVecVisitor(builder);
        deserializer.deserialize_any(visitor)?;
        match shape.len() {
            0 => Ok(Array(Base::Scalar(values[0].clone()), None)),
            _ => Ok(Array(Base::Array(values), Some(shape.into_iter().rev().collect()))),
        }
    }
}

struct ArrayBuilder<'a, T: 'a> {
    values: &'a mut Vec<T>,
    shape: &'a mut Vec<usize>,
    compute_shape: bool,
}

impl<'de, 'a, T> DeserializeSeed<'de> for ArrayBuilder<'a, T>
where
    T: FromPrimitive + Deserialize<'de>,
{
    type Value = ();
    fn deserialize<D>(mut self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: Deserializer<'de>,
    {
        let builder = ArrayBuilder {
            values: &mut self.values,
            shape: &mut self.shape,
            compute_shape: self.compute_shape,
        };
        deserializer.deserialize_any(ExtendVecVisitor(builder))?;
        Ok(())
    }
}

struct ExtendVecVisitor<'a, T: 'a>(ArrayBuilder<'a, T>);

impl<'de, 'a, T: FromPrimitive + Deserialize<'de>> Visitor<'de> for ExtendVecVisitor<'a, T> {
    type Value = ();

    fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
        formatter.write_str("array of numbers of the same dtype")
    }

    #[inline]
    fn visit_f32<E: de::Error>(self, value: f32) -> Result<Self::Value, E> {
        if let Some(scalar) = FromPrimitive::from_f32(value) {
            Ok(self.0.values.push(scalar))
        } else {
            Err(E::custom(format!(
                "Could not cast {} (f32) into: {:?}",
                value,
                std::any::type_name::<T>()
            )))
        }
    }
    #[inline]
    fn visit_f64<E: de::Error>(self, value: f64) -> Result<Self::Value, E> {
        if let Some(scalar) = FromPrimitive::from_f64(value) {
            Ok(self.0.values.push(scalar))
        } else {
            Err(E::custom(format!(
                "Could not cast {} (f64) into: {:?}",
                value,
                std::any::type_name::<T>()
            )))
        }
    }
    #[inline]
    fn visit_i8<E: de::Error>(self, value: i8) -> Result<Self::Value, E> {
        if let Some(scalar) = FromPrimitive::from_i8(value) {
            Ok(self.0.values.push(scalar))
        } else {
            Err(E::custom(format!(
                "Could not cast {} (i8) into: {:?}",
                value,
                std::any::type_name::<T>()
            )))
        }
    }
    #[inline]
    fn visit_i16<E: de::Error>(self, value: i16) -> Result<Self::Value, E> {
        if let Some(scalar) = FromPrimitive::from_i16(value) {
            Ok(self.0.values.push(scalar))
        } else {
            Err(E::custom(format!(
                "Could not cast {} (i16) into: {:?}",
                value,
                std::any::type_name::<T>()
            )))
        }
    }
    #[inline]
    fn visit_i32<E: de::Error>(self, value: i32) -> Result<Self::Value, E> {
        if let Some(scalar) = FromPrimitive::from_i32(value) {
            Ok(self.0.values.push(scalar))
        } else {
            Err(E::custom(format!(
                "Could not cast {} (i32) into: {:?}",
                value,
                std::any::type_name::<T>()
            )))
        }
    }
    #[inline]
    fn visit_i64<E: de::Error>(self, value: i64) -> Result<Self::Value, E> {
        if let Some(scalar) = FromPrimitive::from_i64(value) {
            Ok(self.0.values.push(scalar))
        } else {
            Err(E::custom(format!(
                "Could not cast {} (i64) into: {:?}",
                value,
                std::any::type_name::<T>()
            )))
        }
    }
    #[inline]
    fn visit_u8<E: de::Error>(self, value: u8) -> Result<Self::Value, E> {
        if let Some(scalar) = FromPrimitive::from_u8(value) {
            Ok(self.0.values.push(scalar))
        } else {
            Err(E::custom(format!(
                "Could not cast {} (u8) into: {:?}",
                value,
                std::any::type_name::<T>()
            )))
        }
    }
    #[inline]
    fn visit_u16<E: de::Error>(self, value: u16) -> Result<Self::Value, E> {
        if let Some(scalar) = FromPrimitive::from_u16(value) {
            Ok(self.0.values.push(scalar))
        } else {
            Err(E::custom(format!(
                "Could not cast {} (u16) into: {:?}",
                value,
                std::any::type_name::<T>()
            )))
        }
    }
    #[inline]
    fn visit_u32<E: de::Error>(self, value: u32) -> Result<Self::Value, E> {
        if let Some(scalar) = FromPrimitive::from_u32(value) {
            Ok(self.0.values.push(scalar))
        } else {
            Err(E::custom(format!(
                "Could not cast {} (u32) into: {:?}",
                value,
                std::any::type_name::<T>()
            )))
        }
    }
    #[inline]
    fn visit_u64<E: de::Error>(self, value: u64) -> Result<Self::Value, E> {
        if let Some(scalar) = FromPrimitive::from_u64(value) {
            Ok(self.0.values.push(scalar))
        } else {
            Err(E::custom(format!(
                "Could not cast {} (u64) into: {:?}",
                value,
                std::any::type_name::<T>()
            )))
        }
    }
    
    fn visit_seq<S>(mut self, mut seq: S) -> Result<Self::Value, S::Error>
    where
        S: SeqAccess<'de>,
    {
        if self.0.compute_shape {
            let mut outer_size: usize = 0;
            let mut compute_shape: bool = true;

            while let Some(_) = seq.next_element_seed(ArrayBuilder {
                values: &mut self.0.values,
                shape: &mut self.0.shape,
                compute_shape: compute_shape,
            })? {
                outer_size += 1;
                compute_shape = false;
            }

            self.0.shape.push(outer_size);

            Ok(())
        } else {

            while let Some(_) = seq.next_element_seed(ArrayBuilder {
                values: &mut self.0.values,
                shape: &mut self.0.shape,
                compute_shape: false,
            })? {
                
            }
            Ok(())
        }
    }
}
