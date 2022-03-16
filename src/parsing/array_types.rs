use std::marker::PhantomData;
use std::fmt;

use serde::de;
use serde::de::{SeqAccess, Visitor, DeserializeSeed, Deserializer};
use serde::Deserialize;

use num_traits::cast::FromPrimitive;
use ndarray::ShapeBuilder;
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

impl<'de, T: FromPrimitive + Deserialize<'de>> Deserialize<'de> for Array<T> {
    fn deserialize<D>(deserializer: D) -> Result<Array<T>, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        deserializer.deserialize_any(BaseVisitor::<T>(PhantomData))
    }
}



struct BaseVisitor<T>(PhantomData<T>);

impl<'de, T: FromPrimitive + Deserialize<'de>> Visitor<'de> for BaseVisitor<T> 
{
    type Value = Array<T>;
    

    fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
        formatter.write_str("array of numbers of the same dtype")
    }
    #[inline]
    fn visit_f32<E: de::Error>(self, value: f32) -> Result<Self::Value, E> {
        if let Some(scalar) = FromPrimitive::from_f32(value) {
            Ok(Array(Base::Scalar(scalar), None))
        }
        else {
            Err(E::custom(format!("Could not cast {} (f32) into: {:?}", value, std::any::type_name::<T>())))
        }
    }
    #[inline]
    fn visit_f64<E: de::Error>(self, value: f64) -> Result<Self::Value, E> {
        if let Some(scalar) = FromPrimitive::from_f64(value) {
            Ok(Array(Base::Scalar(scalar), None))
        }
        else {
            Err(E::custom(format!("Could not cast {} (f64) into: {:?}", value, std::any::type_name::<T>())))
        }
    }
    #[inline]
    fn visit_i8<E: de::Error>(self, value: i8) -> Result<Self::Value, E> {
        if let Some(scalar) = FromPrimitive::from_i8(value) {
            Ok(Array(Base::Scalar(scalar), None))
        }
        else {
            Err(E::custom(format!("Could not cast {} (i8) into: {:?}", value, std::any::type_name::<T>())))
        }
    }
    #[inline]
    fn visit_i16<E: de::Error>(self, value: i16) -> Result<Self::Value, E> {
        if let Some(scalar) = FromPrimitive::from_i16(value) {
            Ok(Array(Base::Scalar(scalar), None))
        }
        else {
            Err(E::custom(format!("Could not cast {} (i16) into: {:?}", value, std::any::type_name::<T>())))
        }
    }
    #[inline]
    fn visit_i32<E: de::Error>(self, value: i32) -> Result<Self::Value, E> {
        if let Some(scalar) = FromPrimitive::from_i32(value) {
            Ok(Array(Base::Scalar(scalar), None))
        }
        else {
            Err(E::custom(format!("Could not cast {} (i32) into: {:?}", value, std::any::type_name::<T>())))
        }
    }
    #[inline]
    fn visit_i64<E: de::Error>(self, value: i64) -> Result<Self::Value, E> {
        if let Some(scalar) = FromPrimitive::from_i64(value) {
            Ok(Array(Base::Scalar(scalar), None))
        }
        else {
            Err(E::custom(format!("Could not cast {} (i64) into: {:?}", value, std::any::type_name::<T>())))
        }
    }
    #[inline]
    fn visit_u8<E: de::Error>(self, value: u8) -> Result<Self::Value, E> {
        if let Some(scalar) = FromPrimitive::from_u8(value) {
            Ok(Array(Base::Scalar(scalar), None))
        }
        else {
            Err(E::custom(format!("Could not cast {} (u8) into: {:?}", value, std::any::type_name::<T>())))
        }
    }
    #[inline]
    fn visit_u16<E: de::Error>(self, value: u16) -> Result<Self::Value, E> {
        if let Some(scalar) = FromPrimitive::from_u16(value) {
            Ok(Array(Base::Scalar(scalar), None))
        }
        else {
            Err(E::custom(format!("Could not cast {} (u16) into: {:?}", value, std::any::type_name::<T>())))
        }
    }
    #[inline]
    fn visit_u32<E: de::Error>(self, value: u32) -> Result<Self::Value, E> {
        if let Some(scalar) = FromPrimitive::from_u32(value) {
            Ok(Array(Base::Scalar(scalar), None))
        }
        else {
            Err(E::custom(format!("Could not cast {} (u32) into: {:?}", value, std::any::type_name::<T>())))
        }
    }
    #[inline]
    fn visit_u64<E: de::Error>(self, value: u64) -> Result<Self::Value, E> {
        if let Some(scalar) = FromPrimitive::from_u64(value) {
            Ok(Array(Base::Scalar(scalar), None))
        }
        else {
            Err(E::custom(format!("Could not cast {} (u64) into: {:?}", value, std::any::type_name::<T>())))
        }
    }
    
    fn visit_seq<S>(self, mut seq: S) -> Result<Self::Value, S::Error>
    where
        S: SeqAccess<'de>,
    {
        let mut vec = Vec::<T>::new();
        // let mut dim = None;
        // let mut length = 0;
        // while let Some(elem) = seq.next_element::<Array<T>>()? {
        //     length += 1;
        //     match elem.0 {
        //         Base::Scalar(val) => vec.push(val),
        //         Base::Array(arr) => {
        //             vec.extend(arr.into_iter());
        //             if dim.is_none() {
        //                 dim = elem.1;
        //             }
        //         }
        //     }
        // }
        // let mut shape = vec![length];
        // match dim {
        //     Some(dim) => {
        //         shape.extend(dim);
        //         Ok(Array(Base::Array(vec), Some(shape)))
        //     }
        //     None => Ok(Array(Base::Array(vec), Some(shape))),
        // }

        // Each iteration through this loop is one inner array.
        let mut outer_size: usize = 0;
        while let Some(()) = seq.next_element_seed(ExtendVec(&mut vec))? {
            outer_size += 1;
        }

        // Return the finished vec.
        let shape = Some(vec![outer_size, vec.len() / outer_size]);
        Ok(Array(Base::Array(vec), shape))
    }
}


struct ExtendVec<'a, T: 'a>(&'a mut Vec<T>);

impl<'de, 'a, T> DeserializeSeed<'de> for ExtendVec<'a, T>
where
    T: Deserialize<'de>,
{
    // The return type of the `deserialize` method. This implementation
    // appends onto an existing vector but does not create any new data
    // structure, so the return type is ().
    type Value = ();

    fn deserialize<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: Deserializer<'de>,
    {
        // Visitor implementation that will walk an inner array of the JSON
        // input.
        struct ExtendVecVisitor<'a, T: 'a>(&'a mut Vec<T>);

        impl<'de, 'a, T> Visitor<'de> for ExtendVecVisitor<'a, T>
        where
            T: Deserialize<'de>,
        {
            type Value = ();

            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                write!(formatter, "an array of integers")
            }

            fn visit_seq<A>(self, mut seq: A) -> Result<(), A::Error>
            where
                A: SeqAccess<'de>,
            {
                // Visit each element in the inner array and push it onto
                // the existing vector.
                while let Some(elem) = seq.next_element()? {
                    self.0.push(elem);
                }
                Ok(())
            }
        }

        deserializer.deserialize_seq(ExtendVecVisitor(self.0))
    }
}