use pyo3::exceptions::PyValueError;
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

impl<T: IntoPy<PyObject> + numpy::Element> IntoPy<PyResult<PyObject>> for Array<T> {
    fn into_py(self, py: Python) -> PyResult<PyObject> {
        match self {
            Array(Base::Scalar(val), _) => Ok(val.into_py(py)),
            Array(Base::Array(arr), shape) => {
                let expected_shape = shape.clone().unwrap();
                let n_elements = arr.len();
                ndarray::ArrayBase::from_shape_vec(shape.unwrap().into_shape(), arr).map_or(
                    Err(PyValueError::new_err(
                        format!("Irregular shape found cannot parse as {} array. Expected shape: {:?}  Total elements: {}", std::any::type_name::<T>(), expected_shape, n_elements),
                    )),
                    |arr| Ok(arr.into_pyarray(py).into_py(py)),
                )
            }
        }
    }
}

impl<T> Array<T> {
    pub fn new() -> Array<T> {
        Array(Base::Array(vec![]), Some(vec![0]))
    }

    pub fn push(&mut self, value: T) {
        match self {
            Array(Base::Array(ref mut vec), Some(ref mut shape)) => {
                vec.push(value);
                if shape.len() != 1 {
                    panic!("not working for non 1D arrays")
                };
                shape[0] += 1
            }
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
            _ => Ok(Array(
                Base::Array(values),
                Some(shape.into_iter().rev().collect()),
            )),
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

macro_rules! make_extend_visit {
    ($dtype:ty, $name:ident, $conversion:tt) => {
        #[inline]
        fn $name<E: de::Error>(self, value: $dtype) -> Result<Self::Value, E> {
            if let Some(scalar) = FromPrimitive::$conversion(value) {
                Ok(self.0.values.push(scalar))
            } else {
                Err(E::custom(format!(
                    "Could not cast {} ({}) into: {:?}",
                    value,
                    std::any::type_name::<$dtype>(),
                    std::any::type_name::<T>()
                )))
            }
        }
    };
}

struct ExtendVecVisitor<'a, T: 'a>(ArrayBuilder<'a, T>);

impl<'de, 'a, T: FromPrimitive + Deserialize<'de>> Visitor<'de> for ExtendVecVisitor<'a, T> {
    type Value = ();

    fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
        formatter.write_str("array of numbers of the same dtype")
    }

    make_extend_visit!(f32, visit_f32, from_f32);
    make_extend_visit!(f64, visit_f64, from_f64);
    make_extend_visit!(i8, visit_i8, from_i8);
    make_extend_visit!(i16, visit_i16, from_i16);
    make_extend_visit!(i32, visit_i32, from_i32);
    make_extend_visit!(i64, visit_i64, from_i64);
    make_extend_visit!(u8, visit_u8, from_u8);
    make_extend_visit!(u16, visit_u16, from_u16);
    make_extend_visit!(u32, visit_u32, from_u32);
    make_extend_visit!(u64, visit_u64, from_u64);

    fn visit_seq<S>(mut self, mut seq: S) -> Result<Self::Value, S::Error>
    where
        S: SeqAccess<'de>,
    {   
        // only compute shape of first inner array visited
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
        // we now know the shape of the following arrays at this dimension
        } else {
            while let Some(_) = seq.next_element_seed(ArrayBuilder {
                values: &mut self.0.values,
                shape: &mut self.0.shape,
                compute_shape: false,
            })? {}
            Ok(())
        }
    }
}

// The following is a seperate implementation for boolean arrays - BoolArray - because num_traits is not implemented for booleans
// The implementation should remain equivalent to the above generic implementation

#[derive(Debug, PartialEq)]
pub struct BoolArray(pub Base<bool>, pub Option<Vec<usize>>);

impl IntoPy<PyResult<PyObject>> for BoolArray {
    fn into_py(self, py: Python) -> PyResult<PyObject> {
        match self {
            BoolArray(Base::<bool>::Scalar(val), _) => Ok(val.into_py(py)),
            BoolArray(Base::<bool>::Array(arr), shape) => {
                let expected_shape = shape.clone().unwrap();
                let n_elements = arr.len();
                ndarray::ArrayBase::from_shape_vec(shape.unwrap().into_shape(), arr).map_or(
                    Err(PyValueError::new_err(
                        format!("Irregular shape found cannot parse as {} array. Expected shape: {:?}  Total elements: {}", "bool", expected_shape, n_elements),
                    )),
                    |arr| Ok(arr.into_pyarray(py).into_py(py)),
                )
            }
        }
    }
}

impl BoolArray {
    pub fn new() -> BoolArray {
        BoolArray(Base::Array(vec![]), Some(vec![0]))
    }

    pub fn push(&mut self, value: bool) {
        match self {
            BoolArray(Base::Array(ref mut vec), Some(ref mut shape)) => {
                vec.push(value);
                if shape.len() != 1 {
                    panic!("not working for non 1D arrays")
                };
                shape[0] += 1
            }
            _ => panic!("not implemented"),
        }
    }
}

impl<'de> Deserialize<'de> for BoolArray {
    fn deserialize<D>(deserializer: D) -> Result<BoolArray, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let mut values = Vec::<bool>::new();
        let mut shape = Vec::new();
        let builder = BoolArrayBuilder {
            values: &mut values,
            shape: &mut shape,
            compute_shape: true,
        };
        let visitor = BoolExtendVecVisitor(builder);
        deserializer.deserialize_any(visitor)?;
        match shape.len() {
            0 => Ok(BoolArray(Base::Scalar(values[0].clone()), None)),
            _ => Ok(BoolArray(
                Base::Array(values),
                Some(shape.into_iter().rev().collect()),
            )),
        }
    }
}

struct BoolArrayBuilder<'a> {
    values: &'a mut Vec<bool>,
    shape: &'a mut Vec<usize>,
    compute_shape: bool,
}

impl<'de, 'a> DeserializeSeed<'de> for BoolArrayBuilder<'a> {
    type Value = ();
    fn deserialize<D>(mut self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: Deserializer<'de>,
    {
        let builder = BoolArrayBuilder {
            values: &mut self.values,
            shape: &mut self.shape,
            compute_shape: self.compute_shape,
        };
        deserializer.deserialize_any(BoolExtendVecVisitor(builder))?;
        Ok(())
    }
}

struct BoolExtendVecVisitor<'a>(BoolArrayBuilder<'a>);

impl<'de, 'a> Visitor<'de> for BoolExtendVecVisitor<'a> {
    type Value = ();

    fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
        formatter.write_str("array of numbers of the same dtype")
    }

    #[inline]
    fn visit_bool<E: de::Error>(self, value: bool) -> Result<Self::Value, E> {
        Ok(self.0.values.push(value))
    }

    fn visit_seq<S>(mut self, mut seq: S) -> Result<Self::Value, S::Error>
    where
        S: SeqAccess<'de>,
    {
        if self.0.compute_shape {
            let mut outer_size: usize = 0;
            let mut compute_shape: bool = true;

            while let Some(_) = seq.next_element_seed(BoolArrayBuilder {
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
            while let Some(_) = seq.next_element_seed(BoolArrayBuilder {
                values: &mut self.0.values,
                shape: &mut self.0.shape,
                compute_shape: false,
            })? {}
            Ok(())
        }
    }
}
