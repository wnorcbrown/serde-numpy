
use serde_json::Value;
use ndarray::{ArrayBase, OwnedRepr, Dim};
use numpy::IntoPyArray;
use pyo3::{Python, IntoPy, PyObject};
use pyo3::conversion::AsPyPointer;

pub enum ReturnTypes {
    String(String),
    Bool(bool),
    Int(i64),
    Float(f64),

    Float32Array1D(Vec<f32>),
    Float32Array2D(ArrayBase<OwnedRepr<f32>, Dim<[usize; 2]>>),
    Float64Array1D(Vec<f64>),
    Float64Array2D(ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>>),

    Int32Array1D(Vec<i32>),
    Int32Array2D(ArrayBase<OwnedRepr<i32>, Dim<[usize; 2]>>),
    Int64Array1D(Vec<i64>),
    Int64Array2D(ArrayBase<OwnedRepr<i64>, Dim<[usize; 2]>>),
}

impl IntoPy<PyObject> for ReturnTypes {
    fn into_py(self, py: Python<'_>) -> PyObject {
        match self {
            ReturnTypes::String(value) => value.into_py(py),
            ReturnTypes::Bool(value) => value.into_py(py),
            ReturnTypes::Int(value) => value.into_py(py),
            ReturnTypes::Float(value) => value.into_py(py),

            ReturnTypes::Float32Array1D(value) => unsafe { PyObject::from_borrowed_ptr(py, value.into_pyarray(py).as_ptr())},
            ReturnTypes::Float32Array2D(value) => unsafe { PyObject::from_borrowed_ptr(py, value.into_pyarray(py).as_ptr())},
            ReturnTypes::Float64Array1D(value) => unsafe { PyObject::from_borrowed_ptr(py, value.into_pyarray(py).as_ptr())},
            ReturnTypes::Float64Array2D(value) => unsafe { PyObject::from_borrowed_ptr(py, value.into_pyarray(py).as_ptr())},

            ReturnTypes::Int32Array1D(value) => unsafe { PyObject::from_borrowed_ptr(py, value.into_pyarray(py).as_ptr())},
            ReturnTypes::Int32Array2D(value) => unsafe { PyObject::from_borrowed_ptr(py, value.into_pyarray(py).as_ptr())},
            ReturnTypes::Int64Array1D(value) => unsafe { PyObject::from_borrowed_ptr(py, value.into_pyarray(py).as_ptr())},
            ReturnTypes::Int64Array2D(value) => unsafe { PyObject::from_borrowed_ptr(py, value.into_pyarray(py).as_ptr())},
        }
    }
}


fn get_shape_helper(value: &Value, mut shape: Vec<usize>) -> Vec<usize> {
    if let Some(stream) = value.as_array() {
        shape.push(stream.len());
        get_shape_helper(&value[0], shape)
    }
    else {
        shape
    }
}

pub fn get_shape(value: &Value) -> Vec<usize> {
    get_shape_helper(value, Vec::new())
}


