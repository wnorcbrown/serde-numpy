use numpy::ndarray::{Dim};
use numpy::{IntoPyArray, PyArray};
use pyo3::prelude::{pymodule, PyModule, PyResult, Python};
use pyo3::types::PyList;
use::pyo3::{PyErr};
use::pyo3::exceptions::{PyValueError, PyIOError};
use serde_json::{Value, Error};

mod parsing;
use parsing::{parse_float_column};
use parsing::{VectorTypes};




#[pymodule]
fn serde_numpy(_py: Python<'_>, m: &PyModule) -> PyResult<()> {


    fn _parse_float_array<'py>(py: Python<'py>, value: &Value, key: &str, index: usize) -> PyResult<&'py PyArray<f64, Dim<[usize; 1]>>> {
        if let Ok(VectorTypes::FloatArray(vector)) = parse_float_column(&value, key, index) {
            return Ok(vector.into_pyarray(py))
        }
        else {
            return Err(PyErr::new::<PyValueError, _>("JSON Key does not exist!"))
        }
    }
    
    
    #[pyfn(m)]
    #[pyo3(name = "parse_float_array")]
    fn parse_float_array<'py>(py: Python<'py>, json_str: &str, key: &str, index: usize) -> PyResult<&'py PyArray<f64, Dim<[usize; 1]>>> {

        let result = serde_json::from_str(json_str);
        
        match result {
            Ok(value) => _parse_float_array(py, &value, key, index),
            Err(_err) => return Err(PyErr::new::<PyIOError, _>("Invalid JSON"))
        }

        
    }



    Ok(())
}
