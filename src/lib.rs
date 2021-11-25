use numpy::ndarray::{Dim};
use numpy::PyArray;
use pyo3::prelude::{pymodule, PyModule, PyResult, Python};
use::pyo3::PyErr;
use::pyo3::exceptions::{PyValueError, PyIOError};
use serde_json::Value;

mod parsing;
use parsing::{parse_float_column, parse_keys};
use parsing::{NumpyTypes};




#[pymodule]
fn serde_numpy(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    
    
    #[pyfn(m)]
    #[pyo3(name = "parse_float_array")]
    fn parse_float_array<'py>(py: Python<'py>, json_str: &str, key: &str, index: usize) -> PyResult<NumpyTypes<'py>> {

        let result = serde_json::from_str(json_str);
        
        match result {
            Ok(value) => parse_float_column(py, &value, key, index),
            Err(_err) => return Err(PyErr::new::<PyIOError, _>("Invalid JSON"))
        }

        
    }



    Ok(())
}
