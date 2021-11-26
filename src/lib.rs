use std::collections::HashMap;

use pyo3::prelude::{pymodule, PyModule, PyResult, Python};
use::pyo3::PyErr;
use::pyo3::exceptions::{PyIOError};

mod parsing;
use parsing::{parse_float_column, parse_bool_column, parse_int_column, parse_keys};
use parsing::{NumpyTypes};




#[pymodule]
fn serde_numpy(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
        
    #[pyfn(m)]
    #[pyo3(name = "parse_float_array")]
    fn parse_float_array<'py>(py: Python<'py>, json_str: &[u8], key: &str, index: usize) -> PyResult<NumpyTypes<'py>> {

        let result = serde_json::from_slice(json_str);
        
        match result {
            Ok(value) => parse_float_column(py, &value, key, index),
            Err(_err) => return Err(PyErr::new::<PyIOError, _>("Invalid JSON"))
        }

        
    }

    #[pyfn(m)]
    #[pyo3(name = "parse_bool_array")]
    fn parse_bool_array<'py>(py: Python<'py>, json_str: &[u8], key: &str, index: usize) -> PyResult<NumpyTypes<'py>> {

        let result = serde_json::from_slice(json_str);
        
        match result {
            Ok(value) => parse_bool_column(py, &value, key, index),
            Err(_err) => return Err(PyErr::new::<PyIOError, _>("Invalid JSON"))
        }

        
    }


    #[pyfn(m)]
    #[pyo3(name = "parse_int_array")]
    fn parse_int_array<'py>(py: Python<'py>, json_str: &[u8], key: &str, index: usize) -> PyResult<NumpyTypes<'py>> {

        let result = serde_json::from_slice(json_str);
        
        match result {
            Ok(value) => parse_int_column(py, &value, key, index),
            Err(_err) => return Err(PyErr::new::<PyIOError, _>("Invalid JSON"))
        }

        
    }


    #[pyfn(m)]
    #[pyo3(name = "parse_keys")]
    fn parse_keys_<'py>(py: Python<'py>, json_str: &[u8], keys: Vec<&str>, indexes: Vec<Vec<usize>>) -> PyResult<HashMap<std::string::String, Vec<NumpyTypes<'py>>>> {

        let result = serde_json::from_slice(json_str);
        
        match result {
            Ok(value) => parse_keys(py, &value, keys, indexes),
            Err(_err) => return Err(PyErr::new::<PyIOError, _>("Invalid JSON"))
        }

        
    }



    Ok(())
}
