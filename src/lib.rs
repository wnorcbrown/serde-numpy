use std::collections::HashMap;

use pyo3::prelude::{pymodule, PyModule, PyResult, Python};
use::pyo3::PyErr;
use::pyo3::exceptions::{PyIOError};
use pyo3::types::PyDict;

mod parsing;
use parsing::Structure;



#[pymodule]
fn serde_numpy(_py: Python<'_>, m: &PyModule) -> PyResult<()> {

    #[pyfn(m)]
    #[pyo3(name = "deserialize")]
    fn deserialize<'py>(py: Python<'py>, json_str: &[u8], structure: HashMap<&'py str, Structure>) -> PyResult<&'py PyDict> {

        let result = serde_json::from_slice(json_str);
        
        match result {
            Ok(value) => parsing::deserialize(py, &value, structure),
            Err(_err) => return Err(PyErr::new::<PyIOError, _>("Invalid JSON"))
        }

        
    }



    Ok(())
}
