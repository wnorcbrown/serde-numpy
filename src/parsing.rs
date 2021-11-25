use std::collections::HashMap;
use pyo3::types::IntoPyDict;
use pyo3::{IntoPy, ToPyObject};
use serde_json::Value;


#[derive(Debug)]
pub enum VectorTypes {
    IntArray(Vec<i64>),
    FloatArray(Vec<f64>),
    BoolArray(Vec<bool>),
    StringArray(Vec<String>),
}


pub fn parse_int_column(value: &Value, key: &str, index: usize) -> Result<VectorTypes, String> {
    if let Some(stream) = value[key].as_array() {
        let mut out: Vec<i64> = vec![0; stream.len()];
        for i in 0..stream.len() {
            if let Some(v) = value[key][i][index].as_i64() {
                out[i] = v;
            };
        }
        Ok(VectorTypes::IntArray(out))
    }
    else {
        Err(format!("json key {} does not exist", &key))
    }
}


pub fn parse_float_column(value: &Value, key: &str, index: usize) -> Result<VectorTypes, String> {
    if let Some(stream) = value[key].as_array() {
        let mut out: Vec<f64> = vec![0.0; stream.len()];
        for i in 0..stream.len() {
            if let Some(v) = value[key][i][index].as_f64() {
                out[i] = v;
            };
        }
        Ok(VectorTypes::FloatArray(out))
    }
    else {
        Err(format!("json key {} does not exist", &key))
    }
}


pub fn parse_bool_column(value: &Value, key: &str, index: usize) -> Result<VectorTypes, String> {
    if let Some(stream) = value[key].as_array() {
        let mut out: Vec<bool> = vec![false; stream.len()];
        for i in 0..stream.len() {
            if let Some(v) = value[key][i][index].as_bool() {
                out[i] = v;
            };
        }
        Ok(VectorTypes::BoolArray(out))
    }
    else {
        Err(format!("json key {} does not exist", &key))
    }
}


pub fn parse_str_column(value: &Value, key: &str, index: usize) -> Result<VectorTypes, String> {
    if let Some(stream) = value[key].as_array() {
        let mut out: Vec<String> = vec!["".to_string(); stream.len()];
        for i in 0..stream.len() {
            if let Some(v) = value[key][i][index].as_str() {
                out[i] = v.to_string();
            };
        }
        Ok(VectorTypes::StringArray(out))
    }
    else {
        Err(format!("json key {} does not exist", &key))
    }
}


pub fn parse_column(value: &Value, key: &str, index: usize) -> Result<VectorTypes, String> {
    let initial_value = &value[key][0][index];
    if initial_value.is_f64() {
        return parse_float_column(value, key, index)
    }
    else if initial_value.is_i64() {
        return parse_int_column(value, key, index)
    }
    else if initial_value.is_boolean() {
        return parse_bool_column(value, key, index)
    }
    else if initial_value.is_string() {
        return parse_str_column(value, key, index)
    }
    Err(format!("cannot parse column with initial type {:?}", initial_value))
}


pub fn parse_columns(value: &Value, key: &str, indexes: Vec<usize>) -> Result<Vec<VectorTypes>, String> {
    let mut out = Vec::new();
    for index in indexes {
        if let Ok(vector) = parse_column(&value, &key, index) {
            out.push(vector);
        }
        else {
            return Err(format!("cannot parse column at index {} for key {}", index, key))
        }
    }
    Ok(out)
}


pub struct ArrayMap {
    data: HashMap<String, Vec<VectorTypes>>
}

impl ArrayMap {
    fn new() -> ArrayMap {
        ArrayMap{data: HashMap::<String, Vec<VectorTypes>>::new()}
    }

}

impl ArrayMap {
    fn insert(&mut self, key: String, value: Vec<VectorTypes>) {
        self.data.insert(key, value);
    }
}

impl IntoPy for ArrayMap {

}



pub fn parse_keys<'a>(value: &Value, keys: Vec<&'a str>, indexes: Vec<Vec<usize>>) -> Result<ArrayMap, String> {
    assert_eq!(keys.len(), indexes.len(), "Number of keys: {} is not the same as number of indexes: {}", keys.len(), indexes.len());
    let mut out = ArrayMap::new();
    for (&key, indexes_) in keys.iter().zip(indexes) {
        let result = parse_columns(&value, key, indexes_);
        match result  {
            Ok(vector) => {out.insert(String::from(key), vector);},
            Err(err) => return Err(err)
        }
    }
    Ok(out)
}