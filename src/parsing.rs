use serde_json::Value;
use serde_json::Result as SerdeResult;


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
