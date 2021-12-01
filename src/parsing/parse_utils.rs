use serde_json::Value;


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


