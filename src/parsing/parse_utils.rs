use serde_json::Value;


fn get_shape_helper<I>(value: &Value, mut shape: Vec<usize>, opt_column_selector: &Option<I>) -> Vec<usize> {
    if let Some(stream) = value.as_array() {
        shape.push(stream.len());
        if let Some(_) = opt_column_selector {
            return shape
        }
        get_shape_helper(&value[0], shape, opt_column_selector)
    }
    else {
        shape
    }
}

pub fn get_shape<I>(value: &Value, opt_column_selector: &Option<I>) -> Vec<usize> {
    get_shape_helper(value, Vec::new(), opt_column_selector)
}

