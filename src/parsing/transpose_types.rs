use std::collections::HashMap;
use std::collections::HashSet;

use itertools::Itertools;
use serde::de;
use serde::de::{DeserializeSeed, Deserializer, IgnoredAny, MapAccess, SeqAccess, Visitor};

use super::errors::Error;
use crate::parsing::OutputTypes;

pub struct TransposeSeq<'s>(pub &'s mut Vec<OutputTypes>);

impl<'de, 's> DeserializeSeed<'de> for TransposeSeq<'s> {
    type Value = TransposeSeq<'s>;

    fn deserialize<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_seq(TransposeSeqVisitor(self))
    }
}

struct TransposeSeqVisitor<'s>(TransposeSeq<'s>);

impl<'de, 's> Visitor<'de> for TransposeSeqVisitor<'s> {
    type Value = TransposeSeq<'s>;

    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "TODO")
    }

    fn visit_seq<S>(self, mut seq: S) -> Result<Self::Value, S::Error>
    where
        S: SeqAccess<'de>,
    {
        let out: &mut Vec<OutputTypes> = self.0 .0;
        for (i, output_type) in out.iter_mut().enumerate() {
            let success = match output_type {
                OutputTypes::I8(arr) => seq.next_element()?.map(|new_arr| arr.push(new_arr)),
                OutputTypes::I16(arr) => seq.next_element()?.map(|new_arr| arr.push(new_arr)),
                OutputTypes::I32(arr) => seq.next_element()?.map(|new_arr| arr.push(new_arr)),
                OutputTypes::I64(arr) => seq.next_element()?.map(|new_arr| arr.push(new_arr)),

                OutputTypes::U8(arr) => seq.next_element()?.map(|new_arr| arr.push(new_arr)),
                OutputTypes::U16(arr) => seq.next_element()?.map(|new_arr| arr.push(new_arr)),
                OutputTypes::U32(arr) => seq.next_element()?.map(|new_arr| arr.push(new_arr)),
                OutputTypes::U64(arr) => seq.next_element()?.map(|new_arr| arr.push(new_arr)),

                OutputTypes::F32(arr) => seq.next_element()?.map(|new_arr| arr.push(new_arr)),
                OutputTypes::F64(arr) => seq.next_element()?.map(|new_arr| arr.push(new_arr)),

                OutputTypes::Bool(arr) => seq.next_element()?.map(|new_arr| arr.push(new_arr)),

                OutputTypes::PyList(arr) => seq.next_element()?.map(|new_arr| arr.push(new_arr)),

                _ => panic!(
                    "other variants shoudn't be able to occur because of logic in StructureVisitor"
                ),
            };
            match success {
                Some(()) => {}
                None => {
                    return Err(de::Error::custom(format!(
                        "Too many columns specified: [{}] ({}) \nFound: ({})",
                        // TODO: fix space and comma and repeated code in parsing:
                        out.iter().fold(String::new(), |agg, var| agg + var.to_string().as_str() + ", "), 
                        out.len(),
                        i
                    )))
                }
            };
        }
        while let Some(_) = seq.next_element::<IgnoredAny>()? {
            // empty any remaining items from the list with unspecified types
        }
        Ok(self.0)
    }
}

pub struct TransposeMap<'s>(pub &'s mut HashMap<String, OutputTypes>);

impl<'de, 's> DeserializeSeed<'de> for TransposeMap<'s> {
    type Value = TransposeMap<'s>;

    fn deserialize<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_map(TransposeMapVisitor(self))
    }
}

struct TransposeMapVisitor<'s>(TransposeMap<'s>);

impl<'de, 's> Visitor<'de> for TransposeMapVisitor<'s> {
    type Value = TransposeMap<'s>;

    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "TODO")
    }

    fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
    where
        A: MapAccess<'de>,
    {
        let out: &mut HashMap<String, OutputTypes> = self.0 .0;
        let n_keys = out.len();
        let mut seen_keys = HashSet::with_capacity(n_keys);
        while let Some(key) = map.next_key::<String>()? {
            if let Some(output_type) = out.get_mut(&key) {
                seen_keys.insert(key);
                match output_type {
                    OutputTypes::I8(arr) => arr.push(map.next_value()?),
                    OutputTypes::I16(arr) => arr.push(map.next_value()?),
                    OutputTypes::I32(arr) => arr.push(map.next_value()?),
                    OutputTypes::I64(arr) => arr.push(map.next_value()?),

                    OutputTypes::U8(arr) => arr.push(map.next_value()?),
                    OutputTypes::U16(arr) => arr.push(map.next_value()?),
                    OutputTypes::U32(arr) => arr.push(map.next_value()?),
                    OutputTypes::U64(arr) => arr.push(map.next_value()?),

                    OutputTypes::F32(arr) => arr.push(map.next_value()?),
                    OutputTypes::F64(arr) => arr.push(map.next_value()?),

                    OutputTypes::Bool(arr) => arr.push(map.next_value()?),

                    OutputTypes::PyList(arr) => arr.push(map.next_value()?),
                    _ => panic!(
                        "other variants shoudn't be able to occur because of logic in StructureVisitor"
                    ),
                }
            } else {
                // if the `out` map doesn't contain a key in the map (i.e. it wasn't included in the structure) we ignore it
                map.next_value::<IgnoredAny>()?;
            }
        }
        if n_keys > seen_keys.len() {
            let not_seen_keys = out
                .iter()
                .filter(|(k, _)| !seen_keys.contains(k.clone()))
                .map(|(k, _)| k.clone())
                .collect_vec();
            return Err(de::Error::custom(format!(
                "Key(s) not found: {not_seen_keys:?}"
            )));
        }
        Ok(self.0)
    }
}
