use std::collections::HashMap;

use serde::de::{DeserializeSeed, Deserializer, MapAccess, SeqAccess, Visitor};

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
        for output_type in out.iter_mut() {
            match output_type {
                OutputTypes::I8(arr) => arr.push(seq.next_element()?.unwrap()),
                OutputTypes::I16(arr) => arr.push(seq.next_element()?.unwrap()),
                OutputTypes::I32(arr) => arr.push(seq.next_element()?.unwrap()),
                OutputTypes::I64(arr) => arr.push(seq.next_element()?.unwrap()),

                OutputTypes::U8(arr) => arr.push(seq.next_element()?.unwrap()),
                OutputTypes::U16(arr) => arr.push(seq.next_element()?.unwrap()),
                OutputTypes::U32(arr) => arr.push(seq.next_element()?.unwrap()),
                OutputTypes::U64(arr) => arr.push(seq.next_element()?.unwrap()),

                OutputTypes::F32(arr) => arr.push(seq.next_element()?.unwrap()),
                OutputTypes::F64(arr) => arr.push(seq.next_element()?.unwrap()),

                OutputTypes::Bool(arr) => arr.push(seq.next_element()?.unwrap()),
                _ => panic!(
                    "other variants shoudn't be able to occur because of logic in StructureVisitor"
                ),
            }
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
        while let Some(key) = map.next_key::<String>()? {
            if let Some(output_type) = out.get_mut(&key) {
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
                    _ => panic!(
                        "other variants shoudn't be able to occur because of logic in StructureVisitor"
                    ),
                }
            }
        }
        Ok(self.0)
    }
}
