use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::store::state_dir::{stable_hash_bytes, stable_hash_value};
use crate::trace::TraceEvent;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct FailureFingerprint {
    pub fingerprint_id: String,
    pub program_id: String,
    pub program_version: String,
    pub function: String,
    pub pc: usize,
    pub failed_effect_kind: String,
    pub failed_task_name: Option<String>,
    pub expected_schema_hash: String,
    pub observation_shape_hash: String,
}

pub fn fingerprint_from_capture_event(event: &TraceEvent) -> FailureFingerprint {
    let program_id = string_detail(event, "program_id").unwrap_or("unknown");
    let function = string_detail(event, "function").unwrap_or("unknown");
    let pc = event.detail.get("pc").and_then(Value::as_u64).unwrap_or(0) as usize;
    let failed_effect_kind = string_detail(event, "failed_effect_kind").unwrap_or("unknown");
    let failed_task_name = event
        .detail
        .get("failed_task_name")
        .and_then(Value::as_str)
        .map(ToOwned::to_owned);
    let expected_schema_hash = event
        .detail
        .get("expected_schema")
        .and_then(|value| stable_hash_value(value).ok())
        .unwrap_or_else(|| stable_hash_bytes(b"unknown-schema"));
    let observation_shape_hash = event
        .detail
        .get("observations")
        .map(shape_only)
        .and_then(|value| stable_hash_value(&value).ok())
        .unwrap_or_else(|| stable_hash_bytes(b"unknown-observations"));
    let raw_id = json!({
        "program_id": program_id,
        "function": function,
        "pc": pc,
        "failed_effect_kind": failed_effect_kind,
        "failed_task_name": failed_task_name,
        "expected_schema_hash": expected_schema_hash,
        "observation_shape_hash": observation_shape_hash,
    });
    let fingerprint_id =
        stable_hash_value(&raw_id).unwrap_or_else(|_| stable_hash_bytes(b"fingerprint"));

    FailureFingerprint {
        fingerprint_id,
        program_id: program_id.to_owned(),
        program_version: string_detail(event, "program_version")
            .unwrap_or("unknown")
            .to_owned(),
        function: function.to_owned(),
        pc,
        failed_effect_kind: failed_effect_kind.to_owned(),
        failed_task_name,
        expected_schema_hash,
        observation_shape_hash,
    }
}

pub fn shape_only(value: &Value) -> Value {
    match value {
        Value::Object(map) => {
            let mut shaped = serde_json::Map::new();
            for (key, value) in map {
                shaped.insert(key.clone(), shape_only(value));
            }
            Value::Object(shaped)
        }
        Value::Array(values) => json!({
            "array_len_bucket": array_len_bucket(values.len()),
            "items": values.first().map(shape_only),
        }),
        Value::String(value) => json!({
            "type": "string",
            "len_bucket": string_len_bucket(value.len()),
        }),
        Value::Number(_) => json!({ "type": "number" }),
        Value::Bool(value) => json!({ "type": "bool", "value": value }),
        Value::Null => json!({ "type": "null" }),
    }
}

fn string_detail<'a>(event: &'a TraceEvent, key: &str) -> Option<&'a str> {
    event.detail.get(key).and_then(Value::as_str)
}

fn array_len_bucket(len: usize) -> &'static str {
    match len {
        0 => "0",
        1 => "1",
        2..=8 => "2_8",
        9..=64 => "9_64",
        _ => "65_plus",
    }
}

fn string_len_bucket(len: usize) -> &'static str {
    match len {
        0 => "0",
        1..=32 => "1_32",
        33..=256 => "33_256",
        _ => "257_plus",
    }
}
