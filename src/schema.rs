use anyhow::{Context, Result};
use schemars::schema_for;
use serde_json::{Value, json};

use crate::effects::{Continuation, EffectFrame, HandlerDecision};
use crate::program::{Program, ProgramFragment, ProgramPatch};

pub fn message_event_schema() -> Value {
    json!({
        "type": "object",
        "additionalProperties": false,
        "properties": {
            "event_id": { "type": "string" },
            "text": { "type": "string" }
        },
        "required": ["event_id", "text"]
    })
}

pub fn action_draft_schema() -> Value {
    json!({
        "type": "object",
        "additionalProperties": false,
        "properties": {
            "event_id": { "type": "string" },
            "kind": {
                "type": "string"
            },
            "title": { "type": "string" },
            "datetime_hint": {
                "anyOf": [{ "type": "string" }, { "type": "null" }]
            }
        },
        "required": ["event_id", "kind", "title", "datetime_hint"]
    })
}

pub fn message_events_schema() -> Value {
    json!({
        "type": "array",
        "items": message_event_schema()
    })
}

pub fn action_drafts_schema() -> Value {
    json!({
        "type": "array",
        "items": action_draft_schema()
    })
}

pub fn weak_task_result_schema(output_schema: Value) -> Value {
    json!({
        "type": "object",
        "additionalProperties": false,
        "properties": {
            "value": output_schema,
            "confidence": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 1.0
            },
            "rationale": { "type": "string" }
        },
        "required": ["value", "confidence", "rationale"]
    })
}

pub fn think_decision_schema() -> Value {
    handler_decision_schema()
}

#[derive(serde::Serialize, schemars::JsonSchema)]
struct HandlerDecisionOutput {
    handler_decision: HandlerDecision,
}

pub fn handler_decision_schema() -> Value {
    schema_value(schema_for!(HandlerDecisionOutput))
}

pub fn handler_decision_value_schema(value_schema: Value) -> Value {
    let mut schema = json!({
        "type": "object",
        "additionalProperties": false,
        "required": ["handler_decision"],
        "properties": {
            "handler_decision": {
                "type": "object",
                "additionalProperties": false,
                "required": ["decision", "value", "confidence", "rationale"],
                "properties": {
                    "decision": {
                        "type": "string",
                        "enum": ["return_value"]
                    },
                    "value": value_schema,
                    "confidence": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 1.0
                    },
                    "rationale": {
                        "type": "string"
                    }
                }
            }
        }
    });
    make_strict_structured_output_schema(&mut schema);
    schema
}

pub fn program_schema() -> Value {
    schema_value(schema_for!(Program))
}

pub fn program_fragment_schema() -> Value {
    schema_value(schema_for!(ProgramFragment))
}

pub fn program_patch_schema() -> Value {
    schema_value(schema_for!(ProgramPatch))
}

pub fn effect_frame_schema() -> Value {
    schema_value(schema_for!(EffectFrame))
}

pub fn continuation_schema() -> Value {
    schema_value(schema_for!(Continuation))
}

pub fn schema_bundle() -> Value {
    json!({
        "program": program_schema(),
        "program_fragment": program_fragment_schema(),
        "program_patch": program_patch_schema(),
        "handler_decision": handler_decision_schema(),
        "effect_frame": effect_frame_schema(),
        "continuation": continuation_schema(),
        "message_event": message_event_schema(),
        "message_events": message_events_schema(),
        "action_draft": action_draft_schema(),
        "action_drafts": action_drafts_schema(),
    })
}

pub fn validate_value(schema: &Value, value: &Value) -> Result<()> {
    let validator = jsonschema::validator_for(schema).context("invalid JSON schema")?;
    validator
        .validate(value)
        .map_err(|err| anyhow::anyhow!("model output failed schema validation: {err}"))
}

fn schema_value(schema: impl serde::Serialize) -> Value {
    let mut value = serde_json::to_value(schema).expect("schemars schema should serialize");
    make_strict_structured_output_schema(&mut value);
    value
}

fn make_strict_structured_output_schema(value: &mut Value) {
    match value {
        Value::Object(map) => {
            if map.is_empty() {
                *value = arbitrary_json_value_schema();
                return;
            }
            map.remove("$schema");
            map.remove("title");

            if let Some(one_of) = map.remove("oneOf") {
                map.insert("anyOf".to_owned(), one_of);
            }
            if let Some(const_value) = map.remove("const") {
                map.insert("enum".to_owned(), Value::Array(vec![const_value]));
            }

            let has_numeric_type = match map.get("type") {
                Some(Value::String(type_name)) => type_name == "number" || type_name == "integer",
                Some(Value::Array(type_names)) => type_names
                    .iter()
                    .any(|type_name| matches!(type_name.as_str(), Some("number" | "integer"))),
                _ => false,
            };
            if has_numeric_type {
                map.remove("format");
            }

            let property_names = map
                .get("properties")
                .and_then(Value::as_object)
                .map(|properties| properties.keys().cloned().collect::<Vec<_>>());

            let has_additional_properties = map.contains_key("additionalProperties");
            if property_names.is_some()
                || (map.get("type").and_then(Value::as_str) == Some("object")
                    && !has_additional_properties)
            {
                map.insert("additionalProperties".to_owned(), Value::Bool(false));
                if !map.contains_key("properties") {
                    map.insert("properties".to_owned(), json!({}));
                }

                map.insert(
                    "required".to_owned(),
                    Value::Array(
                        property_names
                            .unwrap_or_default()
                            .into_iter()
                            .map(Value::String)
                            .collect(),
                    ),
                );
            }

            if let Some(property_names) = map
                .get("properties")
                .and_then(Value::as_object)
                .map(|properties| properties.keys().cloned().collect::<Vec<_>>())
            {
                map.insert(
                    "required".to_owned(),
                    Value::Array(property_names.into_iter().map(Value::String).collect()),
                );
            }

            for (key, item) in map {
                if key == "properties" || key == "$defs" {
                    if let Some(object) = item.as_object_mut() {
                        for property_schema in object.values_mut() {
                            make_strict_structured_output_schema(property_schema);
                        }
                    }
                    continue;
                }
                if key == "additionalProperties" {
                    if item.as_bool() == Some(false) {
                        continue;
                    }
                    if item.as_bool() == Some(true) {
                        *item = json!({});
                        continue;
                    }
                }
                make_strict_structured_output_schema(item);
            }
        }
        Value::Array(items) => {
            for item in items {
                make_strict_structured_output_schema(item);
            }
        }
        Value::Bool(true) => {
            *value = arbitrary_json_value_schema();
        }
        Value::Bool(false) => {
            *value = json!({ "enum": [] });
        }
        Value::Null | Value::Number(_) | Value::String(_) => {}
    }
}

fn arbitrary_json_value_schema() -> Value {
    json!({
        "anyOf": [
            {
                "type": "object",
                "additionalProperties": true
            },
            {
                "type": "array",
                "items": {
                    "anyOf": [
                        { "type": "object", "additionalProperties": true },
                        {
                            "type": "array",
                            "items": {
                                "anyOf": [
                                    { "type": "object", "additionalProperties": true },
                                    { "type": "string" },
                                    { "type": "number" },
                                    { "type": "integer" },
                                    { "type": "boolean" },
                                    { "type": "null" }
                                ]
                            }
                        },
                        { "type": "string" },
                        { "type": "number" },
                        { "type": "integer" },
                        { "type": "boolean" },
                        { "type": "null" }
                    ]
                }
            },
            { "type": "string" },
            { "type": "number" },
            { "type": "integer" },
            { "type": "boolean" },
            { "type": "null" }
        ]
    })
}
