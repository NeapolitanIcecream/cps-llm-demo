use anyhow::{Context, Result};
use schemars::schema_for;
use serde_json::{Value, json};

use crate::effects::{Continuation, EffectFrame};
use crate::program::Program;

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
                "type": "string",
                "enum": ["ignore", "create_task", "create_calendar_event", "draft_reply"]
            },
            "title": { "type": "string" },
            "datetime_hint": {
                "anyOf": [{ "type": "string" }, { "type": "null" }]
            },
            "source": {
                "type": "string",
                "enum": ["weak_model", "strong_think"]
            }
        },
        "required": ["event_id", "kind", "title", "datetime_hint", "source"]
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
    let weak_task_spec = json!({
        "type": "object",
        "additionalProperties": false,
        "properties": {
            "name": { "type": "string" },
            "instructions": { "type": "string" }
        },
        "required": ["name", "instructions"]
    });

    let json_expr_defs = json!({
        "JsonExpr": {
            "anyOf": [
                {
                    "type": "object",
                    "additionalProperties": false,
                    "properties": {
                        "kind": { "type": "string", "enum": ["literal"] },
                        "value": {}
                    },
                    "required": ["kind", "value"]
                },
                {
                    "type": "object",
                    "additionalProperties": false,
                    "properties": {
                        "kind": { "type": "string", "enum": ["var"] },
                        "name": { "type": "string" }
                    },
                    "required": ["kind", "name"]
                },
                {
                    "type": "object",
                    "additionalProperties": false,
                    "properties": {
                        "kind": { "type": "string", "enum": ["object"] },
                        "fields": {
                            "type": "array",
                            "items": { "$ref": "#/$defs/JsonObjectField" }
                        }
                    },
                    "required": ["kind", "fields"]
                }
            ]
        },
        "JsonObjectField": {
            "type": "object",
            "additionalProperties": false,
            "properties": {
                "name": { "type": "string" },
                "value": { "$ref": "#/$defs/JsonExpr" }
            },
            "required": ["name", "value"]
        }
    });

    let resume = json!({
        "type": "object",
        "additionalProperties": false,
        "properties": {
            "decision": { "type": "string", "enum": ["resume_with_value"] },
            "value": {},
            "confidence": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 1.0
            },
            "rationale": { "type": "string" }
        },
        "required": ["decision", "value", "confidence", "rationale"]
    });

    let request_weak_probe = json!({
        "type": "object",
        "additionalProperties": false,
        "properties": {
            "decision": { "type": "string", "enum": ["request_weak_probe"] },
            "out": { "type": "string" },
            "task": weak_task_spec,
            "input": { "$ref": "#/$defs/JsonExpr" },
            "output_schema": {},
            "min_confidence": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 1.0
            },
            "rationale": { "type": "string" }
        },
        "required": [
            "decision",
            "out",
            "task",
            "input",
            "output_schema",
            "min_confidence",
            "rationale"
        ]
    });

    let abort = json!({
        "type": "object",
        "additionalProperties": false,
        "properties": {
            "decision": { "type": "string", "enum": ["abort"] },
            "reason": { "type": "string" }
        },
        "required": ["decision", "reason"]
    });

    json!({
        "type": "object",
        "additionalProperties": false,
        "$defs": json_expr_defs,
        "properties": {
            "think_decision": {
                "anyOf": [
                    resume,
                    request_weak_probe,
                    abort
                ]
            }
        },
        "required": ["think_decision"]
    })
}

pub fn program_schema() -> Value {
    schema_value(schema_for!(Program))
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
        "weak_task_result": weak_task_result_schema(json!({})),
        "think_decision": think_decision_schema(),
        "effect_frame": effect_frame_schema(),
        "continuation": continuation_schema(),
        "message_event": message_event_schema(),
        "action_draft": action_draft_schema(),
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

                if let Some(property_names) = property_names {
                    map.insert(
                        "required".to_owned(),
                        Value::Array(property_names.into_iter().map(Value::String).collect()),
                    );
                }
            }

            for (key, item) in map {
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
            *value = json!({});
        }
        Value::Bool(false) => {
            *value = json!({ "enum": [] });
        }
        Value::Null | Value::Number(_) | Value::String(_) => {}
    }
}
