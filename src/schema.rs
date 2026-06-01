use anyhow::{Context, Result};
use schemars::schema_for;
use serde_json::{Value, json};

use crate::domain::WeakIntentGuess;
use crate::effects::{Continuation, EffectFrame};

pub fn weak_intent_guess_schema() -> Value {
    schema_value(schema_for!(WeakIntentGuess))
}

pub fn think_decision_schema() -> Value {
    let resolved_intent = json!({
        "type": "object",
        "additionalProperties": false,
        "properties": {
            "kind": {
                "type": "string",
                "enum": [
                    "ignore",
                    "create_task",
                    "create_calendar_event",
                    "draft_reply"
                ]
            },
            "title": {
                "type": "string"
            },
            "datetime_hint": {
                "type": ["string", "null"]
            },
            "confidence": {
                "type": "number"
            },
            "source": {
                "type": "string",
                "enum": ["strong_think"]
            }
        },
        "required": [
            "kind",
            "title",
            "datetime_hint",
            "confidence",
            "source"
        ]
    });
    let abort_reason = json!({
        "type": "object",
        "additionalProperties": false,
        "properties": {
            "reason": {
                "type": "string"
            }
        },
        "required": ["reason"]
    });

    json!({
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "ThinkDecision",
        "type": "object",
        "oneOf": [
            {
                "type": "object",
                "additionalProperties": false,
                "properties": {
                    "decision": {
                        "type": "string",
                        "enum": ["value"]
                    },
                    "data": resolved_intent
                },
                "required": ["decision", "data"]
            },
            {
                "type": "object",
                "additionalProperties": false,
                "properties": {
                    "decision": {
                        "type": "string",
                        "enum": ["abort"]
                    },
                    "data": abort_reason
                },
                "required": ["decision", "data"]
            }
        ]
    })
}

pub fn effect_frame_schema() -> Value {
    schema_value(schema_for!(EffectFrame))
}

pub fn continuation_schema() -> Value {
    schema_value(schema_for!(Continuation))
}

pub fn schema_bundle() -> Value {
    json!({
        "weak_intent_guess": weak_intent_guess_schema(),
        "think_decision": think_decision_schema(),
        "effect_frame": effect_frame_schema(),
        "continuation": continuation_schema(),
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
            let property_names = map
                .get("properties")
                .and_then(Value::as_object)
                .map(|properties| properties.keys().cloned().collect::<Vec<_>>());

            if map.get("type").and_then(Value::as_str) == Some("object") || property_names.is_some()
            {
                map.insert("additionalProperties".to_owned(), Value::Bool(false));

                if let Some(property_names) = property_names {
                    map.insert(
                        "required".to_owned(),
                        Value::Array(property_names.into_iter().map(Value::String).collect()),
                    );
                }
            }

            for item in map.values_mut() {
                make_strict_structured_output_schema(item);
            }
        }
        Value::Array(items) => {
            for item in items {
                make_strict_structured_output_schema(item);
            }
        }
        Value::Null | Value::Bool(_) | Value::Number(_) | Value::String(_) => {}
    }
}
