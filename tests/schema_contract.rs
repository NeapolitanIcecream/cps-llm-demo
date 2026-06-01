use cps_llm_demo::models::CLASSIFIER_INSTRUCTIONS;
use cps_llm_demo::schema::{
    schema_bundle, think_decision_schema, validate_value, weak_intent_guess_schema,
};
use serde_json::{Value, json};

#[test]
fn schema_bundle_contains_core_contracts() {
    let bundle = schema_bundle();
    assert!(bundle.get("weak_intent_guess").is_some());
    assert!(bundle.get("think_decision").is_some());
    assert!(bundle.get("effect_frame").is_some());
    assert!(bundle.get("continuation").is_some());
}

#[test]
fn weak_classifier_context_uses_proposal_contract_names() {
    let request_context = json!({
        "instructions": CLASSIFIER_INSTRUCTIONS,
        "text": {
            "format": {
                "name": "weak_intent_guess",
                "schema": schema_bundle()["weak_intent_guess"],
            }
        }
    });

    let body = serde_json::to_string(&request_context).unwrap();
    assert!(body.contains("WEAK semantic classifier"));
    assert!(body.contains("WeakIntentGuess"));
    assert!(body.contains("weak_intent_guess"));
}

#[test]
fn live_structured_output_schemas_follow_openai_strict_subset() {
    for (name, schema) in [
        ("weak_intent_guess", weak_intent_guess_schema()),
        ("think_decision", think_decision_schema()),
    ] {
        assert_eq!(
            schema.get("type").and_then(Value::as_str),
            Some("object"),
            "{name} root must be an object"
        );
        assert_eq!(
            schema.get("additionalProperties").and_then(Value::as_bool),
            Some(false),
            "{name} root must disallow extra properties"
        );
        assert!(
            schema.get("oneOf").is_none(),
            "{name} must not use a root oneOf"
        );
        assert!(
            schema.get("anyOf").is_none(),
            "{name} must not use a root anyOf"
        );
        assert_no_one_of(&schema, name);
        assert_no_numeric_format(&schema, name);
        assert_objects_disallow_extra_properties(&schema, name);
    }
}

#[test]
fn think_decision_schema_binds_decision_to_payload_shape() {
    let schema = think_decision_schema();

    let value_decision = json!({
        "think_decision": {
            "decision": "value",
            "data": {
                "kind": "create_task",
                "title": "发送新版 proposal",
                "datetime_hint": "明天 10 点前",
                "confidence": 0.88,
                "source": "strong_think"
            }
        }
    });
    validate_value(&schema, &value_decision).unwrap();

    let abort_decision = json!({
        "think_decision": {
            "decision": "abort",
            "data": {
                "reason": "underspecified"
            }
        }
    });
    validate_value(&schema, &abort_decision).unwrap();

    let mismatched_decision = json!({
        "think_decision": {
            "decision": "abort",
            "data": {
                "kind": "create_task",
                "title": "bad",
                "datetime_hint": null,
                "confidence": 0.88,
                "source": "strong_think"
            }
        }
    });
    assert!(validate_value(&schema, &mismatched_decision).is_err());
}

#[test]
fn weak_classifier_schema_rejects_confidence_outside_probability_range() {
    let schema = weak_intent_guess_schema();
    let valid_guess = json!({
        "kind": "create_task",
        "title": "Send proposal",
        "datetime_hint": null,
        "confidence": 1.0,
        "rationale": "clear request"
    });
    validate_value(&schema, &valid_guess).unwrap();

    for confidence in [-0.01, 1.01, 75.0] {
        let invalid_guess = json!({
            "kind": "create_task",
            "title": "Send proposal",
            "datetime_hint": null,
            "confidence": confidence,
            "rationale": "clear request"
        });

        assert!(
            validate_value(&schema, &invalid_guess).is_err(),
            "confidence {confidence} must be rejected"
        );
    }
}

fn assert_no_one_of(value: &Value, path: &str) {
    match value {
        Value::Object(map) => {
            assert!(
                !map.contains_key("oneOf"),
                "schema must not use unsupported oneOf at {path}"
            );
            for (key, child) in map {
                assert_no_one_of(child, &format!("{path}.{key}"));
            }
        }
        Value::Array(items) => {
            for (index, child) in items.iter().enumerate() {
                assert_no_one_of(child, &format!("{path}[{index}]"));
            }
        }
        Value::Null | Value::Bool(_) | Value::Number(_) | Value::String(_) => {}
    }
}

fn assert_no_numeric_format(value: &Value, path: &str) {
    match value {
        Value::Object(map) => {
            let has_numeric_type = match map.get("type") {
                Some(Value::String(type_name)) => type_name == "number" || type_name == "integer",
                Some(Value::Array(type_names)) => type_names
                    .iter()
                    .any(|type_name| matches!(type_name.as_str(), Some("number" | "integer"))),
                _ => false,
            };
            assert!(
                !(has_numeric_type && map.contains_key("format")),
                "numeric schema must not use unsupported format at {path}"
            );

            for (key, child) in map {
                assert_no_numeric_format(child, &format!("{path}.{key}"));
            }
        }
        Value::Array(items) => {
            for (index, child) in items.iter().enumerate() {
                assert_no_numeric_format(child, &format!("{path}[{index}]"));
            }
        }
        Value::Null | Value::Bool(_) | Value::Number(_) | Value::String(_) => {}
    }
}

fn assert_objects_disallow_extra_properties(value: &Value, path: &str) {
    match value {
        Value::Object(map) => {
            if map.get("type").and_then(Value::as_str) == Some("object")
                || map.contains_key("properties")
            {
                assert_eq!(
                    map.get("additionalProperties").and_then(Value::as_bool),
                    Some(false),
                    "object schema must set additionalProperties=false at {path}"
                );
            }

            for (key, child) in map {
                assert_objects_disallow_extra_properties(child, &format!("{path}.{key}"));
            }
        }
        Value::Array(items) => {
            for (index, child) in items.iter().enumerate() {
                assert_objects_disallow_extra_properties(child, &format!("{path}[{index}]"));
            }
        }
        Value::Null | Value::Bool(_) | Value::Number(_) | Value::String(_) => {}
    }
}
