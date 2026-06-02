use cps_llm_demo::effects::ThinkDecision;
use cps_llm_demo::models::WEAK_TASK_INSTRUCTIONS;
use cps_llm_demo::program::JsonExpr;
use cps_llm_demo::schema::{
    action_draft_schema, program_schema, schema_bundle, think_decision_schema, validate_value,
    weak_task_result_schema,
};
use serde_json::{Value, json};

#[test]
fn schema_bundle_contains_program_runtime_contracts() {
    let bundle = schema_bundle();
    assert!(bundle.get("program").is_some());
    assert!(bundle.get("weak_task_result").is_some());
    assert!(bundle.get("think_decision").is_some());
    assert!(bundle.get("effect_frame").is_some());
    assert!(bundle.get("continuation").is_some());
    assert!(bundle.get("weak_intent_guess").is_none());
}

#[test]
fn weak_task_context_uses_program_effect_contract_names() {
    let request_context = json!({
        "instructions": WEAK_TASK_INSTRUCTIONS,
        "text": {
            "format": {
                "name": "weak_task_result",
                "schema": weak_task_result_schema(action_draft_schema()),
            }
        }
    });

    let body = serde_json::to_string(&request_context).unwrap();
    assert!(body.contains("WEAK semantic effect handler"));
    assert!(body.contains("WeakTaskSpec"));
    assert!(body.contains("weak_task_result"));
}

#[test]
fn live_structured_output_schemas_follow_openai_strict_subset() {
    for (name, schema) in [
        (
            "weak_task_result",
            weak_task_result_schema(action_draft_schema()),
        ),
        ("think_decision", think_decision_schema()),
        ("program", program_schema()),
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
        assert_no_const(&schema, name);
        assert_no_boolean_schema(&schema, name);
        assert_no_root_schema_keyword(&schema, name);
        assert_no_numeric_format(&schema, name);
        assert_strict_object_schemas_disallow_additional_properties(&schema, name);
    }
}

#[test]
fn think_decision_schema_binds_decision_to_payload_shape() {
    let schema = think_decision_schema();

    let resume = json!({
        "think_decision": {
            "decision": "resume_with_value",
            "value": {
                "event_id": "m1",
                "kind": "create_task",
                "title": "发送新版 proposal",
                "datetime_hint": "明天 10 点前",
                "source": "strong_think"
            },
            "confidence": 0.88,
            "rationale": "resolved frame"
        }
    });
    validate_value(&schema, &resume).unwrap();

    let probe = json!({
        "think_decision": {
            "decision": "request_weak_probe",
            "out": "datetime_candidates",
            "task": {
                "name": "extract_datetime_candidates",
                "instructions": "Extract possible datetime hints."
            },
            "input": {
                "kind": "object",
                "fields": [
                    {
                        "name": "message",
                        "value": {
                            "kind": "var",
                            "name": "$input"
                        }
                    },
                    {
                        "name": "fallback",
                        "value": {
                            "kind": "literal",
                            "value": null
                        }
                    }
                ]
            },
            "output_schema": {
                "type": "object"
            },
            "min_confidence": 0.5,
            "rationale": "need local probe"
        }
    });
    validate_value(&schema, &probe).unwrap();

    let abort = json!({
        "think_decision": {
            "decision": "abort",
            "reason": "underspecified"
        }
    });
    validate_value(&schema, &abort).unwrap();

    let mismatched_decision = json!({
        "think_decision": {
            "decision": "abort",
            "value": {
                "event_id": "m1",
                "kind": "create_task"
            },
            "confidence": 0.88,
            "rationale": "bad shape"
        }
    });
    assert!(validate_value(&schema, &mismatched_decision).is_err());

    let parsed: cps_llm_demo::models::WeakTaskResult = serde_json::from_value(json!({
        "value": {
            "event_id": "m1",
            "kind": "create_task",
            "title": "发送新版 proposal",
            "datetime_hint": null,
            "source": "weak_model"
        },
        "confidence": 0.9,
        "rationale": "clear request"
    }))
    .unwrap();
    validate_value(
        &weak_task_result_schema(action_draft_schema()),
        &serde_json::to_value(parsed).unwrap(),
    )
    .unwrap();

    assert_eq!(
        ThinkDecision::RequestWeakProbe {
            out: "x".to_owned(),
            task: cps_llm_demo::program::WeakTaskSpec {
                name: "probe".to_owned(),
                instructions: "probe".to_owned(),
            },
            input: JsonExpr::Var {
                name: "$input".to_owned(),
            },
            output_schema: json!({}),
            min_confidence: 0.5,
            rationale: "probe".to_owned(),
        }
        .decision_name(),
        "request_weak_probe"
    );
}

#[test]
fn weak_task_result_schema_rejects_confidence_outside_probability_range() {
    let schema = weak_task_result_schema(action_draft_schema());
    let valid_guess = json!({
        "value": {
            "event_id": "m1",
            "kind": "create_task",
            "title": "Send proposal",
            "datetime_hint": null,
            "source": "weak_model"
        },
        "confidence": 1.0,
        "rationale": "clear request"
    });
    validate_value(&schema, &valid_guess).unwrap();

    for confidence in [-0.01, 1.01, 75.0] {
        let invalid_guess = json!({
            "value": {
                "event_id": "m1",
                "kind": "create_task",
                "title": "Send proposal",
                "datetime_hint": null,
                "source": "weak_model"
            },
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

fn assert_no_const(value: &Value, path: &str) {
    match value {
        Value::Object(map) => {
            assert!(
                !map.contains_key("const"),
                "schema must use enum instead of const at {path}"
            );
            for (key, child) in map {
                assert_no_const(child, &format!("{path}.{key}"));
            }
        }
        Value::Array(items) => {
            for (index, child) in items.iter().enumerate() {
                assert_no_const(child, &format!("{path}[{index}]"));
            }
        }
        Value::Null | Value::Bool(_) | Value::Number(_) | Value::String(_) => {}
    }
}

fn assert_no_boolean_schema(value: &Value, path: &str) {
    match value {
        Value::Bool(_) => panic!("schema must not contain boolean schemas at {path}"),
        Value::Object(map) => {
            for (key, child) in map {
                if key == "additionalProperties" && child.is_boolean() {
                    continue;
                }
                assert_no_boolean_schema(child, &format!("{path}.{key}"));
            }
        }
        Value::Array(items) => {
            for (index, child) in items.iter().enumerate() {
                assert_no_boolean_schema(child, &format!("{path}[{index}]"));
            }
        }
        Value::Null | Value::Number(_) | Value::String(_) => {}
    }
}

fn assert_no_root_schema_keyword(value: &Value, name: &str) {
    assert!(
        value.get("$schema").is_none(),
        "{name} must not include root $schema metadata"
    );
}

fn assert_strict_object_schemas_disallow_additional_properties(value: &Value, path: &str) {
    match value {
        Value::Object(map) => {
            let is_object_schema = match map.get("type") {
                Some(Value::String(type_name)) => type_name == "object",
                Some(Value::Array(type_names)) => type_names
                    .iter()
                    .any(|type_name| type_name.as_str() == Some("object")),
                _ => map.contains_key("properties"),
            };

            if is_object_schema {
                assert_eq!(
                    map.get("additionalProperties"),
                    Some(&Value::Bool(false)),
                    "strict object schema must set additionalProperties=false at {path}"
                );
            }

            for (key, child) in map {
                assert_strict_object_schemas_disallow_additional_properties(
                    child,
                    &format!("{path}.{key}"),
                );
            }
        }
        Value::Array(items) => {
            for (index, child) in items.iter().enumerate() {
                assert_strict_object_schemas_disallow_additional_properties(
                    child,
                    &format!("{path}[{index}]"),
                );
            }
        }
        Value::Null | Value::Bool(_) | Value::Number(_) | Value::String(_) => {}
    }
}
