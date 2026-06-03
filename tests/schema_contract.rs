use cps_llm_demo::effects::HandlerDecision;
use cps_llm_demo::models::WEAK_HANDLER_INSTRUCTIONS;
use cps_llm_demo::program::Program;
use cps_llm_demo::schema::{
    action_draft_schema, handler_decision_schema, program_schema, schema_bundle, validate_value,
    weak_task_result_schema,
};
use serde_json::{Value, json};

#[test]
fn schema_bundle_contains_v2_runtime_contracts() {
    let bundle = schema_bundle();
    assert!(bundle.get("program").is_some());
    assert!(bundle.get("program_fragment").is_some());
    assert!(bundle.get("program_patch").is_some());
    assert!(bundle.get("handler_decision").is_some());
    assert!(bundle.get("effect_frame").is_some());
    assert!(bundle.get("continuation").is_some());
    assert!(bundle.get("weak_intent_guess").is_none());
}

#[test]
fn weak_handler_context_uses_neutral_effect_handler_wording() {
    let request_context = json!({
        "instructions": WEAK_HANDLER_INSTRUCTIONS,
        "text": {
            "format": {
                "name": "handler_decision",
                "schema": handler_decision_schema(),
            }
        }
    });

    assert!(WEAK_HANDLER_INSTRUCTIONS.contains("effect handler"));
    assert!(
        !WEAK_HANDLER_INSTRUCTIONS
            .to_ascii_lowercase()
            .contains("weak")
    );

    let body = serde_json::to_string(&request_context).unwrap();
    assert!(body.contains("request_effect"));
    assert!(body.contains("handler_decision"));
}

#[test]
fn live_structured_output_schemas_have_object_roots_and_supported_keywords() {
    for (name, schema) in [
        ("handler_decision", handler_decision_schema()),
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
    }
}

#[test]
fn handler_decision_schema_binds_decision_to_payload_shape() {
    let schema = handler_decision_schema();

    let resume = json!({
        "handler_decision": {
            "decision": "return_value",
            "value": {
                "event_id": "m1",
                "kind": "create_task",
                "title": "发送新版 proposal",
                "datetime_hint": "明天 10 点前"
            },
            "confidence": 0.88,
            "rationale": "resolved frame"
        }
    });
    validate_value(&schema, &resume).unwrap();

    let request_effect = json!({
        "handler_decision": {
            "decision": "request_effect",
            "effect": {
                "kind": "think",
                "reason": "weak handler cannot resolve semantic ambiguity"
            },
            "input": {
                "partial": "ambiguous"
            },
            "expected_schema": action_draft_schema(),
            "mode": {
                "mode": "use_as_value"
            },
            "rationale": "need stronger reasoning"
        }
    });
    validate_value(&schema, &request_effect).unwrap();

    let abort = json!({
        "handler_decision": {
            "decision": "abort",
            "reason": "underspecified"
        }
    });
    validate_value(&schema, &abort).unwrap();

    let mismatched_decision = json!({
        "handler_decision": {
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

    assert_eq!(
        HandlerDecision::RequestEffect {
            effect: cps_llm_demo::program::EffectCall::Think {
                reason: "probe".to_owned(),
            },
            input: json!({}),
            expected_schema: json!({}),
            mode: cps_llm_demo::effects::EffectReturnMode::UseAsValue,
            rationale: "probe".to_owned(),
        }
        .decision_name(),
        "request_effect"
    );
}

#[test]
fn weak_task_result_schema_still_rejects_confidence_outside_probability_range() {
    let schema = weak_task_result_schema(action_draft_schema());
    let valid_guess = json!({
        "value": {
            "event_id": "m1",
            "kind": "create_task",
            "title": "Send proposal",
            "datetime_hint": null
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
                "datetime_hint": null
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

#[test]
fn program_schema_rejects_acceptance_min_confidence_outside_probability_range() {
    let schema = program_schema();
    let mut program: Value =
        serde_json::from_str(include_str!("../examples/message_action.v2.program.json")).unwrap();
    validate_value(&schema, &program).unwrap();

    for confidence in [-1.0, 75.0] {
        program["functions"]["process_message"]["body"][0]["acceptance"]["min_confidence"] =
            json!(confidence);

        assert!(
            validate_value(&schema, &program).is_err(),
            "min_confidence {confidence} must be rejected"
        );
        assert!(
            serde_json::from_value::<Program>(program.clone()).is_err(),
            "min_confidence {confidence} must fail Program deserialization"
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
