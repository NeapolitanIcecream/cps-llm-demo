use cps_llm_demo::models::CLASSIFIER_INSTRUCTIONS;
use cps_llm_demo::schema::{schema_bundle, think_decision_schema, validate_value};
use serde_json::json;

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
fn think_decision_schema_binds_decision_to_payload_shape() {
    let schema = think_decision_schema();

    let value_decision = json!({
        "decision": "value",
        "data": {
            "kind": "create_task",
            "title": "发送新版 proposal",
            "datetime_hint": "明天 10 点前",
            "confidence": 0.88,
            "source": "strong_think"
        }
    });
    validate_value(&schema, &value_decision).unwrap();

    let abort_decision = json!({
        "decision": "abort",
        "data": {
            "reason": "underspecified"
        }
    });
    validate_value(&schema, &abort_decision).unwrap();

    let mismatched_decision = json!({
        "decision": "abort",
        "data": {
            "kind": "create_task",
            "title": "bad",
            "datetime_hint": null,
            "confidence": 0.88,
            "source": "strong_think"
        }
    });
    assert!(validate_value(&schema, &mismatched_decision).is_err());
}
