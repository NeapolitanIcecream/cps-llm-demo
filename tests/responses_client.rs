use cps_llm_demo::effects::{HandlerBudget, HandlerRequest, Observation, ObservationSource};
use cps_llm_demo::models::{EffectHandler, ResponsesWeakModel, WeakTaskResult};
use cps_llm_demo::program::{EffectCall, ModelStrength, ModelTaskSpec};
use cps_llm_demo::responses_client::{ResponsesClient, ResponsesClientConfig, extract_output_text};
use cps_llm_demo::schema::{action_draft_schema, weak_task_result_schema};
use httpmock::HttpMockRequest;
use httpmock::Method::POST;
use httpmock::MockServer;
use secrecy::SecretString;
use serde_json::{Value, json};
use std::sync::{Arc, Mutex};
use url::Url;

fn client(server: &MockServer, base_path: &str) -> ResponsesClient {
    ResponsesClient::new(ResponsesClientConfig {
        base_url: Url::parse(&server.url(base_path)).unwrap(),
        api_key: SecretString::from("test-key".to_owned()),
    })
}

fn weak_output_text() -> String {
    serde_json::to_string(&WeakTaskResult {
        value: json!({
            "event_id": "m2",
            "kind": "create_calendar_event",
            "title": "Product review",
            "datetime_hint": "Friday 3pm"
        }),
        confidence: 0.91,
        rationale: "obvious meeting".to_owned(),
    })
    .unwrap()
}

fn assert_no_weak_identity(context: &str, label: &str) {
    assert!(
        !context.to_ascii_lowercase().contains("weak"),
        "{label} must not self-identify the handler as weak: {context}"
    );
}

#[test]
fn extract_output_text_accepts_top_level_field() {
    let value = json!({ "output_text": "{\"ok\":true}" });
    assert_eq!(extract_output_text(&value).unwrap(), "{\"ok\":true}");
}

#[test]
fn extract_output_text_concatenates_message_content() {
    let value = json!({
        "output": [
            {
                "type": "message",
                "content": [
                    { "type": "output_text", "text": "{\"ok\"" },
                    { "type": "output_text", "text": ":true}" }
                ]
            }
        ]
    });

    assert_eq!(extract_output_text(&value).unwrap(), "{\"ok\":true}");
}

#[tokio::test]
async fn create_structured_posts_to_responses_and_parses_top_level_output_text() {
    let server = MockServer::start();
    let mock = server.mock(|when, then| {
        when.method(POST)
            .path("/v1/responses")
            .header("authorization", "Bearer test-key")
            .header_exists("x-client-request-id")
            .json_body_includes(r#"{"store":false,"text":{"format":{"name":"weak_task_result"}}}"#);
        then.status(200)
            .json_body(json!({ "output_text": weak_output_text() }));
    });

    let parsed: WeakTaskResult = client(&server, "/v1")
        .create_structured(
            "test-model",
            "Return JSON.",
            &json!({ "event_id": "m2", "text": "Friday 3pm product review" }),
            "weak_task_result",
            weak_task_result_schema(action_draft_schema()),
        )
        .await
        .unwrap();

    assert_eq!(parsed.value["kind"], "create_calendar_event");
    mock.assert();
}

#[tokio::test]
async fn responses_weak_model_sends_neutral_instructions_and_request_context() {
    let server = MockServer::start();
    let captured_body = Arc::new(Mutex::new(None::<Value>));
    let captured_body_for_matcher = Arc::clone(&captured_body);
    let mock = server.mock(move |when, then| {
        let captured_body = Arc::clone(&captured_body_for_matcher);
        when.method(POST)
            .path("/v1/responses")
            .is_true(move |request: &HttpMockRequest| {
                let Ok(body) = serde_json::from_slice::<Value>(request.body_ref()) else {
                    return false;
                };
                *captured_body.lock().unwrap() = Some(body);
                true
            });
        then.status(200).json_body(json!({
            "output_text": serde_json::to_string(&json!({
                "handler_decision": {
                    "decision": "return_value",
                    "value": {
                        "intent": "send_message"
                    },
                    "confidence": 0.91,
                    "rationale": "clear request"
                }
            })).unwrap()
        }));
    });

    let handler = ResponsesWeakModel::new(client(&server, "/v1"), "fake-weak");
    let decision = handler
        .handle(HandlerRequest {
            effect: EffectCall::ModelTask {
                strength: ModelStrength::Weak,
                task: ModelTaskSpec {
                    name: "classify_intent".to_owned(),
                    instructions: "Classify the message intent.".to_owned(),
                },
            },
            input: json!({
                "message": "Please send the proposal"
            }),
            expected_schema: action_draft_schema(),
            continuation_summary: None,
            effect_frame: None,
            observations: Vec::new(),
            budget: HandlerBudget {
                effect_depth: 0,
                effects_remaining: 8,
                handler_reentries_remaining: 2,
            },
        })
        .await
        .unwrap();

    assert_eq!(decision.decision_name(), "return_value");
    mock.assert();

    let captured_body = captured_body
        .lock()
        .unwrap()
        .clone()
        .expect("mock should capture Responses request body");
    assert_eq!(captured_body["model"], "fake-weak");

    let instructions = captured_body["instructions"]
        .as_str()
        .expect("Responses body should include instructions");
    assert_no_weak_identity(instructions, "weak handler instructions");

    let input_text = captured_body["input"][0]["content"][0]["text"]
        .as_str()
        .expect("Responses body should include serialized input text");
    assert_no_weak_identity(input_text, "weak handler request context");

    let input_context: Value = serde_json::from_str(input_text).unwrap();
    assert_eq!(input_context["effect"]["kind"], "model_task");
    assert_eq!(input_context["effect"]["task"]["name"], "classify_intent");
    assert!(
        input_context["effect"].get("strength").is_none(),
        "weak handler request context should omit internal model strength"
    );
}

#[tokio::test]
async fn responses_weak_model_preserves_schema_shaped_user_payloads() {
    let server = MockServer::start();
    let captured_body = Arc::new(Mutex::new(None::<Value>));
    let captured_body_for_matcher = Arc::clone(&captured_body);
    let mock = server.mock(move |when, then| {
        let captured_body = Arc::clone(&captured_body_for_matcher);
        when.method(POST)
            .path("/v1/responses")
            .is_true(move |request: &HttpMockRequest| {
                let Ok(body) = serde_json::from_slice::<Value>(request.body_ref()) else {
                    return false;
                };
                *captured_body.lock().unwrap() = Some(body);
                true
            });
        then.status(200).json_body(json!({
            "output_text": serde_json::to_string(&json!({
                "handler_decision": {
                    "decision": "return_value",
                    "value": {
                        "intent": "send_message"
                    },
                    "confidence": 0.91,
                    "rationale": "clear request"
                }
            })).unwrap()
        }));
    });

    let schema_shaped_payload = json!({
        "type": "object",
        "properties": {
            "source": {
                "type": "string",
                "enum": ["weak_model"]
            },
            "title": {
                "type": "string"
            }
        },
        "required": ["source", "title"]
    });
    let expected_schema = json!({
        "type": "object",
        "additionalProperties": false,
        "properties": {
            "title": { "type": "string" },
            "source": {
                "type": "string",
                "enum": ["user_supplied"]
            }
        },
        "required": ["title", "source"]
    });
    let handler = ResponsesWeakModel::new(client(&server, "/v1"), "fake-weak");
    let decision = handler
        .handle(HandlerRequest {
            effect: EffectCall::ModelTask {
                strength: ModelStrength::Weak,
                task: ModelTaskSpec {
                    name: "classify_schema_payload".to_owned(),
                    instructions: "Classify the payload.".to_owned(),
                },
            },
            input: json!({
                "user_supplied_schema": schema_shaped_payload.clone()
            }),
            expected_schema: expected_schema.clone(),
            continuation_summary: None,
            effect_frame: None,
            observations: vec![Observation {
                name: "user_payload_observation".to_owned(),
                value: schema_shaped_payload,
                source: ObservationSource::Runtime,
            }],
            budget: HandlerBudget {
                effect_depth: 0,
                effects_remaining: 8,
                handler_reentries_remaining: 2,
            },
        })
        .await
        .unwrap();

    assert_eq!(decision.decision_name(), "return_value");
    mock.assert();

    let captured_body = captured_body
        .lock()
        .unwrap()
        .clone()
        .expect("mock should capture Responses request body");
    let input_text = captured_body["input"][0]["content"][0]["text"]
        .as_str()
        .expect("Responses body should include serialized input text");
    let input_context: Value = serde_json::from_str(input_text).unwrap();

    assert_eq!(input_context["expected_schema"], expected_schema);

    let input_payload = &input_context["input"]["user_supplied_schema"];
    assert!(
        input_payload["properties"].get("source").is_some(),
        "weak handler request context should preserve user payload source fields"
    );
    assert!(
        input_payload["required"]
            .as_array()
            .unwrap()
            .iter()
            .any(|field| field.as_str() == Some("source")),
        "weak handler request context should preserve user payload source requirements"
    );

    let observation_payload = &input_context["observations"][0]["value"];
    assert!(
        observation_payload["properties"].get("source").is_some(),
        "weak handler request context should preserve observation payload source fields"
    );
    assert!(
        observation_payload["required"]
            .as_array()
            .unwrap()
            .iter()
            .any(|field| field.as_str() == Some("source")),
        "weak handler request context should preserve observation payload source requirements"
    );
}

#[tokio::test]
async fn create_structured_accepts_nested_output_text_and_trailing_base_url_slash() {
    let server = MockServer::start();
    let mock = server.mock(|when, then| {
        when.method(POST).path("/v1/responses");
        then.status(200).json_body(json!({
            "output": [
                {
                    "type": "message",
                    "content": [
                        { "type": "output_text", "text": weak_output_text() }
                    ]
                }
            ]
        }));
    });

    let parsed: WeakTaskResult = client(&server, "/v1/")
        .create_structured(
            "test-model",
            "Return JSON.",
            &json!({ "event_id": "m2", "text": "Friday 3pm product review" }),
            "weak_task_result",
            weak_task_result_schema(action_draft_schema()),
        )
        .await
        .unwrap();

    assert_eq!(parsed.value["title"], "Product review");
    mock.assert();
}

#[tokio::test]
async fn create_structured_returns_error_for_api_failure() {
    let server = MockServer::start();
    let mock = server.mock(|when, then| {
        when.method(POST).path("/v1/responses");
        then.status(500).json_body(json!({ "error": "boom" }));
    });

    let error = client(&server, "/v1")
        .create_structured::<WeakTaskResult>(
            "test-model",
            "Return JSON.",
            &json!({ "event_id": "m2", "text": "Friday 3pm product review" }),
            "weak_task_result",
            weak_task_result_schema(action_draft_schema()),
        )
        .await
        .unwrap_err();

    assert!(error.to_string().contains("Responses API error"));
    mock.assert();
}

#[tokio::test]
async fn create_structured_returns_error_for_non_json_output_text() {
    let server = MockServer::start();
    let mock = server.mock(|when, then| {
        when.method(POST).path("/v1/responses");
        then.status(200)
            .json_body(json!({ "output_text": "not-json" }));
    });

    let error = client(&server, "/v1")
        .create_structured::<WeakTaskResult>(
            "test-model",
            "Return JSON.",
            &json!({ "event_id": "m2", "text": "Friday 3pm product review" }),
            "weak_task_result",
            weak_task_result_schema(action_draft_schema()),
        )
        .await
        .unwrap_err();

    assert!(
        error
            .to_string()
            .contains("model output was not valid JSON")
    );
    mock.assert();
}

#[tokio::test]
async fn create_structured_rejects_schema_invalid_model_json_before_deserialize() {
    let server = MockServer::start();
    let mut output = serde_json::from_str::<serde_json::Value>(&weak_output_text()).unwrap();
    output["unexpected"] = json!("must not be ignored");
    let mock = server.mock(|when, then| {
        when.method(POST).path("/v1/responses");
        then.status(200).json_body(json!({
            "output_text": serde_json::to_string(&output).unwrap()
        }));
    });

    let error = client(&server, "/v1")
        .create_structured::<WeakTaskResult>(
            "test-model",
            "Return JSON.",
            &json!({ "event_id": "m2", "text": "Friday 3pm product review" }),
            "weak_task_result",
            weak_task_result_schema(action_draft_schema()),
        )
        .await
        .unwrap_err();

    assert!(
        error
            .to_string()
            .contains("model output failed schema validation")
    );
    mock.assert();
}
