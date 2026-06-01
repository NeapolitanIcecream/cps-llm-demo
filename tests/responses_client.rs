use cps_llm_demo::domain::{IntentKind, WeakIntentGuess};
use cps_llm_demo::responses_client::{ResponsesClient, ResponsesClientConfig, extract_output_text};
use cps_llm_demo::schema::weak_intent_guess_schema;
use httpmock::Method::POST;
use httpmock::MockServer;
use secrecy::SecretString;
use serde_json::json;
use url::Url;

fn client(server: &MockServer, base_path: &str) -> ResponsesClient {
    ResponsesClient::new(ResponsesClientConfig {
        base_url: Url::parse(&server.url(base_path)).unwrap(),
        api_key: SecretString::from("test-key".to_owned()),
    })
}

fn weak_output_text() -> String {
    serde_json::to_string(&WeakIntentGuess {
        kind: IntentKind::CreateCalendarEvent,
        title: Some("Product review".to_owned()),
        datetime_hint: Some("Friday 3pm".to_owned()),
        confidence: 0.91,
        rationale: "obvious meeting".to_owned(),
    })
    .unwrap()
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
            .json_body_includes(
                r#"{"store":false,"text":{"format":{"name":"weak_intent_guess"}}}"#,
            );
        then.status(200)
            .json_body(json!({ "output_text": weak_output_text() }));
    });

    let parsed: WeakIntentGuess = client(&server, "/v1")
        .create_structured(
            "test-model",
            "Return JSON.",
            &json!({ "event_id": "m2", "text": "Friday 3pm product review" }),
            "weak_intent_guess",
            weak_intent_guess_schema(),
        )
        .await
        .unwrap();

    assert_eq!(parsed.kind, IntentKind::CreateCalendarEvent);
    mock.assert();
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

    let parsed: WeakIntentGuess = client(&server, "/v1/")
        .create_structured(
            "test-model",
            "Return JSON.",
            &json!({ "event_id": "m2", "text": "Friday 3pm product review" }),
            "weak_intent_guess",
            weak_intent_guess_schema(),
        )
        .await
        .unwrap();

    assert_eq!(parsed.title.as_deref(), Some("Product review"));
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
        .create_structured::<WeakIntentGuess>(
            "test-model",
            "Return JSON.",
            &json!({ "event_id": "m2", "text": "Friday 3pm product review" }),
            "weak_intent_guess",
            weak_intent_guess_schema(),
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
        .create_structured::<WeakIntentGuess>(
            "test-model",
            "Return JSON.",
            &json!({ "event_id": "m2", "text": "Friday 3pm product review" }),
            "weak_intent_guess",
            weak_intent_guess_schema(),
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
        .create_structured::<WeakIntentGuess>(
            "test-model",
            "Return JSON.",
            &json!({ "event_id": "m2", "text": "Friday 3pm product review" }),
            "weak_intent_guess",
            weak_intent_guess_schema(),
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
