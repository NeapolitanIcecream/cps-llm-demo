use assert_cmd::Command;
use httpmock::Method::POST;
use httpmock::MockServer;
use predicates::prelude::*;
use serde_json::json;
use std::fs;

fn write_temp_messages() -> std::path::PathBuf {
    let path = std::env::temp_dir().join(format!("cps-llm-demo-{}.json", uuid::Uuid::new_v4()));
    fs::write(
        &path,
        serde_json::to_string(&json!([
            {
                "event_id": "m1",
                "text": "明天 10 点前把新版 proposal 发我一下"
            }
        ]))
        .unwrap(),
    )
    .unwrap();
    path
}

#[test]
fn cli_schema_outputs_json() {
    let mut cmd = Command::cargo_bin("cps-llm-demo").unwrap();
    cmd.arg("schema")
        .assert()
        .success()
        .stdout(predicate::str::contains("weak_intent_guess"))
        .stdout(predicate::str::contains("think_decision"));
}

#[test]
fn cli_run_against_mock_responses_endpoint_outputs_json_array_and_trace() {
    let server = MockServer::start();
    let weak_mock = server.mock(|when, then| {
        when.method(POST).path("/v1/responses").json_body_includes(
            r#"{"model":"fake-classifier","text":{"format":{"name":"weak_intent_guess"}}}"#,
        );
        then.status(200).json_body(json!({
            "output_text": serde_json::to_string(&json!({
                "kind": "draft_reply",
                "title": "回复 proposal 请求",
                "datetime_hint": null,
                "confidence": 0.99,
                "rationale": "contains a request"
            })).unwrap()
        }));
    });
    let strong_mock = server.mock(|when, then| {
        when.method(POST).path("/v1/responses").json_body_includes(
            r#"{"model":"fake-thinker","text":{"format":{"name":"think_decision"}}}"#,
        );
        then.status(200).json_body(json!({
            "output_text": serde_json::to_string(&json!({
                "decision": "value",
                "data": {
                    "kind": "create_task",
                    "title": "发送新版 proposal",
                    "datetime_hint": "明天 10 点前",
                    "confidence": 0.88,
                    "source": "strong_think"
                }
            })).unwrap()
        }));
    });
    let input = write_temp_messages();

    let mut cmd = Command::cargo_bin("cps-llm-demo").unwrap();
    cmd.arg("run")
        .arg(&input)
        .arg("--base-url")
        .arg(server.url("/v1"))
        .arg("--api-key")
        .arg("test-key")
        .arg("--weak-model")
        .arg("fake-classifier")
        .arg("--strong-model")
        .arg("fake-thinker")
        .arg("--trace-json")
        .assert()
        .success()
        .stdout(predicate::str::contains("\"source\": \"strong_think\""))
        .stdout(predicate::str::contains("\"kind\": \"create_task\""))
        .stderr(predicate::str::contains("capture_continuation"))
        .stderr(predicate::str::contains("strong_think"))
        .stderr(predicate::str::contains("resume_continuation"));

    weak_mock.assert();
    strong_mock.assert();
    let _ = fs::remove_file(input);
}

#[test]
fn cli_run_without_api_key_has_clear_error() {
    let input = write_temp_messages();
    let mut cmd = Command::cargo_bin("cps-llm-demo").unwrap();
    cmd.arg("run")
        .arg(&input)
        .arg("--base-url")
        .arg("http://localhost:1/v1")
        .assert()
        .failure()
        .stderr(predicate::str::contains(
            "OPENAI_API_KEY is required. Set it via env or --api-key.",
        ));
    let _ = fs::remove_file(input);
}
