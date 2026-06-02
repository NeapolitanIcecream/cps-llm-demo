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

fn write_temp_program() -> std::path::PathBuf {
    let path = std::env::temp_dir().join(format!(
        "cps-llm-demo-program-{}.json",
        uuid::Uuid::new_v4()
    ));
    fs::write(
        &path,
        include_str!("../examples/message_action.v2.program.json"),
    )
    .unwrap();
    path
}

fn write_temp_trace(raw: &str) -> std::path::PathBuf {
    let path =
        std::env::temp_dir().join(format!("cps-llm-demo-trace-{}.jsonl", uuid::Uuid::new_v4()));
    fs::write(&path, raw).unwrap();
    path
}

fn make_temp_workdir() -> std::path::PathBuf {
    let path = std::env::temp_dir().join(format!("cps-llm-demo-workdir-{}", uuid::Uuid::new_v4()));
    fs::create_dir(&path).unwrap();
    path
}

#[test]
fn cli_schema_outputs_json() {
    let mut cmd = Command::cargo_bin("cps-llm-demo").unwrap();
    cmd.arg("schema")
        .assert()
        .success()
        .stdout(predicate::str::contains("program"))
        .stdout(predicate::str::contains("handler_decision"))
        .stdout(predicate::str::contains("continuation"));
}

#[test]
fn cli_validate_program_accepts_v2_fixture() {
    let mut cmd = Command::cargo_bin("cps-llm-demo").unwrap();
    cmd.arg("validate-program")
        .arg("--program")
        .arg("examples/message_action.v2.program.json")
        .assert()
        .success()
        .stdout(predicate::str::contains("\"ok\":true"));
}

#[test]
fn cli_replay_validates_capture_resume_pairs() {
    let trace = write_temp_trace(
        r#"{"event":"capture_continuation","event_id":"t1","detail":{"continuation_id":"k1"}}"#,
    );
    fs::write(
        &trace,
        concat!(
            "{\"event\":\"capture_continuation\",\"event_id\":\"t1\",\"detail\":{\"continuation_id\":\"k1\"}}\n",
            "{\"event\":\"resume_continuation\",\"event_id\":\"t1\",\"detail\":{\"continuation_id\":\"k1\"}}\n"
        ),
    )
    .unwrap();

    let mut cmd = Command::cargo_bin("cps-llm-demo").unwrap();
    cmd.arg("replay")
        .arg("--trace")
        .arg(&trace)
        .assert()
        .success()
        .stdout(predicate::str::contains("\"captures\":1"))
        .stdout(predicate::str::contains("\"resumes\":1"));

    let _ = fs::remove_file(trace);
}

#[test]
fn cli_run_program_against_mock_responses_endpoint_outputs_json_array_and_trace() {
    let server = MockServer::start();
    let intent_mock = server.mock(|when, then| {
        when.method(POST)
            .path("/v1/responses")
            .json_body_includes(
                r#"{"model":"fake-weak","text":{"format":{"name":"handler_decision"}}}"#,
            )
            .body_includes("classify_intent");
        then.status(200).json_body(json!({
            "output_text": serde_json::to_string(&json!({
                "handler_decision": {
                    "decision": "return_value",
                    "value": {
                        "kind": "create_task"
                    },
                    "confidence": 0.91,
                    "rationale": "clear intent"
                }
            })).unwrap()
        }));
    });
    let draft_mock = server.mock(|when, then| {
        when.method(POST)
            .path("/v1/responses")
            .json_body_includes(
                r#"{"model":"fake-weak","text":{"format":{"name":"handler_decision"}}}"#,
            )
            .body_includes("extract_action_draft_from_intent");
        then.status(200).json_body(json!({
            "output_text": serde_json::to_string(&json!({
                "handler_decision": {
                    "decision": "return_value",
                    "value": {
                        "event_id": "m1",
                        "kind": "create_task",
                        "title": "发送新版 proposal",
                        "datetime_hint": "明天 10 点前",
                        "source": "weak_model"
                    },
                    "confidence": 0.42,
                    "rationale": "ambiguous request"
                }
            })).unwrap()
        }));
    });
    let strong_mock = server.mock(|when, then| {
        when.method(POST).path("/v1/responses").json_body_includes(
            r#"{"model":"fake-strong","text":{"format":{"name":"handler_decision"}}}"#,
        );
        then.status(200).json_body(json!({
            "output_text": serde_json::to_string(&json!({
                "handler_decision": {
                    "decision": "return_value",
                    "value": {
                        "event_id": "m1",
                        "kind": "create_task",
                        "title": "发送新版 proposal",
                        "datetime_hint": "明天 10 点前",
                        "source": "strong_think"
                    },
                    "confidence": 0.88,
                    "rationale": "resolved continuation"
                }
            })).unwrap()
        }));
    });
    let input = write_temp_messages();
    let program = write_temp_program();

    let mut cmd = Command::cargo_bin("cps-llm-demo").unwrap();
    cmd.arg("run-program")
        .arg("--program")
        .arg(&program)
        .arg("--input")
        .arg(&input)
        .arg("--base-url")
        .arg(server.url("/v1"))
        .arg("--api-key")
        .arg("test-key")
        .arg("--weak-model")
        .arg("fake-weak")
        .arg("--strong-model")
        .arg("fake-strong")
        .arg("--trace-json")
        .assert()
        .success()
        .stdout(predicate::str::contains("\"source\": \"strong_think\""))
        .stdout(predicate::str::contains("\"kind\": \"create_task\""))
        .stderr(predicate::str::contains("capture_continuation"))
        .stderr(predicate::str::contains("handler_decision"))
        .stderr(predicate::str::contains("resume_continuation"));

    intent_mock.assert();
    draft_mock.assert();
    strong_mock.assert();
    let _ = fs::remove_file(input);
    let _ = fs::remove_file(program);
}

#[test]
fn cli_compile_run_uses_strong_compile_then_same_runtime() {
    let server = MockServer::start();
    let compile_mock = server.mock(|when, then| {
        when.method(POST)
            .path("/v1/responses")
            .json_body_includes(r#"{"model":"fake-strong","text":{"format":{"name":"program"}}}"#);
        then.status(200).json_body(json!({
            "output_text": include_str!("../examples/message_action.v2.program.json")
        }));
    });
    let intent_mock = server.mock(|when, then| {
        when.method(POST)
            .path("/v1/responses")
            .json_body_includes(
                r#"{"model":"fake-weak","text":{"format":{"name":"handler_decision"}}}"#,
            )
            .body_includes("classify_intent");
        then.status(200).json_body(json!({
            "output_text": serde_json::to_string(&json!({
                "handler_decision": {
                    "decision": "return_value",
                    "value": {
                        "kind": "create_task"
                    },
                    "confidence": 0.95,
                    "rationale": "clear intent"
                }
            })).unwrap()
        }));
    });
    let draft_mock = server.mock(|when, then| {
        when.method(POST)
            .path("/v1/responses")
            .json_body_includes(
                r#"{"model":"fake-weak","text":{"format":{"name":"handler_decision"}}}"#,
            )
            .body_includes("extract_action_draft_from_intent");
        then.status(200).json_body(json!({
            "output_text": serde_json::to_string(&json!({
                "handler_decision": {
                    "decision": "return_value",
                    "value": {
                        "event_id": "m1",
                        "kind": "create_task",
                        "title": "发送新版 proposal",
                        "datetime_hint": "明天 10 点前",
                        "source": "weak_model"
                    },
                    "confidence": 0.95,
                    "rationale": "clear request"
                }
            })).unwrap()
        }));
    });
    let input = write_temp_messages();

    let mut cmd = Command::cargo_bin("cps-llm-demo").unwrap();
    cmd.arg("compile-run")
        .arg("--task")
        .arg("examples/message_action.task.md")
        .arg("--input")
        .arg(&input)
        .arg("--base-url")
        .arg(server.url("/v1"))
        .arg("--api-key")
        .arg("test-key")
        .arg("--weak-model")
        .arg("fake-weak")
        .arg("--strong-model")
        .arg("fake-strong")
        .assert()
        .success()
        .stdout(predicate::str::contains("\"source\": \"weak_model\""))
        .stdout(predicate::str::contains("\"kind\": \"create_task\""));

    compile_mock.assert();
    intent_mock.assert();
    draft_mock.assert();
    let _ = fs::remove_file(input);
}

#[test]
fn cli_compile_run_enforces_requested_output_schema_on_compiled_program() {
    let server = MockServer::start();
    let compiled_program = json!({
        "program_id": "relaxed_output_schema",
        "version": "1.0.0",
        "entry": "main",
        "input_schema": {},
        "output_schema": {},
        "functions": {
            "main": {
                "params": ["messages"],
                "output_schema": {},
                "body": [
                    {
                        "op": "return",
                        "value": {
                            "kind": "literal",
                            "value": {
                                "not_an_action_array": true
                            }
                        }
                    }
                ]
            }
        },
        "allowed_effects": []
    });
    let compile_mock = server.mock(|when, then| {
        when.method(POST)
            .path("/v1/responses")
            .json_body_includes(r#"{"model":"fake-strong","text":{"format":{"name":"program"}}}"#);
        then.status(200).json_body(json!({
            "output_text": serde_json::to_string(&compiled_program).unwrap()
        }));
    });
    let input = write_temp_messages();

    let mut cmd = Command::cargo_bin("cps-llm-demo").unwrap();
    cmd.arg("compile-run")
        .arg("--task")
        .arg("examples/message_action.task.md")
        .arg("--input")
        .arg(&input)
        .arg("--base-url")
        .arg(server.url("/v1"))
        .arg("--api-key")
        .arg("test-key")
        .arg("--weak-model")
        .arg("fake-weak")
        .arg("--strong-model")
        .arg("fake-strong")
        .assert()
        .failure()
        .stderr(predicate::str::contains("program output failed schema"))
        .stdout(predicate::str::is_empty());

    compile_mock.assert();
    let _ = fs::remove_file(input);
}

#[test]
fn cli_run_program_without_api_key_has_clear_error() {
    let input = write_temp_messages();
    let program = write_temp_program();
    let workdir = make_temp_workdir();
    let mut cmd = Command::cargo_bin("cps-llm-demo").unwrap();
    cmd.arg("run-program")
        .arg("--program")
        .arg(&program)
        .arg("--input")
        .arg(&input)
        .arg("--base-url")
        .arg("http://localhost:1/v1")
        .current_dir(&workdir)
        .env_remove("OPENAI_API_KEY")
        .assert()
        .failure()
        .stderr(predicate::str::contains(
            "OPENAI_API_KEY is required. Set it via env or --api-key.",
        ));
    let _ = fs::remove_file(input);
    let _ = fs::remove_file(program);
    let _ = fs::remove_dir(workdir);
}
