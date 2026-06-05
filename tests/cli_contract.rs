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

fn write_minimal_experiment_config(
    workdir: &std::path::Path,
    experiment_id: &str,
    state_dir: &str,
) -> std::path::PathBuf {
    fs::write(workdir.join("all_events.jsonl"), "{}\n").unwrap();
    let config = json!({
        "experiment_id": experiment_id,
        "workflow_id": "notification_triage",
        "state_dir": state_dir,
        "models": {
            "weak_model": "gpt-5.4-mini",
            "strong_model": "gpt-5.5",
            "base_url_env": "CPS_TEST_OPENAI_BASE_URL_MISSING",
            "api_key_env": "CPS_TEST_OPENAI_API_KEY_MISSING",
            "use_responses_api": true,
            "structured_outputs": true
        },
        "budget": {
            "price_catalog": "missing-prices.yaml",
            "soft_cap_usd": 1.0,
            "hard_cap_usd": 100.0,
            "projection_multiplier": 1.5
        },
        "cache": {
            "dir": "model-cache",
            "mode": "read_write"
        },
        "schemas": {
            "event_schema": "event.schema.json",
            "output_schema": "output.schema.json",
            "gold_schema": "gold.schema.json"
        },
        "data": {
            "all_events": "all_events.jsonl",
            "gold_labels": "gold_labels.jsonl",
            "splits_dir": "splits"
        },
        "phases": []
    });
    let path = workdir.join(format!("{experiment_id}.yaml"));
    fs::write(&path, serde_yaml::to_string(&config).unwrap()).unwrap();
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
                        "datetime_hint": "明天 10 点前"
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
                        "datetime_hint": "明天 10 点前"
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
        .stdout(predicate::str::contains("\"kind\": \"create_task\""))
        .stdout(predicate::str::contains("\"source\"").not())
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
                        "datetime_hint": "明天 10 点前"
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
        .stdout(predicate::str::contains("\"kind\": \"create_task\""))
        .stdout(predicate::str::contains("\"source\"").not());

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
fn cli_run_experiment_uses_config_state_dir_when_flag_omitted() {
    let workdir = make_temp_workdir();
    let experiment_id = "config_state_dir_cli_test";
    let config = write_minimal_experiment_config(&workdir, experiment_id, "config-state");

    let mut cmd = Command::cargo_bin("cps-llm-demo").unwrap();
    cmd.current_dir(&workdir)
        .arg("run-experiment")
        .arg("--config")
        .arg(&config)
        .arg("--dry-run-cost")
        .assert()
        .success()
        .stdout(predicate::str::contains("\"fits_budget\": true"));

    assert!(
        workdir
            .join("config-state")
            .join("experiments")
            .join(experiment_id)
            .join("config.lock.yaml")
            .exists()
    );
    assert!(!workdir.join(".cps-real-exp").exists());

    let _ = fs::remove_dir_all(workdir);
}

#[test]
fn cli_run_experiment_state_dir_flag_overrides_config_state_dir() {
    let workdir = make_temp_workdir();
    let experiment_id = "config_state_dir_override_cli_test";
    let config = write_minimal_experiment_config(&workdir, experiment_id, "config-state");

    let mut cmd = Command::cargo_bin("cps-llm-demo").unwrap();
    cmd.current_dir(&workdir)
        .arg("run-experiment")
        .arg("--config")
        .arg(&config)
        .arg("--state-dir")
        .arg("override-state")
        .arg("--dry-run-cost")
        .assert()
        .success()
        .stdout(predicate::str::contains("\"fits_budget\": true"));

    assert!(
        workdir
            .join("override-state")
            .join("experiments")
            .join(experiment_id)
            .join("config.lock.yaml")
            .exists()
    );
    assert!(!workdir.join("config-state").exists());

    let _ = fs::remove_dir_all(workdir);
}

#[test]
fn cli_init_workflow_from_task_records_strong_compile_metric() {
    let server = MockServer::start();
    let compile_mock = server.mock(|when, then| {
        when.method(POST)
            .path("/v1/responses")
            .json_body_includes(r#"{"model":"gpt-5.5","text":{"format":{"name":"program"}}}"#);
        then.status(200).json_body(json!({
            "output_text": include_str!("../examples/message_action.v2.program.json")
        }));
    });
    let state_dir = make_temp_workdir();

    let mut init_cmd = Command::cargo_bin("cps-llm-demo").unwrap();
    init_cmd
        .arg("init-workflow")
        .arg("--workflow")
        .arg("compiled_fixture")
        .arg("--task")
        .arg("examples/message_action.task.md")
        .arg("--state-dir")
        .arg(&state_dir)
        .arg("--base-url")
        .arg(server.url("/v1"))
        .arg("--api-key")
        .arg("test-key")
        .arg("--weak-model")
        .arg("gpt-5.4-mini")
        .arg("--strong-model")
        .arg("gpt-5.5")
        .assert()
        .success()
        .stdout(predicate::str::contains(
            "\"latest_program_version\": \"v0001\"",
        ));

    let mut report_cmd = Command::cargo_bin("cps-llm-demo").unwrap();
    let report = report_cmd
        .arg("metrics-report")
        .arg("--workflow")
        .arg("compiled_fixture")
        .arg("--state-dir")
        .arg(&state_dir)
        .output()
        .unwrap();
    assert!(report.status.success());
    let report: serde_json::Value = serde_json::from_slice(&report.stdout).unwrap();
    assert_eq!(report["summary"]["cps_strong_compile_calls"], 1);
    assert_eq!(report["latest_program_version"], "v0001");

    compile_mock.assert();
    let _ = fs::remove_dir_all(state_dir);
}

#[test]
fn cli_init_workflow_rejects_existing_workflow() {
    let state_dir = make_temp_workdir();

    let mut first = Command::cargo_bin("cps-llm-demo").unwrap();
    first
        .arg("init-workflow")
        .arg("--workflow")
        .arg("existing_fixture")
        .arg("--program")
        .arg("examples/message_action.v2.program.json")
        .arg("--state-dir")
        .arg(&state_dir)
        .assert()
        .success();

    let mut second = Command::cargo_bin("cps-llm-demo").unwrap();
    second
        .arg("init-workflow")
        .arg("--workflow")
        .arg("existing_fixture")
        .arg("--program")
        .arg("examples/message_action.v2.program.json")
        .arg("--state-dir")
        .arg(&state_dir)
        .assert()
        .failure()
        .stderr(predicate::str::contains("already exists"))
        .stdout(predicate::str::is_empty());

    let _ = fs::remove_dir_all(state_dir);
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
