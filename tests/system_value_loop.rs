use assert_cmd::Command;
use cps_llm_demo::local_tools::{
    PredicateExpr, ValidatorInput, ValidatorSpec, apply_fast_path, apply_validators,
};
use cps_llm_demo::program::{FunctionDef, GuardExpr, Instr, JsonExpr, Program};
use cps_llm_demo::runtime::Runtime;
use cps_llm_demo::trace::TraceCollector;
use cps_llm_demo::validator::validate_program;
use cps_llm_demo::{models::FixtureModelHandler, program::EffectPermission};
use serde_json::{Value, json};
use std::collections::BTreeMap;
use std::fs;

#[test]
fn fast_path_apply_is_generic() {
    let output = apply_fast_path(json!({
        "event": { "a": { "b": "x" } },
        "rules": [
            {
                "rule_id": "rule_001",
                "when": {
                    "op": "field_equals",
                    "path": ["a", "b"],
                    "value": "x"
                },
                "emit": {
                    "kind": "object",
                    "fields": {
                        "copied": {
                            "kind": "field",
                            "path": ["a", "b"]
                        }
                    }
                },
                "confidence": 0.99
            }
        ]
    }))
    .unwrap();

    assert!(output.hit);
    assert_eq!(output.rule_id.as_deref(), Some("rule_001"));
    assert_eq!(output.value, Some(json!({ "copied": "x" })));
}

#[test]
fn validator_apply_is_generic() {
    let output = apply_validators(
        serde_json::to_value(ValidatorInput {
            value: json!({ "a": { "b": "x" } }),
            validators: vec![ValidatorSpec {
                validator_id: "has_nested_x".to_owned(),
                predicates: vec![PredicateExpr::FieldEquals {
                    path: vec!["a".to_owned(), "b".to_owned()],
                    value: json!("x"),
                }],
            }],
        })
        .unwrap(),
    )
    .unwrap();

    assert!(output.passed);
    assert!(output.failed_validator_ids.is_empty());
}

#[tokio::test]
async fn branch_selects_then_pc() {
    let program = branch_program();
    validate_program(&program).unwrap();

    let runtime = Runtime::new(
        FixtureModelHandler::weak(),
        FixtureModelHandler::strong(),
        TraceCollector::default(),
    );
    let output = runtime.run_program(program, json!({})).await.unwrap();

    assert_eq!(output, json!("hit"));
}

#[test]
fn jump_target_out_of_range_is_rejected() {
    let mut program = branch_program();
    program.functions.get_mut("main").unwrap().body[1] = Instr::Jump { pc: 99 };

    let error = validate_program(&program).unwrap_err();

    assert!(error.to_string().contains("jump pc 99 is outside"));
}

#[test]
fn cli_fixture_value_loop_proves_kpis() {
    let state = temp_state_dir();

    cargo_ok([
        "init-workflow",
        "--workflow",
        "notification_triage",
        "--program",
        "examples/notification_triage.v1.program.json",
        "--state-dir",
        state.to_str().unwrap(),
    ]);
    cargo_ok([
        "run-stream",
        "--workflow",
        "notification_triage",
        "--events",
        "examples/notification_triage.round1.jsonl",
        "--state-dir",
        state.to_str().unwrap(),
    ]);
    cargo_ok([
        "optimize",
        "--workflow",
        "notification_triage",
        "--patch",
        "examples/notification_triage.fast_path.patch.json",
        "--state-dir",
        state.to_str().unwrap(),
    ]);
    cargo_ok([
        "run-stream",
        "--workflow",
        "notification_triage",
        "--events",
        "examples/notification_triage.round2.jsonl",
        "--state-dir",
        state.to_str().unwrap(),
    ]);
    cargo_ok([
        "baseline-strong-direct",
        "--workflow",
        "notification_triage",
        "--task",
        "examples/notification_triage.task.md",
        "--events",
        "examples/notification_triage.round1.jsonl",
        "--state-dir",
        state.to_str().unwrap(),
    ]);

    let report = cargo_json([
        "metrics-report",
        "--workflow",
        "notification_triage",
        "--state-dir",
        state.to_str().unwrap(),
    ]);

    assert_eq!(report["latest_program_version"], "v0002");
    assert_eq!(report["summary"]["strong_direct_calls"], 8);
    assert!(
        report["summary"]["program_version_advanced"]
            .as_bool()
            .unwrap()
    );
    assert!(
        report["summary"]["fast_path_hit_rate_round2"]
            .as_f64()
            .unwrap()
            > report["summary"]["fast_path_hit_rate_round1"]
                .as_f64()
                .unwrap()
    );
    assert!(
        report["summary"]["strong_think_rate_round2"]
            .as_f64()
            .unwrap()
            < report["summary"]["strong_think_rate_round1"]
                .as_f64()
                .unwrap()
    );
    assert!(
        report["summary"]["continuation_frame_bytes_p95"]
            .as_u64()
            .unwrap()
            <= 16_384
    );

    let _ = fs::remove_dir_all(state);
}

fn branch_program() -> Program {
    let mut functions = BTreeMap::new();
    functions.insert(
        "main".to_owned(),
        FunctionDef {
            params: Vec::new(),
            output_schema: json!({ "type": "string" }),
            body: vec![
                Instr::Let {
                    var: "flag".to_owned(),
                    expr: JsonExpr::Literal { value: json!(true) },
                },
                Instr::Branch {
                    condition: GuardExpr::FieldIsTruthy {
                        var: "flag".to_owned(),
                        path: Vec::new(),
                    },
                    then_pc: 3,
                    else_pc: 2,
                },
                Instr::Return {
                    value: JsonExpr::Literal {
                        value: json!("miss"),
                    },
                },
                Instr::Return {
                    value: JsonExpr::Literal {
                        value: json!("hit"),
                    },
                },
            ],
        },
    );

    Program {
        program_id: "branch_fixture".to_owned(),
        version: "v0001".to_owned(),
        entry: "main".to_owned(),
        input_schema: json!({}),
        output_schema: json!({ "type": "string" }),
        functions,
        allowed_effects: Vec::<EffectPermission>::new(),
    }
}

fn temp_state_dir() -> std::path::PathBuf {
    let path = std::env::temp_dir().join(format!("cps-value-loop-{}", uuid::Uuid::new_v4()));
    fs::create_dir(&path).unwrap();
    path
}

fn cargo_ok<const N: usize>(args: [&str; N]) {
    let output = Command::cargo_bin("cps-llm-demo")
        .unwrap()
        .args(args)
        .env_remove("OPENAI_API_KEY")
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "command failed\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

fn cargo_json<const N: usize>(args: [&str; N]) -> Value {
    let output = Command::cargo_bin("cps-llm-demo")
        .unwrap()
        .args(args)
        .env_remove("OPENAI_API_KEY")
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "command failed\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    serde_json::from_slice(&output.stdout).unwrap()
}
