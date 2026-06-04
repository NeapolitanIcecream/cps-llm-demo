use anyhow::Result;
use assert_cmd::Command;
use async_trait::async_trait;
use cps_llm_demo::effects::{
    AllowedDecision, Continuation, EffectFrame, EffectFrameEncoder, HandlerDecision,
    HandlerRequest, RuntimeFrame,
};
use cps_llm_demo::engine::event_source::JsonlEventSource;
use cps_llm_demo::engine::run_coordinator::run_stream;
use cps_llm_demo::local_tools::{
    PredicateExpr, ValidatorInput, ValidatorSpec, apply_fast_path, apply_validators,
};
use cps_llm_demo::models::EffectHandler;
use cps_llm_demo::optimizer::patch_optimizer::{OptimizerContext, optimize_from_profile};
use cps_llm_demo::program::{FunctionDef, GuardExpr, Instr, JsonExpr, Program};
use cps_llm_demo::runtime::Runtime;
use cps_llm_demo::store::continuation_store::{
    FileContinuationStore, FileEffectFrameEncoder, FrameEncodingConfig,
};
use cps_llm_demo::store::patch_registry::FilePatchRegistry;
use cps_llm_demo::store::profile_store::FileProfileStore;
use cps_llm_demo::store::program_registry::{FileProgramRegistry, fixture_program_metadata};
use cps_llm_demo::store::state_dir::StateDir;
use cps_llm_demo::store::trace_store::FileTraceStore;
use cps_llm_demo::store::value_store::FileValueStore;
use cps_llm_demo::trace::TraceCollector;
use cps_llm_demo::validator::validate_patch;
use cps_llm_demo::validator::validate_program;
use cps_llm_demo::{models::FixtureModelHandler, program::EffectPermission};
use serde_json::Map;
use serde_json::{Value, json};
use std::collections::BTreeMap;
use std::fs;
use std::path::Path;
use std::sync::Arc;

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

#[tokio::test]
async fn branch_selects_else_pc_and_jump_forward_works() {
    let mut program = branch_program();
    program.functions.get_mut("main").unwrap().body = vec![
        Instr::Let {
            var: "flag".to_owned(),
            expr: JsonExpr::Literal {
                value: json!(false),
            },
        },
        Instr::Branch {
            condition: GuardExpr::FieldIsTruthy {
                var: "flag".to_owned(),
                path: Vec::new(),
            },
            then_pc: 2,
            else_pc: 3,
        },
        Instr::Return {
            value: JsonExpr::Literal {
                value: json!("hit"),
            },
        },
        Instr::Jump { pc: 4 },
        Instr::Return {
            value: JsonExpr::Literal {
                value: json!("miss"),
            },
        },
    ];
    validate_program(&program).unwrap();

    let runtime = Runtime::new(
        FixtureModelHandler::weak(),
        FixtureModelHandler::strong(),
        TraceCollector::default(),
    );
    let output = runtime.run_program(program, json!({})).await.unwrap();

    assert_eq!(output, json!("miss"));
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
    assert_eq!(
        fs::read_to_string("examples/notification_triage.round1.jsonl")
            .unwrap()
            .lines()
            .count(),
        100
    );
    assert_eq!(
        fs::read_to_string("examples/notification_triage.round2.jsonl")
            .unwrap()
            .lines()
            .count(),
        100
    );

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
    assert!(
        state
            .join("workflows/notification_triage/profiles/failure_fingerprints.json")
            .exists()
    );
    assert!(
        state
            .join("workflows/notification_triage/profiles/fast_path_stats.json")
            .exists()
    );
    assert!(
        state
            .join("workflows/notification_triage/profiles/effect_stats.json")
            .exists()
    );
    cargo_ok([
        "optimize",
        "--workflow",
        "notification_triage",
        "--patch",
        "examples/notification_triage.fast_path.patch.json",
        "--state-dir",
        state.to_str().unwrap(),
    ]);
    let installed_patch: Value = serde_json::from_str(
        &fs::read_to_string(
            state.join("workflows/notification_triage/patches/installed/fixture_fast_path_v1.json"),
        )
        .unwrap(),
    )
    .unwrap();
    assert!(
        state
            .join("workflows/notification_triage/patches/validated/fixture_fast_path_v1.json")
            .exists()
    );
    assert_eq!(
        installed_patch["metadata"]["metrics_delta"]["base_fast_path_hits"],
        0
    );
    assert!(
        installed_patch["metadata"]["metrics_delta"]["patched_fast_path_hits"]
            .as_u64()
            .unwrap()
            > 0
    );
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
    assert_eq!(report["summary"]["events_total"], 200);
    assert_eq!(report["summary"]["strong_direct_calls"], 100);
    for run in report["runs"].as_array().unwrap() {
        assert_eq!(
            run["events_failed"], 0,
            "fixture value loop must not leave failed events in run {}",
            run["run_id"]
        );
    }
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
        report["summary"]["strong_call_reduction_vs_baseline"]
            .as_f64()
            .unwrap()
            > 0.0
    );
    assert!(
        report["summary"]["continuation_frame_bytes_p95"]
            .as_u64()
            .unwrap()
            <= 16_384
    );

    let _ = fs::remove_dir_all(state);
}

#[tokio::test]
async fn optimizer_without_fixture_patch_installs_strong_returned_patch() {
    let state_path = temp_state_dir();
    let state = StateDir::new(state_path.clone());
    let workflow_id = "notification_triage";
    let program: Program = serde_json::from_str(include_str!(
        "../examples/notification_triage.v1.program.json"
    ))
    .unwrap();
    let programs = FileProgramRegistry::new(state.clone());
    programs
        .init_workflow(
            workflow_id,
            program.clone(),
            fixture_program_metadata(workflow_id, &program),
        )
        .unwrap();
    let weak: Arc<dyn EffectHandler> = Arc::new(FixtureModelHandler::weak());
    let strong: Arc<dyn EffectHandler> = Arc::new(PatchReturningStrong {
        patch: serde_json::from_str(include_str!(
            "../examples/notification_triage.fast_path.patch.json"
        ))
        .unwrap(),
    });
    run_stream(
        state.clone(),
        workflow_id,
        JsonlEventSource::from_path(Path::new("examples/notification_triage.round1.jsonl"))
            .unwrap(),
        Arc::clone(&weak),
        Arc::clone(&strong),
        false,
    )
    .await
    .unwrap();

    let patches = FilePatchRegistry::new(state.clone());
    let traces = FileTraceStore::new(state.clone());
    let profiles = FileProfileStore::new(state);
    let installed = optimize_from_profile(
        OptimizerContext {
            programs: &programs,
            patches: &patches,
            traces: &traces,
            profiles: &profiles,
            weak,
            strong,
        },
        workflow_id,
        1,
    )
    .await
    .unwrap();

    assert_eq!(installed, "v0002");
    assert_eq!(programs.latest_version(workflow_id).unwrap(), "v0002");
    let _ = fs::remove_dir_all(state_path);
}

struct PatchReturningStrong {
    patch: cps_llm_demo::program::ProgramPatch,
}

#[async_trait]
impl EffectHandler for PatchReturningStrong {
    async fn handle(&self, request: HandlerRequest) -> Result<HandlerDecision> {
        if request
            .effect
            .model_task_name()
            .is_some_and(|name| name == "optimize_program_patch")
        {
            return Ok(HandlerDecision::ReturnProgramPatch {
                patch: self.patch.clone(),
                rationale: "test optimizer patch".to_owned(),
            });
        }
        FixtureModelHandler::strong().handle(request).await
    }
}

#[test]
fn frame_encoder_stores_large_values_and_enforces_budget() {
    let state_path = temp_state_dir();
    let state = StateDir::new(state_path.clone());
    let workflow_id = "generic_workflow";
    let mut env = Map::new();
    env.insert("large".to_owned(), json!("x".repeat(4096)));
    let continuation = Continuation {
        continuation_id: "k-large".to_owned(),
        boundary_id: "b1".to_owned(),
        program_id: "p1".to_owned(),
        stack: vec![RuntimeFrame {
            function: "main".to_owned(),
            pc: 1,
            env,
            return_to: None,
        }],
        resume_var: Some("out".to_owned()),
        resume_pc: 2,
        expected_schema: json!({}),
        fuel_remaining: 10,
        effect_depth: 0,
    };
    let frame = EffectFrame {
        effect_id: "e1".to_owned(),
        boundary_id: "b1".to_owned(),
        reason: "test".to_owned(),
        failed_effect: None,
        failed_instruction: None,
        continuation: continuation.clone(),
        observations: Vec::new(),
        allowed_decisions: vec![AllowedDecision::ReturnValue],
    };

    let encoder = FileEffectFrameEncoder::new(
        state.clone(),
        workflow_id,
        FrameEncodingConfig {
            max_inline_value_bytes: 64,
            max_model_visible_frame_bytes: 16_384,
        },
    );
    let encoded = encoder.encode(&frame).unwrap();
    assert!(encoded.encoded_bytes < encoded.original_bytes);
    assert!(
        encoded.model_visible_frame.continuation.stack[0].env["large"]["$value_ref"]
            .as_str()
            .unwrap()
            .starts_with("valuestore://")
    );
    let stored = FileContinuationStore::new(state.clone(), workflow_id)
        .get("k-large")
        .unwrap();
    assert_eq!(stored, continuation);

    let value_store = FileValueStore::new(state.clone(), workflow_id);
    let value_ref = value_store.put(&json!({ "a": "b" })).unwrap();
    assert_eq!(value_store.get(&value_ref).unwrap(), json!({ "a": "b" }));

    let too_small = FileEffectFrameEncoder::new(
        state,
        workflow_id,
        FrameEncodingConfig {
            max_inline_value_bytes: 64,
            max_model_visible_frame_bytes: 64,
        },
    );
    assert!(too_small.encode(&frame).is_err());
    let _ = fs::remove_dir_all(state_path);
}

#[tokio::test]
async fn patch_can_insert_validator_apply() {
    let mut functions = BTreeMap::new();
    functions.insert(
        "main".to_owned(),
        FunctionDef {
            params: vec!["event".to_owned()],
            output_schema: json!({ "type": "object" }),
            body: vec![Instr::Return {
                value: JsonExpr::Var {
                    name: "event".to_owned(),
                },
            }],
        },
    );
    let base = Program {
        program_id: "validator_patch_base".to_owned(),
        version: "v0001".to_owned(),
        entry: "main".to_owned(),
        input_schema: json!({ "type": "object" }),
        output_schema: json!({ "type": "object" }),
        functions,
        allowed_effects: vec![EffectPermission::LocalTool {
            tool_name: "validator_apply".to_owned(),
        }],
    };
    let patch = cps_llm_demo::program::ProgramPatch {
        target_program_id: base.program_id.clone(),
        patch_id: "add_validator".to_owned(),
        rationale: "insert generic validator".to_owned(),
        operations: vec![
            cps_llm_demo::program::PatchOp::InsertInstruction {
                function: "main".to_owned(),
                pc: 0,
                instr: Instr::Perform {
                    out: "validated".to_owned(),
                    effect: cps_llm_demo::program::EffectCall::LocalTool {
                        tool_name: "validator_apply".to_owned(),
                        args_schema: json!({ "type": "object" }),
                    },
                    input: JsonExpr::Object {
                        fields: BTreeMap::from([
                            (
                                "value".to_owned(),
                                JsonExpr::Var {
                                    name: "event".to_owned(),
                                },
                            ),
                            (
                                "validators".to_owned(),
                                JsonExpr::Literal {
                                    value: json!([
                                        {
                                            "validator_id": "has_a",
                                            "predicates": [
                                                { "op": "field_exists", "path": ["a"] }
                                            ]
                                        }
                                    ]),
                                },
                            ),
                        ]),
                    },
                    expected_schema: json!({ "type": "object" }),
                    acceptance: cps_llm_demo::program::AcceptancePolicy {
                        min_confidence: Some(1.0),
                        require_schema_valid: true,
                        on_failure: cps_llm_demo::program::FailureHandler::Abort {
                            reason: "validator failed".to_owned(),
                        },
                    },
                },
            },
            cps_llm_demo::program::PatchOp::InsertInstruction {
                function: "main".to_owned(),
                pc: 1,
                instr: Instr::Branch {
                    condition: GuardExpr::FieldIsTruthy {
                        var: "validated".to_owned(),
                        path: vec!["passed".to_owned()],
                    },
                    then_pc: 2,
                    else_pc: 2,
                },
            },
        ],
    };
    let patched = validate_patch(&base, &patch).unwrap();
    let runtime = Runtime::new(
        FixtureModelHandler::weak(),
        FixtureModelHandler::strong(),
        TraceCollector::default(),
    );
    let output = runtime
        .run_program(patched, json!({ "a": "x" }))
        .await
        .unwrap();

    assert_eq!(output, json!({ "a": "x" }));
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
