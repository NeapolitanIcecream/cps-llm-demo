use std::collections::BTreeMap;
use std::io::Cursor;
use std::sync::{Arc, Mutex};

use anyhow::{Result, anyhow};
use async_trait::async_trait;
use cps_llm_demo::effects::{HandlerDecision, HandlerRequest};
use cps_llm_demo::models::EffectHandler;
use cps_llm_demo::program::{
    AcceptancePolicy, EffectCall, EffectPermission, FailureHandler, FunctionDef, GuardExpr,
    GuardFail, Instr, JsonExpr, PatchOp, Program, ProgramPatch,
};
use cps_llm_demo::runtime::{RunContext, Runtime};
use cps_llm_demo::trace::{TraceCollector, TraceEvent};
use cps_llm_demo::validator::{validate_patch, validate_program};
use cps_llm_demo::value_demo::continuation_compaction::{ContinuationCompactionConfig, ValueStore};
use cps_llm_demo::value_demo::event_source::{EventSource, JsonlEventSource};
use cps_llm_demo::value_demo::fast_path::{
    FastPathApplyInput, FastPathSpec, PredicateSpec, TemplateExpr, apply_fast_paths,
};
use cps_llm_demo::value_demo::local_tools::{LocalTool, LocalToolRegistry};
use cps_llm_demo::value_demo::metrics::{MetricsCollector, RunMode};
use cps_llm_demo::value_demo::patch_evaluator::PatchEvaluator;
use cps_llm_demo::value_demo::program_registry::ProgramRegistry;
use cps_llm_demo::value_demo::state_dir::StateDir;
use serde_json::{Value, json};

#[derive(Clone, Default)]
struct RecordingHandler {
    calls: Arc<Mutex<Vec<HandlerRequest>>>,
    decisions: Arc<Mutex<Vec<HandlerDecision>>>,
}

impl RecordingHandler {
    fn with_decisions(decisions: Vec<HandlerDecision>) -> Self {
        Self {
            calls: Arc::new(Mutex::new(Vec::new())),
            decisions: Arc::new(Mutex::new(decisions.into_iter().rev().collect())),
        }
    }

    fn calls(&self) -> Vec<HandlerRequest> {
        self.calls.lock().unwrap().clone()
    }
}

#[async_trait]
impl EffectHandler for RecordingHandler {
    async fn handle(&self, request: HandlerRequest) -> Result<HandlerDecision> {
        self.calls.lock().unwrap().push(request.clone());
        if let Some(decision) = self.decisions.lock().unwrap().pop() {
            return Ok(decision);
        }
        match request.effect {
            EffectCall::ModelTask { task, .. } if task.name == "classify_notification_intent" => {
                Ok(return_value(json!({ "kind": "create_task" }), 0.95))
            }
            EffectCall::ModelTask { task, .. }
                if task.name == "extract_notification_action_draft" =>
            {
                let event = &request.input["event"];
                Ok(return_value(
                    json!({
                        "event_id": event["event_id"],
                        "kind": request.input["intent"]["kind"],
                        "title": event["payload"]["title"],
                        "datetime_hint": null
                    }),
                    0.95,
                ))
            }
            EffectCall::ModelTask { .. } | EffectCall::Think { .. } => {
                Ok(return_value(json!("ok"), 0.95))
            }
            _ => Err(anyhow!("unexpected effect")),
        }
    }
}

struct EchoTool;

#[async_trait]
impl LocalTool for EchoTool {
    fn name(&self) -> &'static str {
        "echo"
    }

    fn input_schema(&self) -> Value {
        json!({ "type": "string" })
    }

    fn output_schema(&self) -> Value {
        json!({ "type": "string" })
    }

    async fn call(&self, input: Value) -> Result<Value> {
        Ok(input)
    }
}

struct BadOutputTool;

#[async_trait]
impl LocalTool for BadOutputTool {
    fn name(&self) -> &'static str {
        "bad_output"
    }

    fn input_schema(&self) -> Value {
        json!({})
    }

    fn output_schema(&self) -> Value {
        json!({ "type": "string" })
    }

    async fn call(&self, _input: Value) -> Result<Value> {
        Ok(json!(42))
    }
}

#[test]
fn event_source_preserves_order_and_rejects_duplicates() {
    let raw = concat!(
        "{\"event_id\":\"n1\",\"payload\":{\"x\":1}}\n",
        "{\"event_id\":\"n2\",\"payload\":{\"x\":2}}\n"
    );
    let mut source = JsonlEventSource::new(Cursor::new(raw));
    assert_eq!(source.next_event().unwrap().unwrap().event_id, "n1");
    assert_eq!(source.next_event().unwrap().unwrap().event_id, "n2");
    assert!(source.next_event().unwrap().is_none());
    assert_eq!(source.event_count(), 2);

    let raw = concat!(
        "{\"event_id\":\"n1\",\"payload\":{}}\n",
        "{\"event_id\":\"n1\",\"payload\":{}}\n"
    );
    let mut source = JsonlEventSource::new(Cursor::new(raw));
    assert!(source.next_event().unwrap().is_some());
    assert!(
        source
            .next_event()
            .unwrap_err()
            .to_string()
            .contains("duplicate event_id")
    );
}

#[test]
fn state_dir_and_program_registry_persist_versions() {
    let state_dir = StateDir::open_or_create(temp_path("state-registry")).unwrap();
    assert!(state_dir.programs_dir().exists());
    assert!(state_dir.patches_dir().join("pending").exists());
    assert!(state_dir.value_store_dir("run-1").ends_with("run-1"));

    let registry = ProgramRegistry::new(state_dir);
    let program = pure_string_program("registry_demo", "1.0.0", "v1");
    registry.install_initial(&program, "run-1").unwrap();
    assert_eq!(registry.latest_version("registry_demo").unwrap(), "1.0.0");
    assert_eq!(registry.load_latest("registry_demo").unwrap(), program);

    let patched = Program {
        version: "1.0.0+patch-demo".to_owned(),
        ..pure_string_program("registry_demo", "1.0.0", "v2")
    };
    registry
        .install_patched(&patched, "patch-demo", "run-2")
        .unwrap();
    assert_eq!(
        registry.latest_version("registry_demo").unwrap(),
        "1.0.0+patch-demo"
    );
    assert_eq!(
        registry.load_version("registry_demo", "1.0.0").unwrap(),
        program
    );
}

#[test]
fn metrics_counts_trace_events_and_p95() {
    let mut collector = MetricsCollector::new("run-metrics", RunMode::ValueDemo);
    for event in [
        TraceEvent {
            event: "handler_request".to_owned(),
            event_id: "n1".to_owned(),
            detail: json!({ "handler": "weak_model", "effect": "model_task", "strength": "weak" }),
        },
        TraceEvent {
            event: "handler_request".to_owned(),
            event_id: "n1".to_owned(),
            detail: json!({ "handler": "strong_model", "effect": "think" }),
        },
        TraceEvent {
            event: "fast_path_result".to_owned(),
            event_id: "n1".to_owned(),
            detail: json!({ "hit": true }),
        },
        TraceEvent {
            event: "continuation_compacted".to_owned(),
            event_id: "n1".to_owned(),
            detail: json!({ "public_bytes": 10 }),
        },
        TraceEvent {
            event: "continuation_compacted".to_owned(),
            event_id: "n2".to_owned(),
            detail: json!({ "public_bytes": 20 }),
        },
    ] {
        collector.observe_trace_event(&event);
    }
    let metrics = collector.finish();
    assert_eq!(metrics.weak_model_calls, 1);
    assert_eq!(metrics.strong_think_calls, 1);
    assert_eq!(metrics.fast_path_hits, 1);
    assert_eq!(metrics.continuation_frame_bytes_p95, Some(20));
    assert_eq!(metrics.strong_think_rate(), 0.0);
}

#[tokio::test]
async fn branch_and_jump_execute_forward_targets() {
    let runtime = Runtime::new(
        RecordingHandler::default(),
        RecordingHandler::default(),
        TraceCollector::default(),
    );
    let true_output = runtime
        .run_program(branch_program(), json!({ "flag": true }))
        .await
        .unwrap();
    let false_output = runtime
        .run_program(branch_program(), json!({ "flag": false }))
        .await
        .unwrap();
    let jump_output = runtime
        .run_program(jump_program(), json!({}))
        .await
        .unwrap();

    assert_eq!(true_output, json!("then"));
    assert_eq!(false_output, json!("else"));
    assert_eq!(jump_output, json!("jumped"));
}

#[test]
fn backward_and_out_of_range_jumps_are_rejected() {
    let mut program = jump_program();
    let main = program.functions.get_mut("main").unwrap();
    main.body[0] = Instr::Jump { pc: 0 };
    assert!(
        validate_program(&program)
            .unwrap_err()
            .to_string()
            .contains("forward-only")
    );

    let mut program = branch_program();
    let main = program.functions.get_mut("main").unwrap();
    main.body[0] = Instr::Branch {
        condition: GuardExpr::JsonPathExists {
            var: "request".to_owned(),
            path: vec!["flag".to_owned()],
        },
        then_pc: 1,
        else_pc: 99,
    };
    assert!(
        validate_program(&program)
            .unwrap_err()
            .to_string()
            .contains("outside function")
    );
}

#[tokio::test]
async fn local_tool_registry_executes_and_validates_tools() {
    let mut registry = LocalToolRegistry::empty();
    registry.register(Arc::new(EchoTool));
    let runtime = Runtime::with_local_tools(
        RecordingHandler::default(),
        RecordingHandler::default(),
        TraceCollector::default(),
        registry,
    );

    let output = runtime
        .run_program(local_tool_program("echo"), json!({}))
        .await
        .unwrap();
    assert_eq!(output, json!("hello"));

    let runtime = Runtime::new(
        RecordingHandler::default(),
        RecordingHandler::default(),
        TraceCollector::default(),
    );
    assert!(
        runtime
            .run_program(local_tool_program("echo"), json!({}))
            .await
            .unwrap_err()
            .to_string()
            .contains("not registered")
    );

    let mut registry = LocalToolRegistry::empty();
    registry.register(Arc::new(BadOutputTool));
    let runtime = Runtime::with_local_tools(
        RecordingHandler::default(),
        RecordingHandler::default(),
        TraceCollector::default(),
        registry,
    );
    assert!(
        runtime
            .run_program(local_tool_program("bad_output"), json!({}))
            .await
            .unwrap_err()
            .to_string()
            .contains("output failed registered output_schema")
    );
}

#[test]
fn fast_path_apply_hits_misses_and_rejects_invalid_specs() {
    let mut fields = BTreeMap::new();
    fields.insert(
        "event_id".to_owned(),
        TemplateExpr::FromPath {
            path: vec!["event_id".to_owned()],
        },
    );
    fields.insert(
        "kind".to_owned(),
        TemplateExpr::Literal {
            value: json!("create_task"),
        },
    );
    let spec = FastPathSpec {
        fast_path_id: "buildbot".to_owned(),
        predicates: vec![PredicateSpec::JsonPathEquals {
            path: vec!["payload".to_owned(), "app".to_owned()],
            value: json!("BuildBot"),
        }],
        output_template: TemplateExpr::Object { fields },
        output_schema: json!({
            "type": "object",
            "additionalProperties": false,
            "properties": {
                "event_id": { "type": "string" },
                "kind": { "type": "string" }
            },
            "required": ["event_id", "kind"]
        }),
    };

    let hit = apply_fast_paths(FastPathApplyInput {
        payload: json!({ "event_id": "n1", "payload": { "app": "BuildBot" } }),
        fast_paths: vec![spec.clone()],
    })
    .unwrap();
    assert!(hit.hit);
    assert_eq!(hit.value.unwrap()["kind"], "create_task");

    let miss = apply_fast_paths(FastPathApplyInput {
        payload: json!({ "event_id": "n2", "payload": { "app": "Mail" } }),
        fast_paths: vec![spec],
    })
    .unwrap();
    assert!(!miss.hit);

    let invalid_regex = FastPathSpec {
        fast_path_id: "bad".to_owned(),
        predicates: vec![PredicateSpec::RegexMatch {
            path: vec!["title".to_owned()],
            pattern: "[".to_owned(),
        }],
        output_template: TemplateExpr::Null,
        output_schema: json!({}),
    };
    assert!(
        apply_fast_paths(FastPathApplyInput {
            payload: json!({}),
            fast_paths: vec![invalid_regex],
        })
        .unwrap_err()
        .to_string()
        .contains("invalid fast path")
    );
}

#[tokio::test]
async fn patch_evaluator_accepts_fast_path_improvement_and_registry_installs_it() {
    let program: Program = serde_json::from_str(include_str!(
        "../examples/notification_triage.v1.program.json"
    ))
    .unwrap();
    let patch: ProgramPatch = serde_json::from_str(include_str!(
        "../examples/patches/buildbot_fast_path.patch.json"
    ))
    .unwrap();
    let patched = validate_patch(&program, &patch).unwrap();
    assert!(patched.version.contains("patch-buildbot-fast-path"));

    let events = read_fixture_events(include_str!("../examples/notification_events.round2.jsonl"));
    let outcome = PatchEvaluator::default()
        .evaluate(
            &program,
            &patch,
            &events,
            RecordingHandler::default(),
            RecordingHandler::default(),
            LocalToolRegistry::with_default_tools(),
        )
        .await
        .unwrap();
    assert!(outcome.report.accepted, "{}", outcome.report.reason);
    let patched_program = outcome.patched_program.unwrap();
    assert!(
        outcome
            .report
            .patched_metrics
            .as_ref()
            .unwrap()
            .fast_path_hit_rate()
            > outcome
                .report
                .original_metrics
                .as_ref()
                .unwrap()
                .fast_path_hit_rate()
    );

    let state_dir = StateDir::open_or_create(temp_path("patch-install")).unwrap();
    let registry = ProgramRegistry::new(state_dir);
    registry.install_initial(&program, "run-1").unwrap();
    registry
        .install_patched(&patched_program, &patch.patch_id, "run-2")
        .unwrap();
    assert_eq!(
        registry.latest_version("notification_triage").unwrap(),
        patched_program.version
    );
}

#[tokio::test]
async fn continuation_compaction_sends_small_public_frame_and_resumes_full_state() {
    let large_payload = "x".repeat(64 * 1024);
    let state_dir = StateDir::open_or_create(temp_path("compaction")).unwrap();
    let value_store =
        ValueStore::new("run-compact", state_dir.value_store_dir("run-compact")).unwrap();
    let strong = RecordingHandler::with_decisions(vec![return_value(json!("ok"), 0.99)]);
    let trace = TraceCollector::default();
    let runtime = Runtime::new(RecordingHandler::default(), strong.clone(), trace.clone());

    let output = runtime
        .run_program_with_report(
            guard_think_program(),
            json!({ "large": large_payload }),
            Some(RunContext {
                run_id: "run-compact".to_owned(),
                event_id: Some("large-1".to_owned()),
                value_store: Some(value_store),
                continuation_compaction: Some(ContinuationCompactionConfig {
                    max_public_continuation_frame_bytes: 4096,
                    max_inline_value_bytes: 128,
                }),
            }),
        )
        .await
        .unwrap()
        .output;

    assert_eq!(output, json!("ok"));
    let strong_calls = strong.calls();
    let frame = strong_calls[0].effect_frame.as_ref().unwrap();
    assert!(serde_json::to_vec(frame).unwrap().len() <= 4096);
    assert!(
        trace.events().iter().any(|event| {
            event.event == "continuation_compacted"
                && event.detail["full_bytes"].as_u64().unwrap()
                    > event.detail["public_bytes"].as_u64().unwrap()
                && event.detail["stored_refs"].as_u64().unwrap() > 0
        }),
        "expected continuation_compacted trace with stored refs"
    );
}

#[tokio::test]
async fn run_program_with_report_exposes_pending_patches() {
    let patch = ProgramPatch {
        target_program_id: "patchable".to_owned(),
        patch_id: "patch-return-null".to_owned(),
        rationale: "test patch".to_owned(),
        operations: vec![PatchOp::ReplaceInstruction {
            function: "main".to_owned(),
            pc: 1,
            instr: Instr::Return {
                value: JsonExpr::Literal { value: Value::Null },
            },
        }],
    };
    let strong = RecordingHandler::with_decisions(vec![HandlerDecision::ReturnProgramPatch {
        patch: patch.clone(),
        rationale: "propose patch".to_owned(),
    }]);
    let runtime = Runtime::new(
        RecordingHandler::default(),
        strong,
        TraceCollector::default(),
    );
    let report = runtime
        .run_program_with_report(patchable_program(), json!({}), None)
        .await
        .unwrap();

    assert_eq!(report.output, Value::Null);
    assert_eq!(report.pending_patches, vec![patch]);
}

#[test]
fn notification_fixture_program_and_patch_validate() {
    let program: Program = serde_json::from_str(include_str!(
        "../examples/notification_triage.v1.program.json"
    ))
    .unwrap();
    let patch: ProgramPatch = serde_json::from_str(include_str!(
        "../examples/patches/buildbot_fast_path.patch.json"
    ))
    .unwrap();
    validate_program(&program).unwrap();
    validate_patch(&program, &patch).unwrap();
}

fn return_value(value: Value, confidence: f32) -> HandlerDecision {
    HandlerDecision::ReturnValue {
        value,
        confidence,
        rationale: "test".to_owned(),
    }
}

fn accept() -> AcceptancePolicy {
    AcceptancePolicy {
        min_confidence: Some(0.8),
        require_schema_valid: true,
        on_failure: FailureHandler::Abort {
            reason: "not accepted".to_owned(),
        },
    }
}

fn pure_string_program(program_id: &str, version: &str, value: &str) -> Program {
    Program {
        program_id: program_id.to_owned(),
        version: version.to_owned(),
        entry: "main".to_owned(),
        input_schema: json!({}),
        output_schema: json!({ "type": "string" }),
        functions: BTreeMap::from([(
            "main".to_owned(),
            FunctionDef {
                params: vec!["request".to_owned()],
                output_schema: json!({ "type": "string" }),
                body: vec![Instr::Return {
                    value: JsonExpr::Literal {
                        value: json!(value),
                    },
                }],
            },
        )]),
        allowed_effects: Vec::new(),
    }
}

fn branch_program() -> Program {
    Program {
        program_id: "branch_demo".to_owned(),
        version: "1.0.0".to_owned(),
        entry: "main".to_owned(),
        input_schema: json!({ "type": "object" }),
        output_schema: json!({ "type": "string" }),
        functions: BTreeMap::from([(
            "main".to_owned(),
            FunctionDef {
                params: vec!["request".to_owned()],
                output_schema: json!({ "type": "string" }),
                body: vec![
                    Instr::Branch {
                        condition: GuardExpr::JsonPathEquals {
                            var: "request".to_owned(),
                            path: vec!["flag".to_owned()],
                            value: json!(true),
                        },
                        then_pc: 1,
                        else_pc: 2,
                    },
                    Instr::Return {
                        value: JsonExpr::Literal {
                            value: json!("then"),
                        },
                    },
                    Instr::Return {
                        value: JsonExpr::Literal {
                            value: json!("else"),
                        },
                    },
                ],
            },
        )]),
        allowed_effects: Vec::new(),
    }
}

fn jump_program() -> Program {
    Program {
        program_id: "jump_demo".to_owned(),
        version: "1.0.0".to_owned(),
        entry: "main".to_owned(),
        input_schema: json!({}),
        output_schema: json!({ "type": "string" }),
        functions: BTreeMap::from([(
            "main".to_owned(),
            FunctionDef {
                params: vec!["request".to_owned()],
                output_schema: json!({ "type": "string" }),
                body: vec![
                    Instr::Jump { pc: 2 },
                    Instr::Return {
                        value: JsonExpr::Literal {
                            value: json!("bad"),
                        },
                    },
                    Instr::Return {
                        value: JsonExpr::Literal {
                            value: json!("jumped"),
                        },
                    },
                ],
            },
        )]),
        allowed_effects: Vec::new(),
    }
}

fn local_tool_program(tool_name: &str) -> Program {
    Program {
        program_id: format!("local_tool_{tool_name}"),
        version: "1.0.0".to_owned(),
        entry: "main".to_owned(),
        input_schema: json!({}),
        output_schema: json!({ "type": "string" }),
        functions: BTreeMap::from([(
            "main".to_owned(),
            FunctionDef {
                params: vec!["request".to_owned()],
                output_schema: json!({ "type": "string" }),
                body: vec![
                    Instr::Perform {
                        out: "value".to_owned(),
                        effect: EffectCall::LocalTool {
                            tool_name: tool_name.to_owned(),
                            args_schema: json!({}),
                        },
                        input: JsonExpr::Literal {
                            value: json!("hello"),
                        },
                        expected_schema: json!({ "type": "string" }),
                        acceptance: accept(),
                    },
                    Instr::Return {
                        value: JsonExpr::Var {
                            name: "value".to_owned(),
                        },
                    },
                ],
            },
        )]),
        allowed_effects: vec![EffectPermission::LocalTool {
            tool_name: tool_name.to_owned(),
        }],
    }
}

fn guard_think_program() -> Program {
    Program {
        program_id: "guard_think_compaction".to_owned(),
        version: "1.0.0".to_owned(),
        entry: "main".to_owned(),
        input_schema: json!({}),
        output_schema: json!({ "type": "string" }),
        functions: BTreeMap::from([(
            "main".to_owned(),
            FunctionDef {
                params: vec!["request".to_owned()],
                output_schema: json!({ "type": "string" }),
                body: vec![
                    Instr::Guard {
                        condition: GuardExpr::VarExists {
                            name: "answer".to_owned(),
                        },
                        on_fail: GuardFail::Think {
                            reason: "need answer".to_owned(),
                        },
                    },
                    Instr::Return {
                        value: JsonExpr::Var {
                            name: "answer".to_owned(),
                        },
                    },
                ],
            },
        )]),
        allowed_effects: vec![EffectPermission::Think],
    }
}

fn patchable_program() -> Program {
    let mut program = guard_think_program();
    program.program_id = "patchable".to_owned();
    program.output_schema = json!({});
    program.functions.get_mut("main").unwrap().output_schema = json!({});
    program
}

fn read_fixture_events(raw: &str) -> Vec<cps_llm_demo::value_demo::event_source::EventEnvelope> {
    let mut source = JsonlEventSource::new(Cursor::new(raw));
    let mut events = Vec::new();
    while let Some(event) = source.next_event().unwrap() {
        events.push(event);
    }
    events
}

fn temp_path(prefix: &str) -> std::path::PathBuf {
    let path = std::env::temp_dir().join(format!("cps-llm-demo-{prefix}-{}", uuid::Uuid::new_v4()));
    let _ = std::fs::remove_dir_all(&path);
    path
}
