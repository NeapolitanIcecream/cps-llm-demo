use anyhow::Result;
use assert_cmd::Command;
use async_trait::async_trait;
use cps_llm_demo::effects::{
    AllowedDecision, Continuation, EffectFrame, EffectFrameEncoder, HandlerDecision,
    HandlerRequest, RuntimeFrame,
};
use cps_llm_demo::engine::event_source::{InMemoryEventSource, JsonlEventSource};
use cps_llm_demo::engine::run_coordinator::run_stream;
use cps_llm_demo::evaluation::strong_direct_baseline::baseline_strong_direct;
use cps_llm_demo::local_tools::{
    PredicateExpr, ValidatorInput, ValidatorSpec, apply_fast_path, apply_validators,
};
use cps_llm_demo::models::EffectHandler;
use cps_llm_demo::models::FixtureModelHandler;
use cps_llm_demo::observability::metrics::MetricsAccumulator;
use cps_llm_demo::optimizer::patch_installer::install_fixture_patch;
use cps_llm_demo::optimizer::patch_optimizer::{OptimizerContext, optimize_from_profile};
use cps_llm_demo::optimizer::patch_request::PatchRequest;
use cps_llm_demo::program::{
    AcceptancePolicy, EffectCall, EffectPermission, FailureHandler, FunctionDef, GuardExpr, Instr,
    JsonExpr, PatchOp, Program, ProgramPatch,
};
use cps_llm_demo::runtime::Runtime;
use cps_llm_demo::store::continuation_store::{
    FileContinuationStore, FileEffectFrameEncoder, FrameEncodingConfig,
};
use cps_llm_demo::store::metrics_store::{FileMetricsStore, RunMetrics};
use cps_llm_demo::store::patch_registry::{FilePatchRegistry, fixture_patch_metadata};
use cps_llm_demo::store::profile_store::FileProfileStore;
use cps_llm_demo::store::program_registry::{
    FileProgramRegistry, ProgramMetadata, ProgramSource, fixture_program_metadata,
};
use cps_llm_demo::store::state_dir::StateDir;
use cps_llm_demo::store::trace_store::FileTraceStore;
use cps_llm_demo::store::value_store::FileValueStore;
use cps_llm_demo::trace::{TraceCollector, TraceEvent};
use cps_llm_demo::validator::validate_patch;
use cps_llm_demo::validator::validate_program;
use serde_json::Map;
use serde_json::{Value, json};
use std::collections::{BTreeMap, VecDeque};
use std::fs;
use std::path::Path;
use std::sync::{Arc, Mutex};

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

#[test]
fn metrics_count_uncaptured_acceptance_from_effect_accepted_events() {
    let mut metrics = test_run_metrics();
    let mut accumulator = MetricsAccumulator::default();
    let events = vec![
        TraceEvent {
            event: "perform_effect".to_owned(),
            event_id: "direct".to_owned(),
            detail: json!({
                "function": "main",
                "pc": 0,
                "out": "draft",
                "effect": "model_task",
                "strength": "weak"
            }),
        },
        TraceEvent {
            event: "effect_accepted".to_owned(),
            event_id: "direct".to_owned(),
            detail: json!({
                "effect": "model_task",
                "captured": false
            }),
        },
        TraceEvent {
            event: "capture_continuation".to_owned(),
            event_id: "guard-1".to_owned(),
            detail: json!({
                "continuation_id": "k-guard-1",
                "failed_instruction_op": "guard",
                "failed_effect_kind": "think"
            }),
        },
        TraceEvent {
            event: "perform_effect".to_owned(),
            event_id: "aborted".to_owned(),
            detail: json!({
                "function": "main",
                "pc": 1,
                "out": "draft",
                "effect": "model_task",
                "strength": "weak"
            }),
        },
        TraceEvent {
            event: "program_aborted".to_owned(),
            event_id: "aborted".to_owned(),
            detail: json!({
                "reason": "perform failed"
            }),
        },
        TraceEvent {
            event: "perform_effect".to_owned(),
            event_id: "captured".to_owned(),
            detail: json!({
                "function": "main",
                "pc": 2,
                "out": "draft",
                "effect": "model_task",
                "strength": "weak"
            }),
        },
        TraceEvent {
            event: "capture_continuation".to_owned(),
            event_id: "captured".to_owned(),
            detail: json!({
                "continuation_id": "k-perform",
                "failed_instruction_op": "perform",
                "failed_effect_kind": "model_task"
            }),
        },
        TraceEvent {
            event: "effect_accepted".to_owned(),
            event_id: "captured".to_owned(),
            detail: json!({
                "effect": "model_task",
                "captured": true
            }),
        },
        TraceEvent {
            event: "capture_continuation".to_owned(),
            event_id: "guard-2".to_owned(),
            detail: json!({
                "continuation_id": "k-guard-2",
                "failed_instruction_op": "guard",
                "failed_effect_kind": "think"
            }),
        },
    ];

    accumulator.update_from_trace(&mut metrics, &events);
    accumulator.finalize(&mut metrics);

    assert_eq!(metrics.effects_total, 3);
    assert_eq!(metrics.effects_captured, 3);
    assert_eq!(metrics.effects_accepted_without_capture, 1);
}

#[test]
fn metrics_include_compile_effects_in_estimated_model_calls() {
    let mut metrics = test_run_metrics();
    let mut accumulator = MetricsAccumulator::default();
    let events = vec![
        TraceEvent {
            event: "handler_request".to_owned(),
            event_id: "weak".to_owned(),
            detail: json!({
                "handler": "weak_model",
                "effect": "model_task"
            }),
        },
        TraceEvent {
            event: "handler_request".to_owned(),
            event_id: "strong-task".to_owned(),
            detail: json!({
                "handler": "strong_model",
                "effect": "model_task"
            }),
        },
        TraceEvent {
            event: "handler_request".to_owned(),
            event_id: "think".to_owned(),
            detail: json!({
                "handler": "strong_model",
                "effect": "think"
            }),
        },
        TraceEvent {
            event: "handler_request".to_owned(),
            event_id: "compile".to_owned(),
            detail: json!({
                "handler": "strong_model",
                "effect": "compile_program"
            }),
        },
    ];

    accumulator.update_from_trace(&mut metrics, &events);
    accumulator.finalize(&mut metrics);

    assert_eq!(metrics.program_compile_calls, 1);
    assert_eq!(metrics.estimated_model_calls, 4);
}

#[tokio::test]
async fn strong_direct_baseline_counts_handler_abort_as_failed_event() {
    let state_path = temp_state_dir();
    let state = StateDir::new(state_path.clone());
    let workflow_id = "baseline_abort_accounting";
    let event_source = InMemoryEventSource::new(vec![
        json!({
            "event_id": "ok-1",
            "_fixture_model": {
                "strong": {
                    "value": { "ok": true }
                }
            }
        }),
        json!({
            "event_id": "abort-1",
            "_fixture_model": {
                "strong": {
                    "abort": "model refused"
                }
            }
        }),
    ]);

    let summary = baseline_strong_direct(
        state.clone(),
        workflow_id,
        "return an object".to_owned(),
        event_source,
        Arc::new(FixtureModelHandler::strong()),
    )
    .await
    .unwrap();
    let metrics = FileMetricsStore::new(state)
        .read(workflow_id, &summary.run_id)
        .unwrap();

    assert_eq!(summary.events_total, 2);
    assert_eq!(metrics.events_total, 2);
    assert_eq!(metrics.events_succeeded, 1);
    assert_eq!(metrics.events_failed, 1);
    assert_eq!(metrics.strong_model_task_calls, 2);
    assert_eq!(metrics.estimated_model_calls, 2);

    let _ = fs::remove_dir_all(state_path);
}

#[test]
fn patch_registry_rejects_unsafe_patch_ids_before_writing() {
    let state_path = temp_state_dir();
    let state = StateDir::new(state_path.clone());
    let patches = FilePatchRegistry::new(state.clone());
    let workflow_id = "notification_triage";

    for patch_id in [
        "../../metrics/evil",
        "nested/evil",
        r"nested\evil",
        "",
        ".",
        "..",
    ] {
        let patch = registry_test_patch(patch_id);
        let err = patches
            .record_proposed(
                workflow_id,
                patch.clone(),
                fixture_patch_metadata(workflow_id, &patch.patch_id, "v0001", &patch.rationale),
            )
            .unwrap_err();
        assert!(
            err.to_string().contains("patch_id"),
            "unexpected error for {patch_id:?}: {err}"
        );
    }

    assert!(
        !state
            .workflow_dir(workflow_id)
            .unwrap()
            .join("metrics/evil.json")
            .exists()
    );
    assert!(
        !state
            .workflow_dir(workflow_id)
            .unwrap()
            .join("patches/proposed/nested/evil.json")
            .exists()
    );
    assert!(patches.list_proposed(workflow_id).unwrap().is_empty());

    let _ = fs::remove_dir_all(state_path);
}

#[test]
fn patch_registry_preserves_valid_patch_id_status_records() {
    let state_path = temp_state_dir();
    let state = StateDir::new(state_path.clone());
    let patches = FilePatchRegistry::new(state.clone());
    let workflow_id = "notification_triage";
    let patch = registry_test_patch("fixture_fast_path_v1");
    let metadata = fixture_patch_metadata(workflow_id, &patch.patch_id, "v0001", &patch.rationale);

    patches
        .record_proposed(workflow_id, patch.clone(), metadata.clone())
        .unwrap();
    let proposed = patches.list_proposed(workflow_id).unwrap();
    assert_eq!(proposed.len(), 1);
    assert_eq!(proposed[0].patch.patch_id, patch.patch_id);
    assert_eq!(proposed[0].metadata.patch_id, patch.patch_id);

    patches
        .mark_validated(workflow_id, patch.clone(), metadata.clone())
        .unwrap();
    patches
        .mark_rejected(
            workflow_id,
            patch.clone(),
            metadata.clone(),
            "not enough improvement",
        )
        .unwrap();
    patches
        .mark_installed(workflow_id, patch.clone(), metadata, "v0002")
        .unwrap();

    for status in ["proposed", "validated", "rejected", "installed"] {
        assert!(
            state
                .workflow_dir(workflow_id)
                .unwrap()
                .join("patches")
                .join(status)
                .join("fixture_fast_path_v1.json")
                .exists(),
            "missing {status} record"
        );
    }

    let _ = fs::remove_dir_all(state_path);
}

#[test]
fn state_dir_rejects_unsafe_workflow_ids_before_writing() {
    let state_path = temp_state_dir();
    let state = StateDir::new(state_path.clone());
    let outside_name = format!("cps-workflow-escape-{}", uuid::Uuid::new_v4());
    let traversal = format!("../../{outside_name}");

    for workflow_id in [
        traversal.as_str(),
        "nested/workflow",
        r"nested\workflow",
        "",
        ".",
        "..",
    ] {
        let err = state.ensure_workflow_layout(workflow_id).unwrap_err();
        assert!(
            err.to_string().contains("workflow_id"),
            "unexpected error for {workflow_id:?}: {err}"
        );
    }

    assert!(!state_path.parent().unwrap().join(&outside_name).exists());

    let _ = fs::remove_dir_all(state_path);
}

#[test]
fn init_workflow_rejects_existing_workflow_without_mixing_stale_state() {
    let state_path = temp_state_dir();
    let state = StateDir::new(state_path.clone());
    let workflow_id = "reinit_guard";
    let mut initial_program: Program = serde_json::from_str(include_str!(
        "../examples/notification_triage.v1.program.json"
    ))
    .unwrap();
    initial_program.program_id = "initial_program".to_owned();
    let programs = FileProgramRegistry::new(state.clone());
    programs
        .init_workflow(
            workflow_id,
            initial_program.clone(),
            fixture_program_metadata(workflow_id, &initial_program),
        )
        .unwrap();

    let mut metrics = RunMetrics::new(
        "old-run".to_owned(),
        workflow_id.to_owned(),
        "stream".to_owned(),
        initial_program.program_id.clone(),
        "v0001".to_owned(),
        "started".to_owned(),
    );
    metrics.events_total = 1;
    FileMetricsStore::new(state.clone())
        .write(&metrics)
        .unwrap();

    let traces = FileTraceStore::new(state.clone());
    traces
        .append_events(
            workflow_id,
            "old-run",
            &[TraceEvent {
                event: "stream_event".to_owned(),
                event_id: "old-event".to_owned(),
                detail: json!({ "event": { "event_id": "old-event", "text": "stale" } }),
            }],
        )
        .unwrap();

    let profiles = FileProfileStore::new(state.clone());
    profiles
        .update_from_trace(
            workflow_id,
            &[TraceEvent {
                event: "capture_continuation".to_owned(),
                event_id: "old-capture".to_owned(),
                detail: json!({
                    "continuation_id": "stale-k",
                    "program_id": initial_program.program_id.clone(),
                    "program_version": "v0001",
                    "function": "main",
                    "pc": 0,
                    "failed_instruction_op": "perform",
                    "failed_effect_kind": "model_task",
                    "expected_schema": { "type": "object" },
                    "observations": []
                }),
            }],
        )
        .unwrap();

    let patches = FilePatchRegistry::new(state.clone());
    let patch = registry_test_patch("stale_patch");
    patches
        .record_proposed(
            workflow_id,
            patch.clone(),
            fixture_patch_metadata(workflow_id, &patch.patch_id, "v0001", &patch.rationale),
        )
        .unwrap();

    let continuation_store = FileContinuationStore::new(state.clone(), workflow_id);
    let continuation = Continuation {
        continuation_id: "stale-k".to_owned(),
        boundary_id: "boundary".to_owned(),
        program_id: "initial_program".to_owned(),
        stack: vec![RuntimeFrame {
            function: "main".to_owned(),
            pc: 0,
            env: Map::new(),
            return_to: None,
        }],
        resume_var: None,
        resume_pc: 0,
        expected_schema: json!({ "type": "null" }),
        fuel_remaining: 1,
        effect_depth: 0,
    };
    continuation_store.put(&continuation).unwrap();
    let value_store = FileValueStore::new(state.clone(), workflow_id);
    let value_ref = value_store.put(&json!({ "stale": true })).unwrap();

    let mut new_program = initial_program.clone();
    new_program.program_id = "new_initial_program".to_owned();
    let err = programs
        .init_workflow(
            workflow_id,
            new_program,
            fixture_program_metadata(workflow_id, &initial_program),
        )
        .unwrap_err();

    assert!(err.to_string().contains("already exists"));
    assert_eq!(programs.latest_version(workflow_id).unwrap(), "v0001");
    assert_eq!(
        programs.load_latest(workflow_id).unwrap().program_id,
        "initial_program"
    );
    assert_eq!(
        FileMetricsStore::new(state.clone())
            .list(workflow_id)
            .unwrap()
            .len(),
        1
    );
    assert_eq!(traces.read_run(workflow_id, "old-run").unwrap().len(), 1);
    assert_eq!(
        profiles
            .load(workflow_id)
            .unwrap()
            .failure_fingerprints
            .len(),
        1
    );
    assert_eq!(patches.list_proposed(workflow_id).unwrap().len(), 1);
    assert_eq!(continuation_store.get("stale-k").unwrap(), continuation);
    assert_eq!(
        value_store.get(&value_ref).unwrap(),
        json!({ "stale": true })
    );

    let _ = fs::remove_dir_all(state_path);
}

#[test]
fn profile_store_records_successful_probe_for_failure_fingerprint() {
    let state_path = temp_state_dir();
    let store = FileProfileStore::new(StateDir::new(state_path.clone()));
    store
        .update_from_trace(
            "generic_workflow",
            &[
                TraceEvent {
                    event: "capture_continuation".to_owned(),
                    event_id: "e1".to_owned(),
                    detail: json!({
                        "continuation_id": "k1",
                        "program_id": "p1",
                        "program_version": "v0001",
                        "function": "main",
                        "pc": 3,
                        "failed_effect_kind": "think",
                        "expected_schema": { "type": "object" },
                        "observations": [{ "schema_valid": false, "value": { "a": "redacted" } }]
                    }),
                },
                TraceEvent {
                    event: "request_nested_effect".to_owned(),
                    event_id: "e1".to_owned(),
                    detail: json!({
                        "to_handler": "weak_model",
                        "to_effect": "model_task"
                    }),
                },
                TraceEvent {
                    event: "nested_effect_result".to_owned(),
                    event_id: "e1".to_owned(),
                    detail: json!({
                        "schema_valid": true
                    }),
                },
            ],
        )
        .unwrap();

    let profile = store.load("generic_workflow").unwrap();
    let failure = profile.failure_fingerprints.values().next().unwrap();
    assert_eq!(failure.count, 1);
    assert_eq!(failure.successful_probes.len(), 1);
    assert_eq!(
        failure.successful_probes[0].probe_id,
        "weak_model:model_task"
    );
    assert_eq!(failure.successful_probes[0].success_count, 1);

    let _ = fs::remove_dir_all(state_path);
}

#[test]
fn profile_store_keeps_same_failure_shape_separate_by_program_version() {
    let state_path = temp_state_dir();
    let store = FileProfileStore::new(StateDir::new(state_path.clone()));

    store
        .update_from_trace(
            "generic_workflow",
            &[
                versioned_failure_capture_event(
                    "stale-v1",
                    "k-stale-v1",
                    "profile_program",
                    "v0001",
                ),
                versioned_failure_capture_event(
                    "latest-v2",
                    "k-latest-v2",
                    "profile_program",
                    "v0002",
                ),
            ],
        )
        .unwrap();

    let profile = store.load("generic_workflow").unwrap();
    assert_eq!(profile.failure_fingerprints.len(), 2);

    let mut failures = profile.failure_fingerprints.values().collect::<Vec<_>>();
    failures.sort_by(|left, right| {
        left.fingerprint
            .program_version
            .cmp(&right.fingerprint.program_version)
    });

    assert_eq!(failures[0].fingerprint.program_version, "v0001");
    assert_eq!(failures[0].count, 1);
    assert_eq!(failures[1].fingerprint.program_version, "v0002");
    assert_eq!(failures[1].count, 1);
    assert_ne!(
        failures[0].fingerprint.fingerprint_id,
        failures[1].fingerprint.fingerprint_id
    );

    let _ = fs::remove_dir_all(state_path);
}

#[test]
fn profile_store_counts_nested_probe_repair_as_one_accepted_effect_call() {
    let state_path = temp_state_dir();
    let store = FileProfileStore::new(StateDir::new(state_path.clone()));
    store
        .update_from_trace(
            "generic_workflow",
            &[
                TraceEvent {
                    event: "perform_effect".to_owned(),
                    event_id: "e1".to_owned(),
                    detail: json!({
                        "function": "main",
                        "pc": 0,
                        "out": "draft",
                        "effect": "model_task",
                        "strength": "weak",
                        "task": "classify_and_extract_action_draft"
                    }),
                },
                TraceEvent {
                    event: "handler_decision".to_owned(),
                    event_id: "e1".to_owned(),
                    detail: json!({
                        "handler": "weak_model",
                        "effect": "model_task",
                        "decision": "return_value",
                        "schema_valid": false,
                        "depth": 0
                    }),
                },
                TraceEvent {
                    event: "capture_continuation".to_owned(),
                    event_id: "e1".to_owned(),
                    detail: json!({
                        "continuation_id": "k1",
                        "program_id": "p1",
                        "program_version": "v0001",
                        "function": "main",
                        "pc": 0,
                        "failed_instruction_op": "perform",
                        "failed_effect_kind": "model_task",
                        "expected_schema": { "type": "object" },
                        "observations": [{ "schema_valid": false, "value": { "a": "redacted" } }]
                    }),
                },
                TraceEvent {
                    event: "handler_decision".to_owned(),
                    event_id: "e1".to_owned(),
                    detail: json!({
                        "handler": "strong_model",
                        "effect": "think",
                        "decision": "request_effect",
                        "schema_valid": null,
                        "depth": 1
                    }),
                },
                TraceEvent {
                    event: "request_nested_effect".to_owned(),
                    event_id: "e1".to_owned(),
                    detail: json!({
                        "from_handler": "strong_model",
                        "from_effect": "think",
                        "to_handler": "weak_model",
                        "to_effect": "model_task",
                        "mode": "reenter_handler",
                        "depth": 2
                    }),
                },
                TraceEvent {
                    event: "handler_decision".to_owned(),
                    event_id: "e1".to_owned(),
                    detail: json!({
                        "handler": "weak_model",
                        "effect": "model_task",
                        "decision": "return_value",
                        "schema_valid": true,
                        "depth": 2
                    }),
                },
                TraceEvent {
                    event: "nested_effect_result".to_owned(),
                    event_id: "e1".to_owned(),
                    detail: json!({
                        "mode": "reenter_handler",
                        "schema_valid": true,
                        "source": "weak_model",
                        "depth": 2
                    }),
                },
                TraceEvent {
                    event: "handler_decision".to_owned(),
                    event_id: "e1".to_owned(),
                    detail: json!({
                        "handler": "strong_model",
                        "effect": "think",
                        "decision": "return_value",
                        "schema_valid": true,
                        "depth": 1
                    }),
                },
                TraceEvent {
                    event: "effect_accepted".to_owned(),
                    event_id: "e1".to_owned(),
                    detail: json!({
                        "effect": "model_task",
                        "source": "strong_model",
                        "captured": true,
                        "continuation_id": "k1"
                    }),
                },
            ],
        )
        .unwrap();

    let profile = store.load("generic_workflow").unwrap();
    let stats = profile.effect_stats.get("model_task").unwrap();
    assert_eq!(stats.calls, 1);
    assert_eq!(stats.captures, 1);
    assert_eq!(stats.accepted, 1);
    let failure = profile.failure_fingerprints.values().next().unwrap();
    assert_eq!(
        failure.successful_probes[0].probe_id,
        "weak_model:model_task"
    );
    assert_eq!(failure.successful_probes[0].success_count, 1);

    let _ = fs::remove_dir_all(state_path);
}

#[test]
fn profile_store_does_not_attribute_later_top_level_probe_to_repaired_failure() {
    let state_path = temp_state_dir();
    let store = FileProfileStore::new(StateDir::new(state_path.clone()));
    store
        .update_from_trace(
            "generic_workflow",
            &[
                TraceEvent {
                    event: "capture_continuation".to_owned(),
                    event_id: "e1".to_owned(),
                    detail: json!({
                        "continuation_id": "k1",
                        "program_id": "p1",
                        "program_version": "v0001",
                        "function": "main",
                        "pc": 0,
                        "failed_instruction_op": "perform",
                        "failed_effect_kind": "model_task",
                        "expected_schema": { "type": "object" },
                        "observations": [{ "schema_valid": false, "value": { "a": "redacted" } }]
                    }),
                },
                TraceEvent {
                    event: "effect_accepted".to_owned(),
                    event_id: "e1".to_owned(),
                    detail: json!({
                        "effect": "model_task",
                        "source": "strong_model",
                        "captured": true,
                        "continuation_id": "k1"
                    }),
                },
                TraceEvent {
                    event: "resume_continuation".to_owned(),
                    event_id: "e1".to_owned(),
                    detail: json!({
                        "continuation_id": "k1",
                        "pc": 1,
                        "resume_var": "draft"
                    }),
                },
                TraceEvent {
                    event: "perform_effect".to_owned(),
                    event_id: "e2".to_owned(),
                    detail: json!({
                        "function": "main",
                        "pc": 2,
                        "out": "later",
                        "effect": "model_task",
                        "strength": "strong",
                        "task": "later_top_level_task"
                    }),
                },
                TraceEvent {
                    event: "request_nested_effect".to_owned(),
                    event_id: "e2".to_owned(),
                    detail: json!({
                        "from_handler": "strong_model",
                        "from_effect": "model_task",
                        "to_handler": "weak_model",
                        "to_effect": "model_task",
                        "mode": "reenter_handler",
                        "depth": 1
                    }),
                },
                TraceEvent {
                    event: "nested_effect_result".to_owned(),
                    event_id: "e2".to_owned(),
                    detail: json!({
                        "mode": "reenter_handler",
                        "schema_valid": true,
                        "source": "weak_model",
                        "depth": 1
                    }),
                },
            ],
        )
        .unwrap();

    let profile = store.load("generic_workflow").unwrap();
    let failure = profile.failure_fingerprints.values().next().unwrap();
    assert!(failure.successful_probes.is_empty());

    let _ = fs::remove_dir_all(state_path);
}

#[test]
fn profile_store_does_not_count_unrelated_guard_repair_as_effect_capture() {
    let state_path = temp_state_dir();
    let store = FileProfileStore::new(StateDir::new(state_path.clone()));
    store
        .update_from_trace(
            "generic_workflow",
            &[
                TraceEvent {
                    event: "perform_effect".to_owned(),
                    event_id: "e1".to_owned(),
                    detail: json!({
                        "function": "main",
                        "pc": 0,
                        "out": "draft",
                        "effect": "model_task",
                        "strength": "weak",
                        "task": "classify_and_extract_action_draft"
                    }),
                },
                TraceEvent {
                    event: "effect_accepted".to_owned(),
                    event_id: "e2".to_owned(),
                    detail: json!({
                        "effect": "model_task",
                        "source": "weak_model",
                        "captured": false
                    }),
                },
                TraceEvent {
                    event: "capture_continuation".to_owned(),
                    event_id: "e3".to_owned(),
                    detail: json!({
                        "continuation_id": "k1",
                        "program_id": "p1",
                        "program_version": "v0001",
                        "function": "main",
                        "pc": 2,
                        "failed_instruction_op": "guard",
                        "failed_effect_kind": "think",
                        "expected_schema": { "type": "object" },
                        "observations": []
                    }),
                },
            ],
        )
        .unwrap();

    let profile = store.load("generic_workflow").unwrap();
    let stats = profile.effect_stats.get("model_task").unwrap();
    assert_eq!(stats.calls, 1);
    assert_eq!(stats.captures, 0);
    assert_eq!(stats.accepted, 1);

    let _ = fs::remove_dir_all(state_path);
}

#[tokio::test]
async fn patch_install_rejects_schema_valid_sample_output_change() {
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

    let baseline_output = json!({
        "event_id": "sample-1",
        "kind": "ignore",
        "title": "Battery optimization completed",
        "datetime_hint": null
    });
    let sample_event = json!({
        "event_id": "sample-1",
        "source": "system",
        "text": "Battery optimization completed",
        "_fixture_fast_path": true,
        "_fixture_fast_value": baseline_output,
        "_fixture_model": {
            "weak": {
                "value": baseline_output,
                "confidence": 0.42
            },
            "strong": {
                "value": baseline_output,
                "confidence": 0.99
            }
        }
    });
    let traces = FileTraceStore::new(state.clone());
    traces
        .append_events(
            workflow_id,
            "round1",
            &[TraceEvent {
                event: "stream_event".to_owned(),
                event_id: "sample-1".to_owned(),
                detail: json!({ "event": sample_event }),
            }],
        )
        .unwrap();

    let patches = FilePatchRegistry::new(state.clone());
    let weak: Arc<dyn EffectHandler> = Arc::new(FixtureModelHandler::weak());
    let strong: Arc<dyn EffectHandler> = Arc::new(FixtureModelHandler::strong());
    let err = install_fixture_patch(
        &programs,
        &patches,
        &traces,
        workflow_id,
        schema_valid_wrong_fast_path_patch(),
        weak,
        strong,
    )
    .await
    .unwrap_err();

    assert!(err.to_string().contains("sampled output"));
    assert_eq!(programs.latest_version(workflow_id).unwrap(), "v0001");
    assert!(
        state
            .workflow_dir(workflow_id)
            .unwrap()
            .join("patches/rejected/bad_fast_path_output.json")
            .exists()
    );

    let _ = fs::remove_dir_all(state_path);
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
fn branch_target_rejects_variable_defined_only_on_skipped_path() {
    let mut program = branch_program();
    program.functions.get_mut("main").unwrap().body = vec![
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
        Instr::Let {
            var: "x".to_owned(),
            expr: JsonExpr::Literal {
                value: json!("defined only on else"),
            },
        },
        Instr::Return {
            value: JsonExpr::Var {
                name: "x".to_owned(),
            },
        },
    ];

    let error = validate_program(&program).unwrap_err();

    assert!(
        error
            .to_string()
            .contains("variable x is used before it is defined")
    );
}

#[test]
fn jump_target_rejects_variable_defined_only_on_skipped_path() {
    let mut program = branch_program();
    program.functions.get_mut("main").unwrap().body = vec![
        Instr::Jump { pc: 2 },
        Instr::Let {
            var: "x".to_owned(),
            expr: JsonExpr::Literal {
                value: json!("unreachable definition"),
            },
        },
        Instr::Return {
            value: JsonExpr::Var {
                name: "x".to_owned(),
            },
        },
    ];

    let error = validate_program(&program).unwrap_err();

    assert!(
        error
            .to_string()
            .contains("variable x is used before it is defined")
    );
}

#[test]
fn validate_patch_rejects_insert_at_existing_branch_target() {
    let base = branch_program();
    let patch = ProgramPatch {
        target_program_id: base.program_id.clone(),
        patch_id: "insert_at_branch_target".to_owned(),
        rationale: "inserting at an existing target would retarget the branch".to_owned(),
        operations: vec![PatchOp::InsertInstruction {
            function: "main".to_owned(),
            pc: 2,
            instr: Instr::Return {
                value: JsonExpr::Literal {
                    value: json!("shifted branch target"),
                },
            },
        }],
    };

    let err = validate_patch(&base, &patch).unwrap_err();

    assert!(
        err.to_string()
            .contains("insert pc 2 would shift existing branch else_pc target 2"),
        "unexpected error: {err}"
    );
}

#[test]
fn validate_patch_rejects_insert_at_existing_jump_target() {
    let mut functions = BTreeMap::new();
    functions.insert(
        "main".to_owned(),
        FunctionDef {
            params: Vec::new(),
            output_schema: json!({ "type": "string" }),
            body: vec![
                Instr::Jump { pc: 2 },
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
    let base = Program {
        program_id: "jump_fixture".to_owned(),
        version: "v0001".to_owned(),
        entry: "main".to_owned(),
        input_schema: json!({}),
        output_schema: json!({ "type": "string" }),
        functions,
        allowed_effects: Vec::new(),
    };
    let patch = ProgramPatch {
        target_program_id: base.program_id.clone(),
        patch_id: "insert_at_jump_target".to_owned(),
        rationale: "inserting at an existing target would retarget the jump".to_owned(),
        operations: vec![PatchOp::InsertInstruction {
            function: "main".to_owned(),
            pc: 2,
            instr: Instr::Return {
                value: JsonExpr::Literal {
                    value: json!("shifted jump target"),
                },
            },
        }],
    };

    let err = validate_patch(&base, &patch).unwrap_err();

    assert!(
        err.to_string()
            .contains("insert pc 2 would shift existing jump pc target 2"),
        "unexpected error: {err}"
    );
}

#[test]
fn validate_patch_allows_insert_after_existing_branch_targets() {
    let base = branch_program();
    let patch = ProgramPatch {
        target_program_id: base.program_id.clone(),
        patch_id: "insert_after_branch_targets".to_owned(),
        rationale: "appending after existing targets preserves branch meaning".to_owned(),
        operations: vec![PatchOp::InsertInstruction {
            function: "main".to_owned(),
            pc: 4,
            instr: Instr::Return {
                value: JsonExpr::Literal {
                    value: json!("tail"),
                },
            },
        }],
    };

    let patched = validate_patch(&base, &patch).unwrap();
    let body = &patched.functions["main"].body;

    assert_eq!(body[1], base.functions["main"].body[1]);
}

#[test]
fn validate_patch_rejects_later_insert_that_shifts_existing_branch_target() {
    let base = branch_program();
    let patch = ProgramPatch {
        target_program_id: base.program_id.clone(),
        patch_id: "later_insert_at_branch_target".to_owned(),
        rationale: "every insert in a multi-operation patch must preserve old targets".to_owned(),
        operations: vec![
            PatchOp::InsertInstruction {
                function: "main".to_owned(),
                pc: 4,
                instr: Instr::Return {
                    value: JsonExpr::Literal {
                        value: json!("tail"),
                    },
                },
            },
            PatchOp::InsertInstruction {
                function: "main".to_owned(),
                pc: 2,
                instr: Instr::Return {
                    value: JsonExpr::Literal {
                        value: json!("shifted branch target"),
                    },
                },
            },
        ],
    };

    let err = validate_patch(&base, &patch).unwrap_err();

    assert!(
        err.to_string()
            .contains("insert pc 2 would shift existing branch else_pc target 2"),
        "unexpected error: {err}"
    );
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
    let round1 = cargo_json([
        "run-stream",
        "--workflow",
        "notification_triage",
        "--events",
        "examples/notification_triage.round1.jsonl",
        "--state-dir",
        state.to_str().unwrap(),
    ]);
    let round1_run_id = round1["run_id"].as_str().unwrap().to_owned();
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
    let round2 = cargo_json([
        "run-stream",
        "--workflow",
        "notification_triage",
        "--events",
        "examples/notification_triage.round2.jsonl",
        "--state-dir",
        state.to_str().unwrap(),
    ]);
    let round2_run_id = round2["run_id"].as_str().unwrap().to_owned();
    let baseline = cargo_json([
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
    let baseline_run_id = baseline["run_id"].as_str().unwrap().to_owned();

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

    let comparison = cargo_json([
        "compare-runs",
        "--baseline-run",
        baseline_run_id.as_str(),
        "--before-run",
        round1_run_id.as_str(),
        "--after-run",
        round2_run_id.as_str(),
        "--state-dir",
        state.to_str().unwrap(),
    ]);
    for (name, value) in comparison.as_object().unwrap() {
        assert_eq!(
            value.as_bool(),
            Some(true),
            "compare-runs boolean {name} should be true"
        );
    }

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

#[tokio::test]
async fn optimizer_after_installed_patch_uses_only_latest_version_failures() {
    let state_path = temp_state_dir();
    let state = StateDir::new(state_path.clone());
    let workflow_id = "versioned_optimizer_profile";
    let program = branch_program();
    let programs = FileProgramRegistry::new(state.clone());
    programs
        .init_workflow(
            workflow_id,
            program.clone(),
            fixture_program_metadata(workflow_id, &program),
        )
        .unwrap();
    let installed = programs
        .install_version(
            workflow_id,
            program.clone(),
            ProgramMetadata {
                workflow_id: workflow_id.to_owned(),
                program_id: program.program_id.clone(),
                version: String::new(),
                created_at: String::new(),
                source: ProgramSource::PatchInstall,
                parent_version: Some("v0001".to_owned()),
                patch_id: Some("advance_to_v2".to_owned()),
                task_hash: "advance_to_v2".to_owned(),
            },
        )
        .unwrap();
    assert_eq!(installed, "v0002");

    let profiles = FileProfileStore::new(state.clone());
    profiles
        .update_from_trace(
            workflow_id,
            &[
                versioned_failure_capture_event(
                    "stale-v1-a",
                    "k-stale-v1-a",
                    &program.program_id,
                    "v0001",
                ),
                versioned_failure_capture_event(
                    "stale-v1-b",
                    "k-stale-v1-b",
                    &program.program_id,
                    "v0001",
                ),
                versioned_failure_capture_event(
                    "latest-v2",
                    "k-latest-v2",
                    &program.program_id,
                    "v0002",
                ),
            ],
        )
        .unwrap();

    let optimizer_requests = Arc::new(Mutex::new(Vec::<PatchRequest>::new()));
    let patches = FilePatchRegistry::new(state.clone());
    let traces = FileTraceStore::new(state.clone());
    let weak: Arc<dyn EffectHandler> = Arc::new(ScriptedHandler::empty());
    let strong: Arc<dyn EffectHandler> = Arc::new(RecordingOptimizerStrong {
        requests: Arc::clone(&optimizer_requests),
    });
    let error = optimize_from_profile(
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
    .unwrap_err();
    assert!(
        error.to_string().contains("strong optimizer aborted"),
        "unexpected optimizer error: {error}"
    );

    let requests = optimizer_requests.lock().unwrap();
    assert_eq!(requests.len(), 1);
    let request = &requests[0];
    assert_eq!(request.base_program.version, "v0002");
    assert_eq!(request.failure_fingerprint.program_version, "v0002");
    assert_eq!(request.compact_examples.len(), 1);
    assert_eq!(request.compact_examples[0].shape["count"], json!(1));

    let _ = fs::remove_dir_all(state_path);
}

#[tokio::test]
async fn run_stream_rejects_malformed_runtime_patch_id_without_aborting_later_events() {
    let state_path = temp_state_dir();
    let state = StateDir::new(state_path.clone());
    let workflow_id = "runtime_patch_id_regression";
    let program = runtime_patch_program();
    let programs = FileProgramRegistry::new(state.clone());
    programs
        .init_workflow(
            workflow_id,
            program.clone(),
            fixture_program_metadata(workflow_id, &program),
        )
        .unwrap();

    let weak: Arc<dyn EffectHandler> = Arc::new(ScriptedHandler::empty());
    let strong: Arc<dyn EffectHandler> = Arc::new(ScriptedHandler::new(vec![
        HandlerDecision::ReturnProgramPatch {
            patch: ProgramPatch {
                target_program_id: program.program_id.clone(),
                patch_id: "bad/id".to_owned(),
                operations: Vec::new(),
                rationale: "malformed patch id should be rejected".to_owned(),
            },
            rationale: "propose malformed runtime patch".to_owned(),
        },
        HandlerDecision::ReturnValue {
            value: Value::Null,
            confidence: 1.0,
            rationale: "second event continues".to_owned(),
        },
    ]));

    let summary = run_stream(
        state.clone(),
        workflow_id,
        InMemoryEventSource::new(vec![
            json!({ "event_id": "first" }),
            json!({ "event_id": "second" }),
        ]),
        weak,
        strong,
        false,
    )
    .await
    .unwrap();

    assert_eq!(summary.events_total, 2);
    assert_eq!(summary.events_succeeded, 1);
    assert_eq!(summary.events_failed, 1);

    let metrics = FileMetricsStore::new(state.clone())
        .read(workflow_id, &summary.run_id)
        .unwrap();
    assert_eq!(metrics.events_total, 2);
    assert_eq!(metrics.events_succeeded, 1);
    assert_eq!(metrics.events_failed, 1);
    assert_eq!(metrics.patches_proposed, 1);
    assert_eq!(metrics.patches_validated, 0);
    assert_eq!(metrics.patches_rejected, 1);

    let events = FileTraceStore::new(state.clone())
        .read_run(workflow_id, &summary.run_id)
        .unwrap();
    assert!(events.iter().any(|event| event.event == "patch_invalid"));
    assert!(
        events
            .iter()
            .any(|event| event.event == "stream_event" && event.event_id == "second")
    );
    assert!(
        FilePatchRegistry::new(state)
            .list_proposed(workflow_id)
            .unwrap()
            .is_empty()
    );

    let _ = fs::remove_dir_all(state_path);
}

struct ScriptedHandler {
    decisions: Mutex<VecDeque<HandlerDecision>>,
}

impl ScriptedHandler {
    fn new(decisions: Vec<HandlerDecision>) -> Self {
        Self {
            decisions: Mutex::new(decisions.into()),
        }
    }

    fn empty() -> Self {
        Self::new(Vec::new())
    }
}

#[async_trait]
impl EffectHandler for ScriptedHandler {
    async fn handle(&self, _request: HandlerRequest) -> Result<HandlerDecision> {
        self.decisions
            .lock()
            .unwrap()
            .pop_front()
            .ok_or_else(|| anyhow::anyhow!("unexpected handler call"))
    }
}

struct PatchReturningStrong {
    patch: ProgramPatch,
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

struct RecordingOptimizerStrong {
    requests: Arc<Mutex<Vec<PatchRequest>>>,
}

#[async_trait]
impl EffectHandler for RecordingOptimizerStrong {
    async fn handle(&self, request: HandlerRequest) -> Result<HandlerDecision> {
        match request.effect.model_task_name() {
            Some("optimize_program_patch") => {}
            _ => return Err(anyhow::anyhow!("unexpected optimizer handler request")),
        }
        self.requests
            .lock()
            .unwrap()
            .push(serde_json::from_value(request.input)?);
        Ok(HandlerDecision::Abort {
            reason: "request captured".to_owned(),
        })
    }
}

fn schema_valid_wrong_fast_path_patch() -> ProgramPatch {
    ProgramPatch {
        target_program_id: "notification_triage".to_owned(),
        patch_id: "bad_fast_path_output".to_owned(),
        rationale: "wrong schema-valid output must be rejected".to_owned(),
        operations: vec![
            PatchOp::InsertInstruction {
                function: "main".to_owned(),
                pc: 0,
                instr: Instr::Perform {
                    out: "fast".to_owned(),
                    effect: EffectCall::LocalTool {
                        tool_name: "fast_path_apply".to_owned(),
                        args_schema: json!({ "type": "object" }),
                    },
                    input: JsonExpr::Object {
                        fields: BTreeMap::from([
                            (
                                "event".to_owned(),
                                JsonExpr::Var {
                                    name: "event".to_owned(),
                                },
                            ),
                            (
                                "rules".to_owned(),
                                JsonExpr::Literal {
                                    value: json!([
                                        {
                                            "rule_id": "wrong_fixture_repeat_rule",
                                            "when": {
                                                "op": "field_equals",
                                                "path": ["_fixture_fast_path"],
                                                "value": true
                                            },
                                            "emit": {
                                                "kind": "literal",
                                                "value": {
                                                    "event_id": "sample-1",
                                                    "kind": "create_task",
                                                    "title": "Wrong but schema-valid action",
                                                    "datetime_hint": null
                                                }
                                            },
                                            "confidence": 1.0
                                        }
                                    ]),
                                },
                            ),
                        ]),
                    },
                    expected_schema: json!({ "type": "object" }),
                    acceptance: AcceptancePolicy {
                        min_confidence: Some(1.0),
                        require_schema_valid: true,
                        on_failure: FailureHandler::Abort {
                            reason: "fast path tool failed".to_owned(),
                        },
                    },
                },
            },
            PatchOp::InsertInstruction {
                function: "main".to_owned(),
                pc: 1,
                instr: Instr::Branch {
                    condition: GuardExpr::FieldIsTruthy {
                        var: "fast".to_owned(),
                        path: vec!["hit".to_owned()],
                    },
                    then_pc: 2,
                    else_pc: 4,
                },
            },
            PatchOp::InsertInstruction {
                function: "main".to_owned(),
                pc: 2,
                instr: Instr::Project {
                    out: "draft".to_owned(),
                    from: JsonExpr::Var {
                        name: "fast".to_owned(),
                    },
                    path: vec!["value".to_owned()],
                },
            },
            PatchOp::InsertInstruction {
                function: "main".to_owned(),
                pc: 3,
                instr: Instr::Return {
                    value: JsonExpr::Var {
                        name: "draft".to_owned(),
                    },
                },
            },
        ],
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

#[test]
fn frame_encoder_elides_stored_continuation_when_small_values_exceed_budget() {
    let state_path = temp_state_dir();
    let state = StateDir::new(state_path.clone());
    let workflow_id = "generic_workflow";
    let stack = (0..64)
        .map(|frame_index| {
            let mut env = Map::new();
            for value_index in 0..8 {
                env.insert(
                    format!("v{value_index}"),
                    json!(format!("small-{frame_index}-{value_index}")),
                );
            }
            RuntimeFrame {
                function: format!("function_{frame_index}"),
                pc: frame_index,
                env,
                return_to: None,
            }
        })
        .collect();
    let continuation = Continuation {
        continuation_id: "k-deep-small".to_owned(),
        boundary_id: "b1".to_owned(),
        program_id: "p1".to_owned(),
        stack,
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
            max_inline_value_bytes: 1024,
            max_model_visible_frame_bytes: 4096,
        },
    );

    let encoded = encoder.encode(&frame).unwrap();

    assert!(encoded.encoded_bytes <= 4096);
    assert_eq!(
        encoded.original_continuation_ref.as_deref(),
        Some("continuationstore://k-deep-small")
    );
    assert_eq!(encoded.model_visible_frame.continuation.stack.len(), 1);
    assert_eq!(
        encoded.model_visible_frame.continuation.stack[0].env["$continuation_ref"],
        json!("continuationstore://k-deep-small")
    );
    let stored = FileContinuationStore::new(state.clone(), workflow_id)
        .get("k-deep-small")
        .unwrap();
    assert_eq!(stored, continuation);

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

#[test]
fn validate_patch_rejects_unsafe_patch_id() {
    let base = branch_program();
    let patch = ProgramPatch {
        target_program_id: base.program_id.clone(),
        patch_id: "bad patch/id".to_owned(),
        operations: Vec::new(),
        rationale: "unsafe IDs must not become version or registry path components".to_owned(),
    };

    let err = validate_patch(&base, &patch).unwrap_err();

    assert!(
        err.to_string().contains("patch_id"),
        "unexpected error: {err}"
    );
    assert!(
        err.to_string().contains("unsafe filename characters"),
        "unexpected error: {err}"
    );
}

fn versioned_failure_capture_event(
    event_id: &str,
    continuation_id: &str,
    program_id: &str,
    program_version: &str,
) -> TraceEvent {
    TraceEvent {
        event: "capture_continuation".to_owned(),
        event_id: event_id.to_owned(),
        detail: json!({
            "continuation_id": continuation_id,
            "program_id": program_id,
            "program_version": program_version,
            "function": "main",
            "pc": 0,
            "failed_instruction_op": "perform",
            "failed_effect_kind": "model_task",
            "failed_task_name": "classify",
            "expected_schema": { "type": "object" },
            "observations": [{ "schema_valid": false, "value": { "status": "bad" } }]
        }),
    }
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

fn registry_test_patch(patch_id: &str) -> ProgramPatch {
    ProgramPatch {
        target_program_id: "notification_triage".to_owned(),
        patch_id: patch_id.to_owned(),
        operations: Vec::new(),
        rationale: "registry path safety test".to_owned(),
    }
}

fn runtime_patch_program() -> Program {
    let mut functions = BTreeMap::new();
    functions.insert(
        "main".to_owned(),
        FunctionDef {
            params: vec!["event".to_owned()],
            output_schema: json!({ "type": "null" }),
            body: vec![
                Instr::Perform {
                    out: "patch_ack".to_owned(),
                    effect: EffectCall::Think {
                        reason: "runtime may propose a future patch".to_owned(),
                    },
                    input: JsonExpr::Var {
                        name: "event".to_owned(),
                    },
                    expected_schema: json!({ "type": "null" }),
                    acceptance: AcceptancePolicy {
                        min_confidence: Some(1.0),
                        require_schema_valid: true,
                        on_failure: FailureHandler::Abort {
                            reason: "patch effect failed".to_owned(),
                        },
                    },
                },
                Instr::Return {
                    value: JsonExpr::Var {
                        name: "patch_ack".to_owned(),
                    },
                },
            ],
        },
    );

    Program {
        program_id: "runtime_patch_id_regression".to_owned(),
        version: "v0001".to_owned(),
        entry: "main".to_owned(),
        input_schema: json!({ "type": "object" }),
        output_schema: json!({ "type": "null" }),
        functions,
        allowed_effects: vec![EffectPermission::Think],
    }
}

fn test_run_metrics() -> RunMetrics {
    RunMetrics::new(
        "test-run".to_owned(),
        "generic_workflow".to_owned(),
        "stream".to_owned(),
        "p1".to_owned(),
        "v0001".to_owned(),
        "started".to_owned(),
    )
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
