use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use cps_llm_demo::config::DEFAULT_BASE_URL;
use cps_llm_demo::engine::event_source::InMemoryEventSource;
use cps_llm_demo::engine::run_coordinator::{
    PredictionWriteOptions, RunStreamOptions, run_stream_with_options,
};
use cps_llm_demo::experiment::config::{
    ExperimentBudget, ExperimentCache, ExperimentConfig, ExperimentData, ExperimentModels,
    ExperimentSchemas, ExperimentSplitCounts, ExperimentWorkflow, load_experiment_config,
    write_locked_config,
};
use cps_llm_demo::experiment::exact_memo::ExactMemoTable;
use cps_llm_demo::experiment::patch_gate::{PatchGateConfig, PatchGateInput, evaluate_patch_gate};
use cps_llm_demo::experiment::prediction::{
    FastPathPredictionMetadata, ModelCallCounts, PredictionRecord, PredictionStore,
};
use cps_llm_demo::experiment::quality::{GoldLabel, evaluate_quality};
use cps_llm_demo::experiment::report::{VariantReportRow, build_pass_fail};
use cps_llm_demo::experiment::runner::run_experiment;
use cps_llm_demo::experiment::semantic_fast_path::{
    SEMANTIC_FAST_PATH_TASK, generalized_semantic_metadata,
    semantic_fast_path_patch_uses_weak_matcher_not_exact_text as patch_uses_weak_matcher,
};
use cps_llm_demo::experiment::shadow::{
    shadow_execution_does_not_change_output as run_shadow_execution, summarize_shadow_audit,
};
use cps_llm_demo::experiment::split::{
    EventSplits, SplitCounts, SplitStrategy, char_ngram_jaccard, leakage_report, split_events_files,
};
use cps_llm_demo::experiment::variants::{
    ExperimentVariant, variant_runner_runs_all_required_variants as variants_include_all,
};
use cps_llm_demo::local_tools::apply_template_emit;
use cps_llm_demo::model_cache::{
    ModelCache, ModelCacheEntryParts, ModelCacheMode, cache_entry_from_response,
    structured_cache_key,
};
use cps_llm_demo::models::FixtureModelHandler;
use cps_llm_demo::pricing::price_catalog::{ModelPrice, PriceCatalog};
use cps_llm_demo::program::{
    AcceptancePolicy, EffectCall, EffectPermission, FailureHandler, FunctionDef,
    GeneralizationScope, GuardExpr, Instr, JsonExpr, ModelStrength, ModelTaskSpec,
    PatchGeneralizationMetadata, PatchKind, PatchOp, Program, ProgramPatch,
};
use cps_llm_demo::responses_client::{
    ModelCallContext, ModelCallRuntime, ResponsesClient, ResponsesClientConfig, extract_usage,
};
use cps_llm_demo::store::budget_store::{BudgetConfig, BudgetSpendRecord, FileBudgetStore};
use cps_llm_demo::store::metrics_store::{FileMetricsStore, RunMetrics};
use cps_llm_demo::store::model_call_store::{
    CacheStatus, CostBreakdown, FileModelCallStore, ModelCallRecord, ModelUsage,
};
use cps_llm_demo::store::program_registry::{FileProgramRegistry, fixture_program_metadata};
use cps_llm_demo::store::state_dir::{StateDir, now_string};
use httpmock::Method::POST;
use httpmock::MockServer;
use secrecy::SecretString;
use serde_json::{Value, json};
use url::Url;

#[tokio::test]
async fn model_call_log_records_usage_and_cost() {
    let state = temp_state();
    let server = mock_structured_response(json!({"ok": true}), Some(usage_json(1000, 100, 50)));
    let client = logged_client(&server, &state, ModelCacheMode::Disabled, default_catalog());

    let response: serde_json::Value = client
        .create_structured_with_context(
            "gpt-5.4-mini",
            "Return JSON.",
            &json!({"event_id": "e1"}),
            "object",
            json!({"type": "object"}),
            call_context("weak_model", "model_task"),
        )
        .await
        .unwrap()
        .parsed;

    assert_eq!(response, json!({"ok": true}));
    let records = FileModelCallStore::new(state).list_all().unwrap();
    assert_eq!(records.len(), 1);
    assert_eq!(records[0].usage.as_ref().unwrap().input_tokens, 1000);
    assert!(records[0].cost.total_usd > 0.0);
    assert!(records[0].success);
}

#[tokio::test]
async fn failed_model_call_is_logged() {
    let state = temp_state();
    let server = MockServer::start();
    server.mock(|when, then| {
        when.method(POST).path("/v1/responses");
        then.status(500)
            .json_body(json!({"error": {"message": "boom"}}));
    });
    let client = logged_client(&server, &state, ModelCacheMode::Disabled, default_catalog());

    let error = client
        .create_structured_with_context::<Value>(
            "gpt-5.4-mini",
            "Return JSON.",
            &json!({}),
            "object",
            json!({"type": "object"}),
            call_context("weak_model", "model_task"),
        )
        .await
        .unwrap_err();

    assert!(error.to_string().contains("Responses API error"));
    let records = FileModelCallStore::new(state).list_all().unwrap();
    assert_eq!(records.len(), 1);
    assert!(!records[0].success);
    assert!(
        records[0]
            .error
            .as_deref()
            .unwrap()
            .contains("Responses API error")
    );
}

#[tokio::test]
async fn cache_hit_logs_zero_marginal_cost() {
    let state = temp_state();
    let server = mock_structured_response(json!({"ok": true}), Some(usage_json(1000, 100, 0)));
    let client = logged_client(
        &server,
        &state,
        ModelCacheMode::ReadWrite,
        default_catalog(),
    );
    for _ in 0..2 {
        let _: Value = client
            .create_structured_with_context(
                "gpt-5.4-mini",
                "Return JSON.",
                &json!({"same": true}),
                "object",
                json!({"type": "object"}),
                call_context("weak_model", "model_task"),
            )
            .await
            .unwrap()
            .parsed;
    }

    let records = FileModelCallStore::new(state).list_all().unwrap();
    assert_eq!(records.len(), 2);
    assert_eq!(records[1].cache_status, CacheStatus::Hit);
    assert_eq!(records[1].cost.total_usd, 0.0);
}

#[test]
fn identical_model_request_hits_cache() {
    let state = temp_state();
    let cache = ModelCache::new(state.root().join("cache"), ModelCacheMode::ReadWrite);
    let key = cache_key_for_schema(json!({"type": "object"}));
    cache
        .put(&cache_entry_from_response(ModelCacheEntryParts {
            key: key.clone(),
            model: "gpt-5.4-mini".to_owned(),
            prompt_hash: "p".to_owned(),
            schema_hash: "s".to_owned(),
            model_config_hash: "m".to_owned(),
            parsed_output: json!({"ok": true}),
            raw_response: json!({"output_text": "{}"}),
            usage: None,
        }))
        .unwrap();
    assert!(cache.get(&key).unwrap().is_some());
}

#[test]
fn schema_change_misses_cache() {
    let object_key = cache_key_for_schema(json!({"type": "object"}));
    let array_key = cache_key_for_schema(json!({"type": "array"}));
    assert_ne!(object_key, array_key);
}

#[test]
fn cache_readonly_errors_on_missing_key() {
    let state = temp_state();
    let cache = ModelCache::new(state.root().join("cache"), ModelCacheMode::ReadOnly);
    let error = cache.get("missing").unwrap_err();
    assert!(error.to_string().contains("read-only mode"));
}

#[test]
fn budget_guard_aborts_projected_overspend() {
    let state = temp_state();
    let store = FileBudgetStore::new(state);
    let guard = cps_llm_demo::store::budget_store::BudgetGuard::new(
        BudgetConfig {
            hard_cap_usd: 0.000001,
            ..BudgetConfig::default()
        },
        default_catalog(),
        store,
    );
    let error = guard
        .ensure_call_allowed("gpt-5.5", 10_000_000, Some(10_000))
        .unwrap_err();
    assert!(error.to_string().contains("budget hard cap"));
}

#[test]
fn budget_ledger_uses_reported_usage_when_available() {
    let catalog = default_catalog();
    let usage = ModelUsage {
        input_tokens: 1_000,
        cached_input_tokens: 100,
        output_tokens: 100,
        reasoning_tokens: 0,
        total_tokens: 1_100,
    };
    let cost = catalog.estimate_cost("gpt-5.5", &usage, false).unwrap();
    assert!(!cost.estimated);
    assert!(cost.cached_input_usd > 0.0);
    assert!(cost.total_usd > cost.cached_input_usd);
}

#[test]
fn budget_report_groups_by_model_and_phase() {
    let state = temp_state();
    let store = FileBudgetStore::new(state);
    store
        .append_spend(&spend("c1", "gpt-5.5", "baseline", 1.0, CacheStatus::Miss))
        .unwrap();
    store
        .append_spend(&spend(
            "c2",
            "gpt-5.4-mini",
            "heldout",
            2.0,
            CacheStatus::Hit,
        ))
        .unwrap();
    let report = store.report(&BudgetConfig::default()).unwrap();
    assert_eq!(report.by_model["gpt-5.5"], 1.0);
    assert_eq!(report.by_phase["heldout"], 2.0);
    assert_eq!(report.cache_hit_rate, 0.5);
}

#[test]
fn experiment_config_locks_model_and_price_config() {
    let dir = temp_dir();
    let price = dir.join("prices.yaml");
    default_catalog().write_yaml(&price).unwrap();
    let config = experiment_config(&dir, &price, 100.0);
    let path = dir.join("config.yaml");
    fs::write(&path, serde_yaml::to_string(&config).unwrap()).unwrap();
    let loaded = load_experiment_config(&path).unwrap();
    let lock = dir.join("config.lock.yaml");
    write_locked_config(&loaded, &lock).unwrap();
    assert_eq!(loaded.models.weak_model, "gpt-5.4-mini");
    assert!(lock.exists());
}

#[tokio::test]
async fn run_experiment_resumes_from_completed_phase() {
    let dir = temp_dir();
    let price = dir.join("prices.yaml");
    default_catalog().write_yaml(&price).unwrap();
    write_events_and_gold(&dir, 4, 1);
    let config = experiment_config(&dir, &price, 100.0);
    let state = StateDir::new(dir.join("state"));
    run_experiment(config.clone(), state.clone(), false)
        .await
        .unwrap();
    run_experiment(config, state.clone(), false).await.unwrap();
    let manifest: serde_json::Value = serde_json::from_str(
        &fs::read_to_string(
            state
                .root()
                .join("experiments")
                .join("notification_triage_real_v1")
                .join("run_manifest.json"),
        )
        .unwrap(),
    )
    .unwrap();
    assert_eq!(manifest["completed_phases"].as_array().unwrap().len(), 3);
}

#[tokio::test]
async fn run_experiment_respects_budget_hard_cap() {
    let dir = temp_dir();
    let price = dir.join("prices.yaml");
    default_catalog().write_yaml(&price).unwrap();
    write_events_and_gold(&dir, 10, 1);
    let config = experiment_config(&dir, &price, 0.000001);
    let error = run_experiment(config, StateDir::new(dir.join("state")), false)
        .await
        .unwrap_err();
    assert!(error.to_string().contains("refused to start"));
}

#[tokio::test]
async fn run_experiment_executes_variants_and_writes_predictions_and_quality() {
    let dir = temp_dir();
    let price = dir.join("prices.yaml");
    default_catalog().write_yaml(&price).unwrap();
    write_fixture_program_and_task(&dir);
    write_experiment_schemas(&dir);
    write_events_and_gold(&dir, 5, 1);
    let mut config = experiment_config(&dir, &price, 100.0);
    config.workflow = Some(ExperimentWorkflow {
        program: dir.join("program.json"),
        task: dir.join("task.md"),
    });
    config.split_counts = Some(ExperimentSplitCounts {
        profile_train: 2,
        patch_validation: 1,
        heldout_test: 2,
        adversarial_test: 1,
    });
    config.phases = vec![
        "split_events".to_owned(),
        "weak_only".to_owned(),
        "strong_direct".to_owned(),
        "cps_unoptimized_profile".to_owned(),
        "cps_unoptimized_heldout".to_owned(),
        "optimize".to_owned(),
        "patch_validation".to_owned(),
        "cps_exact_memo_heldout".to_owned(),
        "cps_generalized_patch_heldout".to_owned(),
        "cps_generalized_patch_no_semantic_weak".to_owned(),
        "adversarial_audit".to_owned(),
        "shadow_audit".to_owned(),
        "quality_eval".to_owned(),
        "report".to_owned(),
    ];
    let state = StateDir::new(dir.join("state"));

    let output = run_experiment(config, state.clone(), false).await.unwrap();

    assert_eq!(output["skipped_phases"].as_array().unwrap().len(), 0);
    let experiment_dir = state
        .root()
        .join("experiments")
        .join("notification_triage_real_v1");
    for name in [
        "weak_only.heldout.jsonl",
        "strong_direct.heldout.jsonl",
        "cps_unoptimized.heldout.jsonl",
        "cps_exact_memo.heldout.jsonl",
        "cps_generalized_patch.heldout.jsonl",
        "cps_generalized_patch_no_semantic_weak.heldout.jsonl",
    ] {
        let predictions =
            PredictionStore::read(&experiment_dir.join("predictions").join(name)).unwrap();
        assert_eq!(predictions.len(), 2, "{name}");
        assert!(
            predictions.iter().all(|prediction| prediction.schema_valid),
            "{name}"
        );
    }
    let generalized = PredictionStore::read(
        &experiment_dir
            .join("predictions")
            .join("cps_generalized_patch.heldout.jsonl"),
    )
    .unwrap();
    assert!(
        generalized
            .iter()
            .all(|prediction| prediction.program_version == "v0002"),
        "generalized heldout predictions should be produced by the installed program"
    );
    assert!(
        generalized
            .iter()
            .all(|prediction| prediction.fast_path.hit),
        "semantic patch should fast-path seen semantic clusters"
    );
    assert!(
        generalized
            .iter()
            .all(|prediction| prediction.model_calls.weak == 1
                && prediction.model_calls.strong_task == 0),
        "semantic fast path should avoid strong task calls"
    );
    let no_semantic = PredictionStore::read(
        &experiment_dir
            .join("predictions")
            .join("cps_generalized_patch_no_semantic_weak.heldout.jsonl"),
    )
    .unwrap();
    assert!(
        no_semantic
            .iter()
            .all(|prediction| !prediction.fast_path.hit
                && prediction.model_calls.weak == 0
                && prediction.model_calls.strong_task == 1),
        "no-semantic control should bypass weak matcher and fall back to strong"
    );
    let adversarial = PredictionStore::read(
        &experiment_dir
            .join("predictions")
            .join("cps_generalized_patch.adversarial.jsonl"),
    )
    .unwrap();
    assert_eq!(adversarial.len(), 1);
    assert!(
        adversarial
            .iter()
            .all(|prediction| !prediction.fast_path.hit && prediction.schema_valid),
        "hard negative should not fast-path when its cluster was not optimized"
    );
    assert!(
        experiment_dir
            .join("artifacts")
            .join("semantic_patch_plan.json")
            .exists()
    );
    assert!(
        experiment_dir
            .join("artifacts")
            .join("optimizer_semantic_patch_request.json")
            .exists()
    );
    assert!(
        experiment_dir
            .join("artifacts")
            .join("optimizer_semantic_patch_response.json")
            .exists()
    );
    assert!(
        experiment_dir
            .join("artifacts")
            .join("optimizer")
            .join("optimizer_request.json")
            .exists()
    );
    assert!(
        experiment_dir
            .join("artifacts")
            .join("optimizer")
            .join("optimizer_response.json")
            .exists()
    );
    assert!(
        experiment_dir
            .join("artifacts")
            .join("semantic_patch_generation.json")
            .exists()
    );
    let optimizer_request: serde_json::Value = serde_json::from_slice(
        &fs::read(
            experiment_dir
                .join("artifacts")
                .join("optimizer")
                .join("optimizer_request.json"),
        )
        .unwrap(),
    )
    .unwrap();
    assert!(
        optimizer_request.get("candidate_plan").is_none(),
        "strong optimizer must generate the plan from evidence, not echo a candidate plan"
    );
    assert!(
        optimizer_request["profile_evidence"]["clusters"]
            .as_array()
            .unwrap()
            .iter()
            .any(|cluster| cluster["fast_path_eligible"] == json!(true)),
        "optimizer request should include profile evidence for rule generation"
    );
    let optimizer_response: serde_json::Value = serde_json::from_slice(
        &fs::read(
            experiment_dir
                .join("artifacts")
                .join("optimizer")
                .join("optimizer_response.json"),
        )
        .unwrap(),
    )
    .unwrap();
    let semantic_patch_plan: serde_json::Value = serde_json::from_slice(
        &fs::read(
            experiment_dir
                .join("artifacts")
                .join("semantic_patch_plan.json"),
        )
        .unwrap(),
    )
    .unwrap();
    assert_eq!(
        semantic_patch_plan["optimizer_source"],
        json!("strong_model_generated_semantic_patch_plan")
    );
    assert_eq!(
        optimizer_response["value"]["optimizer_source"], semantic_patch_plan["optimizer_source"],
        "raw optimizer response should be the final semantic patch plan source"
    );
    let semantic_patch_generation: serde_json::Value = serde_json::from_slice(
        &fs::read(
            experiment_dir
                .join("artifacts")
                .join("semantic_patch_generation.json"),
        )
        .unwrap(),
    )
    .unwrap();
    assert_eq!(
        semantic_patch_generation["raw_optimizer_response_is_final_plan"],
        json!(true)
    );
    assert!(
        experiment_dir
            .join("artifacts")
            .join("semantic_rules.json")
            .exists()
    );
    assert!(
        experiment_dir
            .join("artifacts")
            .join("exact_memo_table.json")
            .exists()
    );
    let shadow_metrics: serde_json::Value = serde_json::from_slice(
        &fs::read(experiment_dir.join("shadow").join("shadow_metrics.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(shadow_metrics["fast_path_hits"], json!(2));
    assert_eq!(shadow_metrics["shadow_checked"], json!(2));
    for name in [
        "weak_only.json",
        "strong_direct.json",
        "cps_unoptimized.json",
        "cps_exact_memo.json",
        "cps_generalized_patch.json",
        "cps_generalized_patch_no_semantic_weak.json",
        "cps_generalized_patch.adversarial.json",
    ] {
        assert!(experiment_dir.join("quality").join(name).exists(), "{name}");
    }
    let report: serde_json::Value =
        serde_json::from_slice(&fs::read(experiment_dir.join("report.json")).unwrap()).unwrap();
    assert_eq!(
        report["experiment_id"],
        json!("notification_triage_real_v1")
    );
    assert!(
        report["variants"]
            .as_array()
            .unwrap()
            .iter()
            .any(|row| row["variant"] == json!("cps_generalized_patch"))
    );
    let generalized_row = report["variants"]
        .as_array()
        .unwrap()
        .iter()
        .find(|row| row["variant"] == json!("cps_generalized_patch"))
        .unwrap();
    assert_eq!(generalized_row["false_fast_path_rate"], json!(0.0));
    assert_eq!(report["patch"]["installed_program_version"], json!("v0002"));
    assert_eq!(report["patch"]["source"], json!("optimizer_strong_model"));
    assert_eq!(
        report["patch"]["optimizer_source"],
        json!("strong_model_generated_semantic_patch_plan")
    );
    assert_eq!(
        report["patch"]["raw_optimizer_response_is_final_plan"],
        json!(true)
    );
    assert_eq!(report["patch_gate"]["accepted"], json!(true));
    let latest = fs::read_to_string(
        state
            .root()
            .join("workflows")
            .join("notification_triage")
            .join("programs")
            .join("latest"),
    )
    .unwrap();
    assert_eq!(latest.trim(), "v0002");
    let installed_program: Program = serde_json::from_slice(
        &fs::read(
            state
                .root()
                .join("workflows")
                .join("notification_triage")
                .join("programs")
                .join("v0002")
                .join("program.json"),
        )
        .unwrap(),
    )
    .unwrap();
    let body = &installed_program.functions["main"].body;
    assert!(
        body.iter()
            .any(|instr| matches!(instr, Instr::Branch { .. })),
        "installed program should branch on semantic fast-path match"
    );
    assert!(
        body.iter().any(|instr| matches!(
            instr,
            Instr::Perform {
                effect: EffectCall::LocalTool { tool_name, .. },
                ..
            } if tool_name == "validator_apply"
        )),
        "installed program should validate semantic fast-path slots"
    );
    assert!(
        body.iter().any(|instr| matches!(
            instr,
            Instr::Perform {
                effect: EffectCall::LocalTool { tool_name, .. },
                ..
            } if tool_name == "template_emit"
        )),
        "installed program should emit fast-path output through template_emit"
    );
}

#[tokio::test]
async fn experiment_report_scopes_spend_and_frame_metrics_to_manifest_runs() {
    let dir = temp_dir();
    let price = dir.join("prices.yaml");
    default_catalog().write_yaml(&price).unwrap();
    write_events_and_gold(&dir, 1, 0);
    let splits_dir = dir.join("splits");
    fs::create_dir_all(&splits_dir).unwrap();
    let gold = vec![gold("e0", "create_task", true, false)];
    write_jsonl(&splits_dir.join("heldout_test.gold.jsonl"), &gold);

    let mut config = experiment_config(&dir, &price, 100.0);
    config.phases = vec!["report".to_owned()];
    let state = StateDir::new(dir.join("state"));
    let experiment_dir = state
        .root()
        .join("experiments")
        .join("notification_triage_real_v1");
    fs::create_dir_all(experiment_dir.join("quality")).unwrap();

    let prediction = prediction("e0", "create_task", true);
    let predictions = vec![prediction.clone()];
    let quality = evaluate_quality(&predictions, &gold).unwrap();
    let prediction_store = PredictionStore::new(experiment_dir.join("predictions"));
    prediction_store
        .append("cps_generalized_patch.heldout.jsonl", &prediction)
        .unwrap();
    prediction_store
        .append("cps_exact_memo.heldout.jsonl", &prediction)
        .unwrap();
    fs::write(
        experiment_dir
            .join("quality")
            .join("cps_generalized_patch.json"),
        serde_json::to_vec_pretty(&quality).unwrap(),
    )
    .unwrap();
    fs::write(
        experiment_dir.join("run_manifest.json"),
        serde_json::to_vec_pretty(&json!({
            "experiment_id": "notification_triage_real_v1",
            "workflow_id": "notification_triage",
            "started_at": now_string(),
            "runs": [
                {
                    "phase": "cps_generalized_patch_heldout",
                    "run_id": "current-run"
                }
            ],
            "completed_phases": [],
            "skipped_phases": [],
            "dry_run_cost": false
        }))
        .unwrap(),
    )
    .unwrap();

    let call_store = FileModelCallStore::new(state.clone());
    let budget_store = FileBudgetStore::new(state.clone());
    for record in [
        model_call_record(
            "current-call",
            "current-run",
            "notification_triage",
            "cps_generalized_patch",
            1.25,
        ),
        model_call_record(
            "stale-call",
            "stale-run",
            "notification_triage",
            "cps_generalized_patch",
            9.75,
        ),
    ] {
        call_store.append(&record).unwrap();
        budget_store.record_model_call(&record).unwrap();
    }

    let metrics_store = FileMetricsStore::new(state.clone());
    metrics_store
        .write(&run_metrics("current-run", "notification_triage", 2_222))
        .unwrap();
    metrics_store
        .write(&run_metrics("stale-run", "notification_triage", 99_999))
        .unwrap();

    run_experiment(config, state.clone(), false).await.unwrap();

    let report: serde_json::Value =
        serde_json::from_slice(&fs::read(experiment_dir.join("report.json")).unwrap()).unwrap();
    let generalized_row = report["variants"]
        .as_array()
        .unwrap()
        .iter()
        .find(|row| row["variant"] == json!("cps_generalized_patch"))
        .unwrap();
    assert_eq!(generalized_row["api_spend_usd"], json!(1.25));
    assert_eq!(generalized_row["p95_frame_bytes"], json!(2_222));
    assert_eq!(report["continuation_frames"]["p95_bytes"], json!(2_222));
    assert_eq!(report["budget"]["spent_usd"], json!(1.25));
    assert_eq!(report["budget"]["calls_total"], json!(1));
}

#[test]
fn exact_duplicates_are_not_cross_split() {
    let dir = temp_dir();
    let events = vec![
        event("e1", "same text", "2026-06-05T09:00:00+08:00"),
        event("e2", "same text", "2026-06-05T09:01:00+08:00"),
        event("e3", "different", "2026-06-05T09:02:00+08:00"),
    ];
    let gold = labels_for(&events, false);
    write_jsonl(&dir.join("events.jsonl"), &events);
    write_jsonl(&dir.join("gold.jsonl"), &gold);
    let result = split_events_files(
        &dir.join("events.jsonl"),
        &dir.join("gold.jsonl"),
        &dir.join("splits"),
        SplitStrategy::TimeCluster,
        SplitCounts {
            profile_train: 2,
            patch_validation: 1,
            heldout_test: 0,
            adversarial_test: 0,
        },
    )
    .unwrap();
    assert_eq!(result.leakage_report.exact_duplicate_cross_split, 0);
}

#[test]
fn leakage_report_detects_near_duplicates() {
    let split = EventSplits {
        profile_train: vec![event("e1", "send the revised proposal today", "t1")],
        patch_validation: vec![],
        heldout_test: vec![event("e2", "send revised proposal today", "t2")],
        adversarial_test: vec![],
    };
    let labels = labels_for(
        &[
            split.profile_train[0].clone(),
            split.heldout_test[0].clone(),
        ],
        false,
    );
    let report = leakage_report(&split, &labels);
    assert!(report.near_duplicate_cross_split_rate > 0.0);
    assert!(char_ngram_jaccard("abcdef", "abcxef", 3) > 0.0);
}

#[test]
fn split_preserves_adversarial_set() {
    let dir = temp_dir();
    let events = vec![
        event("e1", "ordinary", "t1"),
        event("e2", "hard negative", "t2"),
    ];
    let mut gold = labels_for(&events, false);
    gold[1].hard_negative = true;
    write_jsonl(&dir.join("events.jsonl"), &events);
    write_jsonl(&dir.join("gold.jsonl"), &gold);
    let result = split_events_files(
        &dir.join("events.jsonl"),
        &dir.join("gold.jsonl"),
        &dir.join("splits"),
        SplitStrategy::TimeCluster,
        SplitCounts {
            profile_train: 1,
            patch_validation: 0,
            heldout_test: 0,
            adversarial_test: 1,
        },
    )
    .unwrap();
    assert_eq!(result.leakage_report.adversarial_cases_total, 1);
}

#[test]
fn quality_evaluator_scores_intent_accuracy() {
    let predictions = vec![prediction("e1", "create_task", true)];
    let gold = vec![gold("e1", "create_task", true, false)];
    let metrics = evaluate_quality(&predictions, &gold).unwrap();
    assert_eq!(metrics.intent_accuracy, 1.0);
}

#[test]
fn quality_evaluator_treats_ignore_as_no_action() {
    let predictions = vec![prediction("e1", "ignore", true)];
    let gold = vec![gold("e1", "no_action", false, true)];
    let metrics = evaluate_quality(&predictions, &gold).unwrap();
    assert_eq!(metrics.intent_accuracy, 1.0);
    assert_eq!(metrics.hard_negative_false_action_rate, 0.0);
}

#[test]
fn quality_evaluator_counts_critical_misses() {
    let predictions = vec![prediction("e1", "no_action", true)];
    let mut gold = gold("e1", "create_task", true, false);
    gold.criticality = "critical".to_owned();
    let metrics = evaluate_quality(&predictions, &[gold]).unwrap();
    assert_eq!(metrics.critical_miss_rate, 1.0);
}

#[test]
fn gold_labels_must_not_be_generated_from_predictions() {
    let predictions = vec![prediction("e1", "create_task", true)];
    let mut gold = gold("e1", "create_task", true, false);
    gold.generated_from_predictions = true;
    let error = evaluate_quality(&predictions, &[gold]).unwrap_err();
    assert!(error.to_string().contains("generated_from_predictions"));
}

#[tokio::test]
async fn run_stream_writes_predictions() {
    let records = run_prediction_stream_and_read().await;
    assert_eq!(records.len(), 1);
    assert_eq!(records[0].output["kind"], json!("create_task"));
}

#[tokio::test]
async fn run_stream_writes_invalid_prediction_for_failed_event() {
    let state = temp_state();
    let workflow = "prediction_stream_failure";
    let program = weak_program();
    FileProgramRegistry::new(state.clone())
        .init_workflow(
            workflow,
            program.clone(),
            fixture_program_metadata(workflow, &program),
        )
        .unwrap();
    let store = PredictionStore::new(state.root().join("predictions"));

    let summary = run_stream_with_options(
        state.clone(),
        workflow,
        InMemoryEventSource::new(vec![
            json!({
                "event_id": "ok",
                "_fixture_model": {
                    "weak": {
                        "value": {
                            "event_id": "ok",
                            "kind": "create_task",
                            "title": "title",
                            "datetime_hint": "tomorrow"
                        }
                    }
                }
            }),
            json!({
                "event_id": "abort",
                "_fixture_model": {
                    "weak": {
                        "abort": "model refused"
                    }
                }
            }),
        ]),
        Arc::new(FixtureModelHandler::weak()),
        Arc::new(FixtureModelHandler::strong()),
        false,
        RunStreamOptions {
            run_id: None,
            predictions: Some(PredictionWriteOptions {
                store,
                file_name: "cps_generalized_patch.heldout.jsonl".to_owned(),
                variant: "cps_generalized_patch".to_owned(),
            }),
        },
    )
    .await
    .unwrap();

    assert_eq!(summary.events_total, 2);
    assert_eq!(summary.events_succeeded, 1);
    assert_eq!(summary.events_failed, 1);

    let records = PredictionStore::read(
        &state
            .root()
            .join("predictions")
            .join("cps_generalized_patch.heldout.jsonl"),
    )
    .unwrap();
    assert_eq!(records.len(), 2);

    let failed = records
        .iter()
        .find(|record| record.event_id == "abort")
        .expect("failed event should have a prediction row");
    assert!(!failed.schema_valid);
    assert!(
        failed.output["error"]
            .as_str()
            .unwrap()
            .contains("weak_model aborted: model refused")
    );

    let metrics = evaluate_quality(
        &records,
        &[
            gold("ok", "create_task", true, false),
            gold("abort", "create_task", true, false),
        ],
    )
    .unwrap();
    assert_eq!(metrics.events_total, 2);
    assert_eq!(metrics.schema_validity, 0.5);
    assert_eq!(metrics.actionable_recall, 0.5);
}

#[tokio::test]
async fn predictions_include_fast_path_metadata() {
    let records = run_prediction_stream_and_read().await;
    assert!(records[0].fast_path.hit);
    assert_eq!(records[0].fast_path.rule_id.as_deref(), Some("r1"));
}

#[test]
fn semantic_fast_path_patch_uses_weak_matcher_not_exact_text() {
    let patch = semantic_patch();
    assert!(patch_uses_weak_matcher(&patch));
}

#[test]
fn template_emit_supports_nested_paths() {
    let output = apply_template_emit(json!({
        "input": {
            "event": { "event_id": "e1", "sender_hint": "coworker" },
            "slots": { "document": "proposal", "deadline": "tomorrow" }
        },
        "template": {
            "event_id": { "path": ["event", "event_id"] },
            "kind": { "literal": "create_task" },
            "title": { "format": "send {slots.document} to {event.sender_hint}" },
            "datetime_hint": { "path": ["slots", "deadline"] }
        }
    }))
    .unwrap();
    assert_eq!(output["title"], json!("send proposal to coworker"));
}

#[test]
fn patch_metadata_records_generalization_scope() {
    let metadata = generalized_semantic_metadata(vec!["cluster_a".to_owned()], vec![]);
    let raw = serde_json::to_value(&metadata).unwrap();
    assert_eq!(raw["generalization_scope"], json!("semantic_class"));
}

#[test]
fn exact_memo_patch_does_not_match_paraphrase() {
    let events = vec![event("e1", "send proposal today", "t1")];
    let predictions = vec![prediction("e1", "create_task", true)];
    let table = ExactMemoTable::from_predictions(&events, &predictions);
    assert!(
        table
            .lookup(&event("e2", "please send the proposal today", "t2"))
            .is_none()
    );
}

#[test]
fn report_compares_exact_memo_vs_generalized_patch() {
    let budget = cps_llm_demo::store::budget_store::BudgetReport {
        hard_cap_usd: 100.0,
        soft_cap_usd: 60.0,
        spent_usd: 10.0,
        remaining_usd: 90.0,
        calls_total: 1,
        cache_hit_rate: 0.0,
        by_model: BTreeMap::new(),
        by_phase: BTreeMap::new(),
    };
    let strong = row("strong_direct", 0.95, 1.0, 0.0, 0.0);
    let exact = row("cps_exact_memo", 0.94, 0.4, 0.1, 0.03);
    let generalized = row("cps_generalized_patch", 0.94, 0.3, 0.5, 0.46);
    let pass = build_pass_fail(&strong, &exact, &generalized, &budget);
    assert!(pass.generalized_patch_beats_exact_memo);
}

#[test]
fn shadow_execution_does_not_change_output() {
    let prediction = prediction("e1", "create_task", true);
    let (unchanged, audit) = run_shadow_execution(&prediction, json!({"kind": "no_action"}), false);
    assert_eq!(unchanged.output, prediction.output);
    assert!(audit.disagreed);
}

#[test]
fn shadow_audit_records_fast_path_disagreement() {
    let mut prediction = prediction("e1", "create_task", true);
    prediction.fast_path.hit = true;
    let (_, audit) = run_shadow_execution(&prediction, json!({"kind": "no_action"}), true);
    let metrics = summarize_shadow_audit(&[prediction], &[audit]);
    assert_eq!(metrics.shadow_disagreements, 1);
    assert_eq!(metrics.critical_shadow_disagreements, 1);
}

#[test]
fn shadow_audit_ignores_title_only_differences() {
    let mut prediction = prediction("e1", "create_task", true);
    prediction.output["title"] = json!("fast title");
    let (_, audit) = run_shadow_execution(
        &prediction,
        json!({
            "event_id": "e1",
            "kind": "create_task",
            "title": "shadow title",
            "datetime_hint": "tomorrow"
        }),
        false,
    );
    assert!(!audit.disagreed);
}

#[test]
fn shadow_audit_treats_ignore_as_no_action() {
    let prediction = prediction("e1", "no_action", true);
    let (_, audit) = run_shadow_execution(
        &prediction,
        json!({
            "event_id": "e1",
            "kind": "ignore",
            "title": "no action",
            "datetime_hint": null
        }),
        false,
    );
    assert!(!audit.disagreed);
}

#[test]
fn patch_gate_rejects_quality_regression() {
    let input = gate_input(0.95, 0.80, GeneralizationScope::SemanticClass);
    let decision = evaluate_patch_gate(&PatchGateConfig::default(), &input);
    assert!(!decision.accepted);
}

#[test]
fn patch_gate_rejects_exact_only_improvement_when_generalization_required() {
    let input = gate_input(0.95, 0.95, GeneralizationScope::Exact);
    let decision = evaluate_patch_gate(&PatchGateConfig::default(), &input);
    assert!(!decision.accepted);
}

#[test]
fn patch_gate_accepts_noninferior_generalized_patch() {
    let input = gate_input(0.95, 0.94, GeneralizationScope::SemanticClass);
    let decision = evaluate_patch_gate(&PatchGateConfig::default(), &input);
    assert!(decision.accepted, "{:?}", decision.reasons);
}

#[test]
fn variant_runner_runs_all_required_variants() {
    assert!(variants_include_all(&ExperimentVariant::canonical()));
}

#[test]
fn no_semantic_weak_variant_disables_semantic_matcher() {
    assert!(!ExperimentVariant::CpsGeneralizedPatchNoSemanticWeak.semantic_matcher_enabled());
}

#[test]
fn responses_client_extracts_usage() {
    let usage = extract_usage(&json!({ "usage": usage_json(10, 2, 3) })).unwrap();
    assert_eq!(usage.input_tokens, 10);
    assert_eq!(usage.cached_input_tokens, 3);
    assert_eq!(usage.output_tokens, 2);
}

#[tokio::test]
async fn responses_client_logs_latency() {
    let state = temp_state();
    let server = mock_structured_response(json!({"ok": true}), Some(usage_json(10, 2, 0)));
    let client = logged_client(&server, &state, ModelCacheMode::Disabled, default_catalog());
    let _: Value = client
        .create_structured_with_context(
            "gpt-5.4-mini",
            "Return JSON.",
            &json!({}),
            "object",
            json!({"type": "object"}),
            call_context("weak_model", "model_task"),
        )
        .await
        .unwrap()
        .parsed;
    let record = FileModelCallStore::new(state).list_all().unwrap().remove(0);
    assert!(record.latency_ms < 60_000);
}

fn temp_dir() -> PathBuf {
    let path = std::env::temp_dir().join(format!("cps-real-exp-test-{}", uuid::Uuid::new_v4()));
    fs::create_dir_all(&path).unwrap();
    path
}

fn temp_state() -> StateDir {
    StateDir::new(temp_dir())
}

fn default_catalog() -> PriceCatalog {
    let mut prices = BTreeMap::new();
    prices.insert(
        "gpt-5.5".to_owned(),
        ModelPrice {
            input: 5.0,
            cached_input: 0.5,
            output: 30.0,
        },
    );
    prices.insert(
        "gpt-5.4-mini".to_owned(),
        ModelPrice {
            input: 0.75,
            cached_input: 0.075,
            output: 4.5,
        },
    );
    PriceCatalog {
        prices_per_1m_tokens: prices,
    }
}

fn logged_client(
    server: &MockServer,
    state: &StateDir,
    cache_mode: ModelCacheMode,
    catalog: PriceCatalog,
) -> ResponsesClient {
    let runtime = ModelCallRuntime::new(FileModelCallStore::new(state.clone()), catalog)
        .with_budget(FileBudgetStore::new(state.clone()), BudgetConfig::default())
        .with_cache(ModelCache::new(
            state.root().join("model_cache"),
            cache_mode,
        ));
    ResponsesClient::new(ResponsesClientConfig {
        base_url: Url::parse(&server.url("/v1")).unwrap(),
        api_key: SecretString::from("test-key".to_owned()),
        runtime: Some(Arc::new(runtime)),
    })
}

fn mock_structured_response(output: Value, usage: Option<Value>) -> MockServer {
    let server = MockServer::start();
    server.mock(|when, then| {
        when.method(POST).path("/v1/responses");
        let mut body = json!({
            "output_text": serde_json::to_string(&output).unwrap(),
        });
        if let Some(usage) = usage {
            body["usage"] = usage;
        }
        then.status(200).json_body(body);
    });
    server
}

fn usage_json(input: u64, output: u64, cached: u64) -> Value {
    json!({
        "usage": {
            "input_tokens": input,
            "output_tokens": output,
            "total_tokens": input + output,
            "input_tokens_details": {
                "cached_tokens": cached
            }
        }
    })["usage"]
        .clone()
}

fn call_context(handler: &str, effect: &str) -> ModelCallContext {
    ModelCallContext {
        run_id: Some("run_1".to_owned()),
        workflow_id: Some("workflow".to_owned()),
        event_id: Some("e1".to_owned()),
        handler: handler.to_owned(),
        effect_kind: effect.to_owned(),
        task_name: Some("task".to_owned()),
        phase: Some("baseline".to_owned()),
    }
}

fn cache_key_for_schema(schema: Value) -> String {
    structured_cache_key(
        DEFAULT_BASE_URL,
        "gpt-5.4-mini",
        "instructions",
        &json!({"input": true}),
        &schema,
        &json!({"strict": true}),
    )
    .unwrap()
}

fn spend(
    call_id: &str,
    model: &str,
    phase: &str,
    cost: f64,
    cache_status: CacheStatus,
) -> BudgetSpendRecord {
    BudgetSpendRecord {
        call_id: call_id.to_owned(),
        run_id: Some("run".to_owned()),
        phase: Some(phase.to_owned()),
        model: model.to_owned(),
        cost_usd: cost,
        cache_status,
        created_at: now_string(),
    }
}

fn model_call_record(
    call_id: &str,
    run_id: &str,
    workflow_id: &str,
    phase: &str,
    cost: f64,
) -> ModelCallRecord {
    ModelCallRecord {
        call_id: call_id.to_owned(),
        run_id: Some(run_id.to_owned()),
        workflow_id: Some(workflow_id.to_owned()),
        event_id: Some("e0".to_owned()),
        model: "gpt-5.4-mini".to_owned(),
        handler: "weak_model".to_owned(),
        effect_kind: "model_task".to_owned(),
        task_name: Some("semantic_fast_path_match".to_owned()),
        phase: Some(phase.to_owned()),
        request_hash: call_id.to_owned(),
        prompt_hash: call_id.to_owned(),
        schema_hash: call_id.to_owned(),
        model_config_hash: call_id.to_owned(),
        input_bytes: 10,
        output_bytes: 20,
        usage: None,
        estimated_usage: ModelUsage::zero(),
        cost: CostBreakdown {
            input_usd: cost,
            cached_input_usd: 0.0,
            output_usd: 0.0,
            total_usd: cost,
            estimated: false,
        },
        latency_ms: 1,
        cache_status: CacheStatus::Miss,
        success: true,
        error: None,
        created_at: now_string(),
    }
}

fn run_metrics(run_id: &str, workflow_id: &str, p95: u64) -> RunMetrics {
    let mut metrics = RunMetrics::new(
        run_id.to_owned(),
        workflow_id.to_owned(),
        "stream".to_owned(),
        "program".to_owned(),
        "v0002".to_owned(),
        now_string(),
    );
    metrics.events_total = 1;
    metrics.events_succeeded = 1;
    metrics.continuation_frames_total = 1;
    metrics.continuation_frame_bytes_p50 = p95 / 2;
    metrics.continuation_frame_bytes_p95 = p95;
    metrics.continuation_frame_bytes_max = p95;
    metrics.finished_at = now_string();
    metrics
}

fn experiment_config(dir: &Path, price: &Path, hard_cap: f64) -> ExperimentConfig {
    ExperimentConfig {
        experiment_id: "notification_triage_real_v1".to_owned(),
        workflow_id: "notification_triage".to_owned(),
        state_dir: Some(dir.join("state")),
        workflow: None,
        models: ExperimentModels {
            weak_model: "gpt-5.4-mini".to_owned(),
            strong_model: "gpt-5.5".to_owned(),
            base_url_env: "CPS_TEST_OPENAI_BASE_URL_MISSING".to_owned(),
            api_key_env: "CPS_TEST_OPENAI_API_KEY_MISSING".to_owned(),
            use_responses_api: true,
            structured_outputs: true,
        },
        budget: ExperimentBudget {
            price_catalog: price.to_path_buf(),
            soft_cap_usd: 60.0,
            hard_cap_usd: hard_cap,
            projection_multiplier: 1.5,
        },
        cache: ExperimentCache {
            dir: dir.join("model_cache"),
            mode: "read_write".to_owned(),
        },
        schemas: ExperimentSchemas {
            event_schema: dir.join("event.schema.json"),
            output_schema: dir.join("output.schema.json"),
            gold_schema: dir.join("gold.schema.json"),
        },
        data: ExperimentData {
            all_events: dir.join("all_events.jsonl"),
            gold_labels: dir.join("gold_labels.jsonl"),
            splits_dir: dir.join("splits"),
            optimizer_hard_negative_evidence: None,
        },
        split_counts: None,
        phases: vec![
            "split_events".to_owned(),
            "quality_eval".to_owned(),
            "report".to_owned(),
        ],
        shadow: None,
        patch_gate: None,
    }
}

fn write_events_and_gold(dir: &Path, ordinary: usize, hard_negative: usize) {
    let mut events = Vec::new();
    let mut labels = Vec::new();
    for index in 0..ordinary + hard_negative {
        let id = format!("e{index}");
        let is_hard_negative = index >= ordinary;
        let kind = if is_hard_negative {
            "no_action"
        } else {
            "create_task"
        };
        let mut label = gold(&id, kind, !is_hard_negative, is_hard_negative);
        if is_hard_negative {
            label.title_canonical = Some("no action".to_owned());
            label.datetime_canonical = None;
        }
        label.semantic_cluster = if is_hard_negative {
            "hard_negative_cluster".to_owned()
        } else {
            format!("cluster_{}", index % 2)
        };
        let fixture_value = json!({
            "event_id": id,
            "kind": label.kind.clone(),
            "title": label.title_canonical.clone(),
            "datetime_hint": label.datetime_canonical.clone()
        });
        let mut event = event(&id, &format!("message {index}"), &format!("t{index}"));
        event["_fixture_model"] = json!({
            "weak": {
                "value": fixture_value,
                "confidence": 1.0
            },
            "strong": {
                "value": fixture_value,
                "confidence": 1.0
            }
        });
        event["_fixture_model_by_task"] = json!({
            "weak": {
                "semantic_fast_path_match": {
                    "value": {
                        "matched": true,
                        "rule_id": format!("{}_v1", label.semantic_cluster),
                        "slots": {
                            "kind": if label.kind == "no_action" { "ignore" } else { label.kind.as_str() },
                            "title": label.title_canonical.clone().unwrap_or_else(|| "no action".to_owned()),
                            "datetime_hint": label.datetime_canonical.clone()
                        },
                        "confidence": 0.95,
                        "rationale": "fixture semantic match"
                    },
                    "confidence": 1.0
                }
            }
        });
        events.push(event);
        labels.push(label);
    }
    write_jsonl(&dir.join("all_events.jsonl"), &events);
    write_jsonl(&dir.join("gold_labels.jsonl"), &labels);
}

fn write_fixture_program_and_task(dir: &Path) {
    fs::write(
        dir.join("program.json"),
        serde_json::to_vec_pretty(&weak_program()).unwrap(),
    )
    .unwrap();
    fs::write(
        dir.join("task.md"),
        "Return one action draft for the event.",
    )
    .unwrap();
}

fn write_experiment_schemas(dir: &Path) {
    fs::write(
        dir.join("event.schema.json"),
        serde_json::to_vec_pretty(&json!({"type":"object"})).unwrap(),
    )
    .unwrap();
    fs::write(
        dir.join("output.schema.json"),
        serde_json::to_vec_pretty(&action_schema()).unwrap(),
    )
    .unwrap();
    fs::write(
        dir.join("gold.schema.json"),
        serde_json::to_vec_pretty(&json!({"type":"object"})).unwrap(),
    )
    .unwrap();
}

fn event(id: &str, text: &str, timestamp: &str) -> Value {
    json!({
        "event_id": id,
        "timestamp": timestamp,
        "source": "chat",
        "sender_hint": "sender",
        "text": text
    })
}

fn labels_for(events: &[Value], hard_negative: bool) -> Vec<GoldLabel> {
    events
        .iter()
        .map(|event| {
            gold(
                event.get("event_id").and_then(Value::as_str).unwrap(),
                "create_task",
                true,
                hard_negative,
            )
        })
        .collect()
}

fn gold(id: &str, kind: &str, actionable: bool, hard_negative: bool) -> GoldLabel {
    GoldLabel {
        event_id: id.to_owned(),
        semantic_cluster: "cluster".to_owned(),
        is_actionable: actionable,
        kind: kind.to_owned(),
        title_canonical: Some("title".to_owned()),
        datetime_canonical: Some("tomorrow".to_owned()),
        criticality: "normal".to_owned(),
        hard_negative,
        generated_from_predictions: false,
    }
}

fn prediction(id: &str, kind: &str, schema_valid: bool) -> PredictionRecord {
    PredictionRecord {
        event_id: id.to_owned(),
        variant: "variant".to_owned(),
        program_version: "v0001".to_owned(),
        output: json!({
            "event_id": id,
            "kind": kind,
            "title": "title",
            "datetime_hint": "tomorrow"
        }),
        schema_valid,
        trace_run_id: Some("run".to_owned()),
        fast_path: FastPathPredictionMetadata::miss(),
        model_calls: ModelCallCounts::default(),
    }
}

async fn run_prediction_stream_and_read() -> Vec<PredictionRecord> {
    let state = temp_state();
    let workflow = "prediction_stream";
    let program = fast_path_program();
    FileProgramRegistry::new(state.clone())
        .init_workflow(
            workflow,
            program.clone(),
            fixture_program_metadata(workflow, &program),
        )
        .unwrap();
    let store = PredictionStore::new(state.root().join("predictions"));
    run_stream_with_options(
        state.clone(),
        workflow,
        InMemoryEventSource::new(vec![json!({"event_id": "e1", "source": "chat"})]),
        Arc::new(FixtureModelHandler::weak()),
        Arc::new(FixtureModelHandler::strong()),
        false,
        RunStreamOptions {
            run_id: None,
            predictions: Some(PredictionWriteOptions {
                store,
                file_name: "cps_generalized_patch.heldout.jsonl".to_owned(),
                variant: "cps_generalized_patch".to_owned(),
            }),
        },
    )
    .await
    .unwrap();
    PredictionStore::read(
        &state
            .root()
            .join("predictions")
            .join("cps_generalized_patch.heldout.jsonl"),
    )
    .unwrap()
}

fn fast_path_program() -> Program {
    Program {
        program_id: "fast_path_prediction".to_owned(),
        version: "v0001".to_owned(),
        entry: "main".to_owned(),
        input_schema: json!({"type": "object"}),
        output_schema: action_schema(),
        allowed_effects: vec![EffectPermission::LocalTool {
            tool_name: "fast_path_apply".to_owned(),
        }],
        functions: BTreeMap::from([(
            "main".to_owned(),
            FunctionDef {
                params: vec!["event".to_owned()],
                output_schema: action_schema(),
                body: vec![
                    Instr::Perform {
                        out: "fast".to_owned(),
                        effect: EffectCall::LocalTool {
                            tool_name: "fast_path_apply".to_owned(),
                            args_schema: json!({ "type": "object" }),
                        },
                        input: JsonExpr::Literal {
                            value: json!({
                                "event": { "event_id": "e1", "source": "chat" },
                                "rules": [{
                                    "rule_id": "r1",
                                    "when": {
                                        "op": "field_equals",
                                        "path": ["source"],
                                        "value": "chat"
                                    },
                                    "emit": {
                                        "kind": "literal",
                                        "value": {
                                            "event_id": "e1",
                                            "kind": "create_task",
                                            "title": "title",
                                            "datetime_hint": "tomorrow"
                                        }
                                    },
                                    "confidence": 1.0
                                }]
                            }),
                        },
                        expected_schema: json!({"type": "object"}),
                        acceptance: accept(),
                    },
                    Instr::Branch {
                        condition: GuardExpr::FieldIsTruthy {
                            var: "fast".to_owned(),
                            path: vec!["hit".to_owned()],
                        },
                        then_pc: 2,
                        else_pc: 4,
                    },
                    Instr::Project {
                        out: "value".to_owned(),
                        from: JsonExpr::Var {
                            name: "fast".to_owned(),
                        },
                        path: vec!["value".to_owned()],
                    },
                    Instr::Return {
                        value: JsonExpr::Var {
                            name: "value".to_owned(),
                        },
                    },
                    Instr::Return {
                        value: JsonExpr::Literal {
                            value: json!({
                                "event_id": "e1",
                                "kind": "no_action",
                                "title": "none",
                                "datetime_hint": null
                            }),
                        },
                    },
                ],
            },
        )]),
    }
}

fn weak_program() -> Program {
    Program {
        program_id: "notification_triage".to_owned(),
        version: "v0001".to_owned(),
        entry: "main".to_owned(),
        input_schema: json!({"type": "object"}),
        output_schema: action_schema(),
        allowed_effects: vec![EffectPermission::ModelTask {
            strength: ModelStrength::Weak,
        }],
        functions: BTreeMap::from([(
            "main".to_owned(),
            FunctionDef {
                params: vec!["event".to_owned()],
                output_schema: action_schema(),
                body: vec![
                    Instr::Perform {
                        out: "draft".to_owned(),
                        effect: EffectCall::ModelTask {
                            strength: ModelStrength::Weak,
                            task: ModelTaskSpec {
                                name: "draft_action_from_event".to_owned(),
                                instructions: "Return one action draft.".to_owned(),
                            },
                        },
                        input: JsonExpr::Var {
                            name: "event".to_owned(),
                        },
                        expected_schema: action_schema(),
                        acceptance: accept(),
                    },
                    Instr::Return {
                        value: JsonExpr::Var {
                            name: "draft".to_owned(),
                        },
                    },
                ],
            },
        )]),
    }
}

fn semantic_patch() -> ProgramPatch {
    ProgramPatch {
        target_program_id: "p".to_owned(),
        patch_id: "semantic_v1".to_owned(),
        rationale: "semantic fast path".to_owned(),
        generalization: Some(generalized_semantic_metadata(
            vec!["cluster".to_owned()],
            vec![],
        )),
        operations: vec![PatchOp::InsertInstruction {
            function: "main".to_owned(),
            pc: 0,
            instr: Instr::Perform {
                out: "match".to_owned(),
                effect: EffectCall::ModelTask {
                    strength: ModelStrength::Weak,
                    task: ModelTaskSpec {
                        name: SEMANTIC_FAST_PATH_TASK.to_owned(),
                        instructions: "match semantic class".to_owned(),
                    },
                },
                input: JsonExpr::Var {
                    name: "event".to_owned(),
                },
                expected_schema: json!({"type": "object"}),
                acceptance: accept(),
            },
        }],
    }
}

fn action_schema() -> Value {
    json!({
        "type": "object",
        "required": ["event_id", "kind", "title"],
        "properties": {
            "event_id": { "type": "string" },
            "kind": { "type": "string" },
            "title": { "type": "string" },
            "datetime_hint": {}
        }
    })
}

fn accept() -> AcceptancePolicy {
    AcceptancePolicy {
        min_confidence: Some(0.0),
        require_schema_valid: true,
        on_failure: FailureHandler::Abort {
            reason: "failed".to_owned(),
        },
    }
}

fn gate_input(
    baseline_quality: f64,
    candidate_quality: f64,
    scope: GeneralizationScope,
) -> PatchGateInput {
    PatchGateInput {
        baseline_quality,
        candidate_quality,
        baseline_critical_miss_rate: 0.01,
        candidate_critical_miss_rate: 0.01,
        false_fast_path_rate: 0.001,
        fast_path_hit_rate_lift: 0.20,
        strong_think_rate_reduction: 0.30,
        continuation_frame_p95_bytes: 8_000,
        metadata: PatchGeneralizationMetadata {
            patch_kind: PatchKind::WeakSemanticFastPath,
            generalization_scope: scope,
            uses_weak_semantic_matcher: true,
            uses_deterministic_fast_path: false,
            uses_validator: true,
            declared_positive_clusters: vec!["cluster".to_owned()],
            declared_negative_clusters: Vec::new(),
        },
    }
}

fn row(
    variant: &str,
    quality: f64,
    strong_calls_per_event: f64,
    weak_calls_per_event: f64,
    fast_path_hit_rate: f64,
) -> VariantReportRow {
    VariantReportRow {
        variant: variant.to_owned(),
        quality,
        critical_miss: 0.01,
        strong_calls_per_event,
        weak_calls_per_event,
        fast_path_hit_rate,
        false_fast_path_rate: Some(0.001),
        shadow_disagreement_rate: Some(0.01),
        p95_frame_bytes: Some(8_000),
        api_spend_usd: 1.0,
    }
}

fn write_jsonl<T: serde::Serialize>(path: &Path, values: &[T]) {
    let raw = values
        .iter()
        .map(|value| serde_json::to_string(value).unwrap())
        .collect::<Vec<_>>()
        .join("\n");
    fs::write(path, format!("{raw}\n")).unwrap();
}
