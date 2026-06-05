use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result, anyhow};
use async_trait::async_trait;
use regex::Regex;
use secrecy::SecretString;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use tokio::task::JoinSet;
use url::Url;

use crate::config::DEFAULT_BASE_URL;
use crate::effects::{HandlerBudget, HandlerDecision, HandlerRequest, RuntimeBudget};
use crate::engine::event_source::JsonlEventSource;
use crate::engine::run_coordinator::{
    PredictionWriteOptions, RunStreamOptions, run_stream, run_stream_with_options,
};
use crate::experiment::config::{
    ExperimentConfig, ExperimentSplitCounts, PatchGateConfigYaml, load_experiment_config,
    write_locked_config,
};
use crate::experiment::exact_memo::{ExactMemoEntry, ExactMemoTable, normalized_text_hash};
use crate::experiment::patch_gate::{
    PatchGateConfig, PatchGateDecision, PatchGateInput, evaluate_patch_gate,
};
use crate::experiment::prediction::{
    FastPathPredictionMetadata, ModelCallCounts, PredictionRecord, PredictionStore,
    prediction_metadata_from_trace,
};
use crate::experiment::quality::{
    GoldLabel, QualityMetrics, evaluate_quality_files, read_gold_labels,
};
use crate::experiment::report::{
    ExperimentPassFail, ExperimentReport, VariantReportRow, build_pass_fail, write_report,
};
use crate::experiment::semantic_fast_path::{
    SEMANTIC_FAST_PATH_TASK, generalized_semantic_metadata,
};
use crate::experiment::shadow::{shadow_execution_does_not_change_output, summarize_shadow_audit};
use crate::experiment::split::{SplitCounts, SplitStrategy, split_events_files};
use crate::experiment::variants::ExperimentVariant;
use crate::model_cache::{ModelCache, ModelCacheMode};
use crate::models::{EffectHandler, FixtureModelHandler, ResponsesStrongModel, ResponsesWeakModel};
use crate::pricing::price_catalog::PriceCatalog;
use crate::program::{
    AcceptancePolicy, EffectCall, EffectPermission, FailureHandler, GuardExpr, Instr, JsonExpr,
    ModelStrength, ModelTaskSpec, PatchOp, Program, ProgramPatch,
};
use crate::responses_client::{ModelCallRuntime, ResponsesClient, ResponsesClientConfig};
use crate::runtime::Runtime;
use crate::schema::validate_value;
use crate::store::budget_store::{BudgetReport, FileBudgetStore};
use crate::store::model_call_store::{FileModelCallStore, ModelCallRecord};
use crate::store::patch_registry::{
    FilePatchRegistry, PatchEvaluationMetrics, PatchMetadata, PatchSource, PatchStatus,
};
use crate::store::program_registry::{
    FileProgramRegistry, ProgramMetadata, ProgramSource, fixture_program_metadata,
};
use crate::store::state_dir::{StateDir, now_string, read_json, write_json_pretty};
use crate::trace::TraceCollector;
use crate::validator::validate_patch;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RunManifest {
    pub experiment_id: String,
    pub workflow_id: String,
    pub started_at: String,
    #[serde(default)]
    pub completed_phases: Vec<String>,
    #[serde(default)]
    pub skipped_phases: Vec<String>,
    pub dry_run_cost: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DryRunCostEstimate {
    pub estimated_total_usd: f64,
    pub hard_cap_usd: f64,
    pub fits_budget: bool,
    pub largest_phase: String,
    pub recommendation: String,
}

pub async fn run_experiment_from_file(
    config_path: &Path,
    state_dir: StateDir,
    dry_run_cost: bool,
) -> Result<serde_json::Value> {
    let config = load_experiment_config(config_path)?;
    run_experiment(config, state_dir, dry_run_cost).await
}

pub async fn run_experiment(
    config: ExperimentConfig,
    state_dir: StateDir,
    dry_run_cost: bool,
) -> Result<serde_json::Value> {
    let experiment_dir = experiment_dir(&state_dir, &config.experiment_id);
    std::fs::create_dir_all(&experiment_dir)
        .with_context(|| format!("failed to create {}", experiment_dir.display()))?;
    write_locked_config(&config, &experiment_dir.join("config.lock.yaml"))?;
    let catalog = PriceCatalog::load(&config.budget.price_catalog)
        .unwrap_or_else(|_| PriceCatalog::default_openai());
    catalog.write_yaml(&experiment_dir.join("price_catalog.lock.yaml"))?;
    let budget_store = FileBudgetStore::new(state_dir.clone());
    let budget_config = config.budget.budget_config();
    budget_store.write_config(&budget_config)?;

    if dry_run_cost {
        let estimate = dry_run_estimate(&config, &catalog)?;
        return Ok(serde_json::to_value(estimate)?);
    }

    let estimate = dry_run_estimate(&config, &catalog)?;
    if !estimate.fits_budget {
        return Err(anyhow!(
            "run-experiment refused to start: projected ${:.4} exceeds hard cap ${:.4}",
            estimate.estimated_total_usd,
            estimate.hard_cap_usd
        ));
    }

    let manifest_path = experiment_dir.join("run_manifest.json");
    let mut manifest = if manifest_path.exists() {
        read_json::<RunManifest>(&manifest_path)?
    } else {
        RunManifest {
            experiment_id: config.experiment_id.clone(),
            workflow_id: config.workflow_id.clone(),
            started_at: now_string(),
            completed_phases: Vec::new(),
            skipped_phases: Vec::new(),
            dry_run_cost: false,
        }
    };
    let completed = manifest
        .completed_phases
        .iter()
        .cloned()
        .collect::<BTreeSet<_>>();
    let mut skipped = manifest
        .skipped_phases
        .iter()
        .cloned()
        .collect::<BTreeSet<_>>();
    ensure_workflow_initialized(&state_dir, &config)?;
    let (weak, strong) = handler_pair_for_experiment(&config, &state_dir, &catalog)?;
    for phase in &config.phases {
        if completed.contains(phase) {
            continue;
        }
        match execute_phase(
            &config,
            &state_dir,
            &experiment_dir,
            phase,
            Arc::clone(&weak),
            Arc::clone(&strong),
        )
        .await?
        {
            PhaseOutcome::Completed => {
                write_phase_marker(&experiment_dir, phase, "completed")?;
                manifest.completed_phases.push(phase.clone());
                write_json_pretty(&manifest_path, &manifest)?;
            }
            PhaseOutcome::Skipped(reason) => {
                if skipped.insert(phase.clone()) {
                    write_skipped_phase_marker(&experiment_dir, phase, &reason)?;
                    manifest.skipped_phases.push(phase.clone());
                    write_json_pretty(&manifest_path, &manifest)?;
                }
            }
        }
    }
    if !experiment_dir.join("report.json").exists() {
        write_minimal_report(
            &experiment_dir,
            &config,
            &budget_store.report(&budget_config)?,
        )?;
    }
    Ok(json!({
        "ok": true,
        "experiment_id": config.experiment_id,
        "completed_phases": manifest.completed_phases,
        "skipped_phases": manifest.skipped_phases,
    }))
}

enum PhaseOutcome {
    Completed,
    Skipped(String),
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
struct SemanticRule {
    rule_id: String,
    semantic_cluster: String,
    kind: String,
    title: Option<String>,
    datetime_hint: Option<String>,
    examples: Vec<String>,
    negative_examples: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
struct SemanticNegativeGuard {
    guard_id: String,
    rule_id: String,
    pattern: String,
    #[serde(default)]
    rationale: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
struct SemanticPatchPlan {
    patch_id: String,
    rationale: String,
    rules: Vec<SemanticRule>,
    #[serde(default)]
    negative_guards: Vec<SemanticNegativeGuard>,
    optimizer_source: String,
}

const STRONG_MODEL_GENERATED_SEMANTIC_PATCH_PLAN: &str =
    "strong_model_generated_semantic_patch_plan";

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SemanticMatchOutput {
    matched: bool,
    rule_id: String,
    slots: Value,
    confidence: f64,
    rationale: String,
}

async fn execute_phase(
    config: &ExperimentConfig,
    state_dir: &StateDir,
    experiment_dir: &Path,
    phase: &str,
    weak: Arc<dyn EffectHandler>,
    strong: Arc<dyn EffectHandler>,
) -> Result<PhaseOutcome> {
    match phase {
        "split_events" => {
            let counts = config
                .split_counts
                .clone()
                .unwrap_or(ExperimentSplitCounts {
                    profile_train: 120,
                    patch_validation: 80,
                    heldout_test: 160,
                    adversarial_test: 40,
                });
            split_events_files(
                &config.data.all_events,
                &config.data.gold_labels,
                &config.data.splits_dir,
                SplitStrategy::TimeCluster,
                SplitCounts {
                    profile_train: counts.profile_train,
                    patch_validation: counts.patch_validation,
                    heldout_test: counts.heldout_test,
                    adversarial_test: counts.adversarial_test,
                },
            )?;
            Ok(PhaseOutcome::Completed)
        }
        "weak_only" => {
            run_direct_variant(
                config,
                experiment_dir,
                ExperimentVariant::WeakOnly,
                ModelStrength::Weak,
                weak,
            )
            .await?;
            Ok(PhaseOutcome::Completed)
        }
        "strong_direct" => {
            run_direct_variant(
                config,
                experiment_dir,
                ExperimentVariant::StrongDirect,
                ModelStrength::Strong,
                strong,
            )
            .await?;
            Ok(PhaseOutcome::Completed)
        }
        "cps_unoptimized_profile" => {
            let events = config.data.splits_dir.join("profile_train.events.jsonl");
            let source = JsonlEventSource::from_path(&events)?;
            let run_id = uuid::Uuid::new_v4().to_string();
            let weak = attributed_handler(
                weak,
                run_id.clone(),
                config.workflow_id.clone(),
                "cps_unoptimized_profile",
            );
            let strong = attributed_handler(
                strong,
                run_id,
                config.workflow_id.clone(),
                "cps_unoptimized_profile",
            );
            run_stream(
                state_dir.clone(),
                &config.workflow_id,
                source,
                weak,
                strong,
                false,
            )
            .await?;
            Ok(PhaseOutcome::Completed)
        }
        "cps_unoptimized_heldout" => {
            let events = config.data.splits_dir.join("heldout_test.events.jsonl");
            let source = JsonlEventSource::from_path(&events)?;
            let run_id = uuid::Uuid::new_v4().to_string();
            let weak = attributed_handler(
                weak,
                run_id.clone(),
                config.workflow_id.clone(),
                "cps_unoptimized_heldout",
            );
            let strong = attributed_handler(
                strong,
                run_id,
                config.workflow_id.clone(),
                "cps_unoptimized_heldout",
            );
            run_stream_with_options(
                state_dir.clone(),
                &config.workflow_id,
                source,
                weak,
                strong,
                false,
                RunStreamOptions {
                    predictions: Some(PredictionWriteOptions {
                        store: PredictionStore::new(experiment_dir.join("predictions")),
                        file_name: format!(
                            "{}.heldout.jsonl",
                            ExperimentVariant::CpsUnoptimized.as_str()
                        ),
                        variant: ExperimentVariant::CpsUnoptimized.as_str().to_owned(),
                    }),
                },
            )
            .await?;
            Ok(PhaseOutcome::Completed)
        }
        "optimize" => {
            write_optimization_artifacts(config, experiment_dir, strong).await?;
            Ok(PhaseOutcome::Completed)
        }
        "patch_validation" => {
            run_patch_validation_phase(config, state_dir, experiment_dir, weak, strong).await?;
            Ok(PhaseOutcome::Completed)
        }
        "cps_exact_memo_heldout" => {
            run_exact_memo_variant(config, experiment_dir, strong).await?;
            Ok(PhaseOutcome::Completed)
        }
        "cps_generalized_patch_heldout" => {
            run_installed_patch_variant(
                config,
                state_dir,
                experiment_dir,
                "heldout_test",
                ExperimentVariant::CpsGeneralizedPatch,
                weak,
                strong,
            )
            .await?;
            Ok(PhaseOutcome::Completed)
        }
        "cps_generalized_patch_no_semantic_weak" => {
            run_semantic_patch_variant(
                config,
                experiment_dir,
                "heldout_test",
                ExperimentVariant::CpsGeneralizedPatchNoSemanticWeak,
                false,
                weak,
                strong,
            )
            .await?;
            Ok(PhaseOutcome::Completed)
        }
        "adversarial_audit" => {
            run_installed_patch_variant(
                config,
                state_dir,
                experiment_dir,
                "adversarial_test",
                ExperimentVariant::CpsGeneralizedPatch,
                weak,
                strong,
            )
            .await?;
            Ok(PhaseOutcome::Completed)
        }
        "shadow_audit" => {
            run_shadow_audit(config, experiment_dir, strong).await?;
            Ok(PhaseOutcome::Completed)
        }
        "quality_eval" => {
            evaluate_present_predictions(experiment_dir, &config.data.splits_dir)?;
            Ok(PhaseOutcome::Completed)
        }
        "report" => {
            write_experiment_report(config, experiment_dir, state_dir)?;
            Ok(PhaseOutcome::Completed)
        }
        other => Ok(PhaseOutcome::Skipped(format!(
            "phase {other} is not implemented by the real variant executor yet"
        ))),
    }
}

fn ensure_workflow_initialized(state_dir: &StateDir, config: &ExperimentConfig) -> Result<()> {
    let Some(workflow) = &config.workflow else {
        return Ok(());
    };
    if state_dir.workflow_dir(&config.workflow_id)?.exists() {
        return Ok(());
    }
    let program: Program = read_json(&workflow.program)?;
    FileProgramRegistry::new(state_dir.clone()).init_workflow(
        &config.workflow_id,
        program.clone(),
        fixture_program_metadata(&config.workflow_id, &program),
    )
}

fn handler_pair_for_experiment(
    config: &ExperimentConfig,
    state_dir: &StateDir,
    catalog: &PriceCatalog,
) -> Result<(Arc<dyn EffectHandler>, Arc<dyn EffectHandler>)> {
    let api_key = env::var(&config.models.api_key_env)
        .ok()
        .filter(|value| !value.trim().is_empty());
    let Some(api_key) = api_key else {
        return Ok((
            Arc::new(FixtureModelHandler::weak()),
            Arc::new(FixtureModelHandler::strong()),
        ));
    };
    let base_url =
        env::var(&config.models.base_url_env).unwrap_or_else(|_| DEFAULT_BASE_URL.to_owned());
    let client = responses_client_for_experiment(config, state_dir, catalog, base_url, api_key)?;
    Ok((
        Arc::new(ResponsesWeakModel::new(
            client.clone(),
            config.models.weak_model.clone(),
        )),
        Arc::new(ResponsesStrongModel::new(
            client,
            config.models.strong_model.clone(),
        )),
    ))
}

fn responses_client_for_experiment(
    config: &ExperimentConfig,
    state_dir: &StateDir,
    catalog: &PriceCatalog,
    base_url: String,
    api_key: String,
) -> Result<ResponsesClient> {
    let cache_mode = match config.cache.mode.as_str() {
        "read_write" => ModelCacheMode::ReadWrite,
        "read_only" | "readonly" => ModelCacheMode::ReadOnly,
        "refresh" => ModelCacheMode::Refresh,
        "disabled" | "none" => ModelCacheMode::Disabled,
        other => return Err(anyhow!("unsupported model cache mode {other}")),
    };
    let budget_store = FileBudgetStore::new(state_dir.clone());
    let runtime =
        ModelCallRuntime::new(FileModelCallStore::new(state_dir.clone()), catalog.clone())
            .with_budget(budget_store, config.budget.budget_config())
            .with_cache(ModelCache::new(config.cache.dir.clone(), cache_mode));
    Ok(ResponsesClient::new(ResponsesClientConfig {
        base_url: Url::parse(&base_url).context("invalid experiment base URL")?,
        api_key: SecretString::from(api_key),
        runtime: Some(Arc::new(runtime)),
    }))
}

#[derive(Clone)]
struct AttributedEffectHandler {
    inner: Arc<dyn EffectHandler>,
    run_id: String,
    workflow_id: String,
    phase: String,
}

#[async_trait]
impl EffectHandler for AttributedEffectHandler {
    async fn handle(&self, mut request: HandlerRequest) -> Result<HandlerDecision> {
        request.run_id.get_or_insert_with(|| self.run_id.clone());
        request
            .workflow_id
            .get_or_insert_with(|| self.workflow_id.clone());
        request.phase.get_or_insert_with(|| self.phase.clone());
        self.inner.handle(request).await
    }
}

fn attributed_handler(
    inner: Arc<dyn EffectHandler>,
    run_id: String,
    workflow_id: String,
    phase: impl Into<String>,
) -> Arc<dyn EffectHandler> {
    Arc::new(AttributedEffectHandler {
        inner,
        run_id,
        workflow_id,
        phase: phase.into(),
    })
}

async fn run_direct_variant(
    config: &ExperimentConfig,
    experiment_dir: &Path,
    variant: ExperimentVariant,
    strength: ModelStrength,
    handler: Arc<dyn EffectHandler>,
) -> Result<()> {
    let output_schema: Value = read_json(&config.schemas.output_schema)?;
    let task_spec = experiment_task_spec(config)?;
    let events = read_jsonl_values(&config.data.splits_dir.join("heldout_test.events.jsonl"))?;
    let predictions_dir = experiment_dir.join("predictions");
    let file_name = format!("{}.heldout.jsonl", variant.as_str());
    let prediction_path = predictions_dir.join(&file_name);
    if prediction_path.exists() {
        std::fs::remove_file(&prediction_path)
            .with_context(|| format!("failed to remove {}", prediction_path.display()))?;
    }
    let store = PredictionStore::new(predictions_dir);
    let run_id = uuid::Uuid::new_v4().to_string();
    let handler = attributed_handler(
        handler,
        run_id.clone(),
        config.workflow_id.clone(),
        variant.as_str(),
    );
    let mut records = Vec::new();
    let mut indexed_events = events.into_iter().enumerate();
    let mut tasks = JoinSet::new();
    spawn_direct_variant_tasks(
        &mut tasks,
        &mut indexed_events,
        experiment_concurrency(),
        DirectVariantTask {
            variant,
            strength,
            task_spec: task_spec.clone(),
            output_schema: output_schema.clone(),
            handler: Arc::clone(&handler),
            run_id: run_id.clone(),
        },
    );
    while let Some(joined) = tasks.join_next().await {
        let (index, record) = joined.context("direct variant task panicked")??;
        records.push((index, record));
        spawn_direct_variant_tasks(
            &mut tasks,
            &mut indexed_events,
            1,
            DirectVariantTask {
                variant,
                strength,
                task_spec: task_spec.clone(),
                output_schema: output_schema.clone(),
                handler: Arc::clone(&handler),
                run_id: run_id.clone(),
            },
        );
    }
    records.sort_by_key(|(index, _)| *index);
    for (_, record) in records {
        store.append(&file_name, &record)?;
    }
    Ok(())
}

#[derive(Clone)]
struct DirectVariantTask {
    variant: ExperimentVariant,
    strength: ModelStrength,
    task_spec: String,
    output_schema: Value,
    handler: Arc<dyn EffectHandler>,
    run_id: String,
}

fn spawn_direct_variant_tasks(
    tasks: &mut JoinSet<Result<(usize, PredictionRecord)>>,
    indexed_events: &mut impl Iterator<Item = (usize, Value)>,
    limit: usize,
    task: DirectVariantTask,
) {
    for _ in 0..limit {
        let Some((index, event)) = indexed_events.next() else {
            break;
        };
        let task = task.clone();
        tasks.spawn(async move { run_direct_variant_event(index, event, task).await });
    }
}

async fn run_direct_variant_event(
    index: usize,
    event: Value,
    task: DirectVariantTask,
) -> Result<(usize, PredictionRecord)> {
    let event_id = event_id_or_generate(&event);
    let request = HandlerRequest {
        run_id: Some(task.run_id.clone()),
        workflow_id: None,
        phase: Some(task.variant.as_str().to_owned()),
        effect: EffectCall::ModelTask {
            strength: task.strength,
            task: ModelTaskSpec {
                name: task.variant.as_str().to_owned(),
                instructions: task.task_spec,
            },
        },
        input: event,
        expected_schema: task.output_schema.clone(),
        continuation_summary: None,
        effect_frame: None,
        observations: Vec::new(),
        budget: HandlerBudget {
            effect_depth: 0,
            effects_remaining: RuntimeBudget::default().max_effects,
            handler_reentries_remaining: RuntimeBudget::default().max_handler_reentries,
        },
    };
    let (output, schema_valid) = match task.handler.handle(request).await {
        Ok(HandlerDecision::ReturnValue { value, .. }) => {
            let schema_valid = validate_value(&task.output_schema, &value).is_ok();
            (value, schema_valid)
        }
        Ok(other) => (
            json!({
                "error": format!("handler returned {}", other.decision_name())
            }),
            false,
        ),
        Err(err) => (
            json!({
                "error": err.to_string()
            }),
            false,
        ),
    };
    Ok((
        index,
        PredictionRecord {
            event_id,
            variant: task.variant.as_str().to_owned(),
            program_version: match task.strength {
                ModelStrength::Weak => "weak_only".to_owned(),
                ModelStrength::Strong => "strong_direct".to_owned(),
            },
            output,
            schema_valid,
            trace_run_id: Some(task.run_id),
            fast_path: FastPathPredictionMetadata::miss(),
            model_calls: match task.strength {
                ModelStrength::Weak => ModelCallCounts {
                    weak: 1,
                    strong_think: 0,
                    strong_task: 0,
                },
                ModelStrength::Strong => ModelCallCounts {
                    weak: 0,
                    strong_think: 0,
                    strong_task: 1,
                },
            },
        },
    ))
}

fn experiment_concurrency() -> usize {
    env::var("CPS_EXPERIMENT_CONCURRENCY")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(4)
}

async fn write_optimization_artifacts(
    config: &ExperimentConfig,
    experiment_dir: &Path,
    strong: Arc<dyn EffectHandler>,
) -> Result<()> {
    let artifacts_dir = experiment_dir.join("artifacts");
    std::fs::create_dir_all(&artifacts_dir)
        .with_context(|| format!("failed to create {}", artifacts_dir.display()))?;
    let optimizer_input = build_semantic_optimizer_input(config)?;
    let plan = run_strong_semantic_patch_optimizer(config, experiment_dir, strong, optimizer_input)
        .await?;
    write_json_pretty(&artifacts_dir.join("semantic_patch_plan.json"), &plan)?;
    write_json_pretty(&artifacts_dir.join("semantic_rules.json"), &plan.rules)?;
    let memo = build_exact_memo_table(config)?;
    write_json_pretty(&artifacts_dir.join("exact_memo_table.json"), &memo)?;
    Ok(())
}

async fn ensure_optimization_artifacts(
    config: &ExperimentConfig,
    experiment_dir: &Path,
    strong: Arc<dyn EffectHandler>,
) -> Result<()> {
    let artifacts_dir = experiment_dir.join("artifacts");
    if artifacts_dir.join("semantic_patch_plan.json").exists()
        && artifacts_dir.join("semantic_rules.json").exists()
        && artifacts_dir.join("exact_memo_table.json").exists()
    {
        return Ok(());
    }
    write_optimization_artifacts(config, experiment_dir, strong).await
}

fn build_semantic_optimizer_input(config: &ExperimentConfig) -> Result<Value> {
    let clusters = build_profile_cluster_evidence(config)?;
    let hard_negative_examples = read_optimizer_hard_negative_evidence(config)?;
    Ok(json!({
        "workflow_id": config.workflow_id,
        "proposal_constraint": "Generate the generalized weak semantic fast-path ProgramPatch plan from evidence. Rust will only validate schema/safety/gates and render the typed ProgramPatch; the returned plan is the optimizer-generated patch data.",
        "optimizer_contract": {
            "patch_id": "semantic_patch_v1",
            "optimizer_source": STRONG_MODEL_GENERATED_SEMANTIC_PATCH_PLAN,
            "rule_id_format": "<semantic_cluster>_v1",
            "must_emit_rules_for_fast_path_eligible_clusters": true,
            "must_emit_negative_guards_for_failure_evidence": !hard_negative_examples.is_empty()
        },
        "profile_evidence": {
            "split": "profile_train",
            "clusters": clusters
        },
        "failure_cluster_evidence": {
            "source": config
                .data
                .optimizer_hard_negative_evidence
                .as_ref()
                .map(|path| path.display().to_string()),
            "hard_negative_examples": hard_negative_examples
        },
        "validation_constraints": {
            "allowed_output_kinds": ["ignore", "create_task", "create_calendar_event", "draft_reply"],
            "ineligible_profile_kinds": ["needs_review"],
            "regex_engine": "Rust regex crate",
            "negative_guard_semantics": "Each negative guard is rendered as NOT(rule_id equals guard.rule_id AND event.text matches guard.pattern)."
        }
    }))
}

fn build_profile_cluster_evidence(config: &ExperimentConfig) -> Result<Vec<Value>> {
    let events = read_jsonl_values(&config.data.splits_dir.join("profile_train.events.jsonl"))?;
    let gold = read_gold_labels(&config.data.splits_dir.join("profile_train.gold.jsonl"))?;
    let event_by_id = events
        .iter()
        .filter_map(|event| {
            event
                .get("event_id")
                .and_then(Value::as_str)
                .map(|id| (id, event))
        })
        .collect::<BTreeMap<_, _>>();
    let mut grouped: BTreeMap<String, Vec<&GoldLabel>> = BTreeMap::new();
    for label in &gold {
        grouped
            .entry(label.semantic_cluster.clone())
            .or_default()
            .push(label);
    }
    let mut rows = Vec::new();
    for (cluster, labels) in grouped {
        let Some(first) = labels.first() else {
            continue;
        };
        let positive_examples = labels
            .iter()
            .filter_map(|label| event_by_id.get(label.event_id.as_str()))
            .filter_map(|event| event.get("text").and_then(Value::as_str))
            .take(6)
            .map(ToOwned::to_owned)
            .collect::<Vec<_>>();
        let contraindication_examples = gold
            .iter()
            .filter(|label| label.semantic_cluster != cluster)
            .filter_map(|label| event_by_id.get(label.event_id.as_str()))
            .filter_map(|event| event.get("text").and_then(Value::as_str))
            .take(8)
            .map(ToOwned::to_owned)
            .collect::<Vec<_>>();
        rows.push(json!({
            "semantic_cluster": cluster,
            "suggested_rule_id": format!("{}_v1", first.semantic_cluster),
            "fast_path_eligible": label_kind_can_fast_path(first),
            "gold_kind": first.kind,
            "output_kind": output_kind_for_label(first),
            "title_canonical": first.title_canonical,
            "datetime_canonical": first.datetime_canonical,
            "profile_count": labels.len(),
            "positive_examples": positive_examples,
            "contraindication_examples": contraindication_examples
        }));
    }
    Ok(rows)
}

fn read_optimizer_hard_negative_evidence(config: &ExperimentConfig) -> Result<Vec<Value>> {
    let Some(path) = &config.data.optimizer_hard_negative_evidence else {
        return Ok(Vec::new());
    };
    if !path.exists() {
        return Ok(Vec::new());
    }
    read_jsonl_values(path)
}

async fn run_strong_semantic_patch_optimizer(
    config: &ExperimentConfig,
    experiment_dir: &Path,
    strong: Arc<dyn EffectHandler>,
    request_input: Value,
) -> Result<SemanticPatchPlan> {
    let artifacts_dir = experiment_dir.join("artifacts");
    std::fs::create_dir_all(&artifacts_dir)
        .with_context(|| format!("failed to create {}", artifacts_dir.display()))?;
    let optimizer_dir = artifacts_dir.join("optimizer");
    std::fs::create_dir_all(&optimizer_dir)
        .with_context(|| format!("failed to create {}", optimizer_dir.display()))?;
    write_json_pretty(
        &artifacts_dir.join("optimizer_semantic_patch_request.json"),
        &request_input,
    )?;
    write_json_pretty(
        &optimizer_dir.join("optimizer_request.json"),
        &request_input,
    )?;
    let run_id = uuid::Uuid::new_v4().to_string();
    let decision = strong
        .handle(HandlerRequest {
            run_id: Some(run_id),
            workflow_id: Some(config.workflow_id.clone()),
            phase: Some("optimize".to_owned()),
            effect: EffectCall::ModelTask {
                strength: ModelStrength::Strong,
                task: ModelTaskSpec {
                    name: "optimize_semantic_patch_plan".to_owned(),
                    instructions: STRONG_SEMANTIC_PATCH_OPTIMIZER_INSTRUCTIONS.to_owned(),
                },
            },
            input: request_input.clone(),
            expected_schema: semantic_patch_plan_schema(),
            continuation_summary: None,
            effect_frame: None,
            observations: Vec::new(),
            budget: HandlerBudget {
                effect_depth: 0,
                effects_remaining: RuntimeBudget::default().max_effects,
                handler_reentries_remaining: RuntimeBudget::default().max_handler_reentries,
            },
        })
        .await?;
    write_json_pretty(
        &artifacts_dir.join("optimizer_semantic_patch_response.json"),
        &decision,
    )?;
    write_json_pretty(&optimizer_dir.join("optimizer_response.json"), &decision)?;
    let HandlerDecision::ReturnValue { value, .. } = decision else {
        anyhow::bail!(
            "strong semantic patch optimizer returned {}, expected return_value",
            decision.decision_name()
        );
    };
    let plan: SemanticPatchPlan = serde_json::from_value(value)
        .context("strong semantic patch optimizer returned invalid plan")?;
    validate_semantic_patch_plan(&plan, &request_input)?;
    write_json_pretty(
        &artifacts_dir.join("semantic_patch_generation.json"),
        &json!({
            "generation_source": "optimizer_strong_model",
            "generator_model": config.models.strong_model,
            "phase": "optimize",
            "task_name": "optimize_semantic_patch_plan",
            "plan_optimizer_source": plan.optimizer_source,
            "raw_optimizer_response_is_final_plan": true
        }),
    )?;
    Ok(plan)
}

const STRONG_SEMANTIC_PATCH_OPTIMIZER_INSTRUCTIONS: &str = r#"You are the strong optimizer for a typed CPS Program IR experiment.
Return JSON only through return_value, matching the expected schema.
Generate the semantic ProgramPatch plan from input.profile_evidence, input.failure_cluster_evidence, and input.validation_constraints.
Return the final plan as handler_decision.return_value. This final return value is the source artifact for the installed generalized patch.
Set optimizer_source to "strong_model_generated_semantic_patch_plan".
Create one rule for each profile cluster marked fast_path_eligible. Use rule_id exactly "<semantic_cluster>_v1".
Use the provided profile examples as rule examples and contraindications. You may rewrite rationale text, but do not invent clusters or output kinds.
If failure_cluster_evidence contains related_rule_ids, emit negative_guards for those rule ids. Generate Rust-regex-compatible patterns from the failure examples; do not emit code.
Do not output Rust code. Domain-specific words may appear only as data inside the returned plan.
"#;

fn semantic_patch_plan_schema() -> Value {
    let nullable_string = json!({
        "anyOf": [
            { "type": "string" },
            { "type": "null" }
        ]
    });
    json!({
        "type": "object",
        "additionalProperties": false,
        "required": ["patch_id", "rationale", "rules", "negative_guards", "optimizer_source"],
        "properties": {
            "patch_id": { "type": "string", "minLength": 1 },
            "rationale": { "type": "string", "minLength": 1 },
            "rules": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "object",
                    "additionalProperties": false,
                    "required": [
                        "rule_id",
                        "semantic_cluster",
                        "kind",
                        "title",
                        "datetime_hint",
                        "examples",
                        "negative_examples"
                    ],
                    "properties": {
                        "rule_id": { "type": "string", "minLength": 1 },
                        "semantic_cluster": { "type": "string", "minLength": 1 },
                        "kind": { "type": "string", "minLength": 1 },
                        "title": nullable_string.clone(),
                        "datetime_hint": nullable_string.clone(),
                        "examples": {
                            "type": "array",
                            "items": { "type": "string" }
                        },
                        "negative_examples": {
                            "type": "array",
                            "items": { "type": "string" }
                        }
                    }
                }
            },
            "negative_guards": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": false,
                    "required": ["guard_id", "rule_id", "pattern", "rationale"],
                    "properties": {
                        "guard_id": { "type": "string", "minLength": 1 },
                        "rule_id": { "type": "string", "minLength": 1 },
                        "pattern": { "type": "string", "minLength": 1 },
                        "rationale": nullable_string
                    }
                }
            },
            "optimizer_source": {
                "type": "string",
                "enum": [STRONG_MODEL_GENERATED_SEMANTIC_PATCH_PLAN]
            }
        }
    })
}

fn validate_semantic_patch_plan(plan: &SemanticPatchPlan, optimizer_input: &Value) -> Result<()> {
    if plan.patch_id != "semantic_patch_v1" {
        anyhow::bail!(
            "strong semantic patch optimizer returned unexpected patch_id {:?}",
            plan.patch_id
        );
    }
    if plan.optimizer_source != STRONG_MODEL_GENERATED_SEMANTIC_PATCH_PLAN {
        anyhow::bail!(
            "strong semantic patch optimizer returned optimizer_source {:?}, expected {:?}",
            plan.optimizer_source,
            STRONG_MODEL_GENERATED_SEMANTIC_PATCH_PLAN
        );
    }
    if plan.rules.is_empty() {
        anyhow::bail!("strong semantic patch optimizer returned no semantic rules");
    }

    let cluster_evidence = optimizer_input
        .get("profile_evidence")
        .and_then(|value| value.get("clusters"))
        .and_then(Value::as_array)
        .ok_or_else(|| anyhow!("optimizer input missing profile_evidence.clusters"))?;
    let mut eligible_clusters = BTreeMap::new();
    for cluster in cluster_evidence {
        if cluster
            .get("fast_path_eligible")
            .and_then(Value::as_bool)
            .unwrap_or(false)
        {
            let semantic_cluster = cluster
                .get("semantic_cluster")
                .and_then(Value::as_str)
                .ok_or_else(|| anyhow!("eligible profile cluster is missing semantic_cluster"))?;
            let output_kind = cluster
                .get("output_kind")
                .and_then(Value::as_str)
                .ok_or_else(|| anyhow!("eligible profile cluster is missing output_kind"))?;
            eligible_clusters.insert(semantic_cluster.to_owned(), output_kind.to_owned());
        }
    }

    let mut rule_ids = BTreeSet::new();
    let mut emitted_clusters = BTreeSet::new();
    for rule in &plan.rules {
        if !rule_ids.insert(rule.rule_id.as_str()) {
            anyhow::bail!("duplicate semantic rule_id {:?}", rule.rule_id);
        }
        let expected_kind = eligible_clusters
            .get(&rule.semantic_cluster)
            .ok_or_else(|| {
                anyhow!(
                    "semantic rule {:?} references a cluster not eligible in profile evidence",
                    rule.rule_id
                )
            })?;
        let expected_rule_id = format!("{}_v1", rule.semantic_cluster);
        if rule.rule_id != expected_rule_id {
            anyhow::bail!(
                "semantic rule {:?} should use evidence-derived rule_id {:?}",
                rule.rule_id,
                expected_rule_id
            );
        }
        if &rule.kind != expected_kind {
            anyhow::bail!(
                "semantic rule {:?} has kind {:?}, expected {:?}",
                rule.rule_id,
                rule.kind,
                expected_kind
            );
        }
        if rule.examples.is_empty() {
            anyhow::bail!("semantic rule {:?} has no examples", rule.rule_id);
        }
        emitted_clusters.insert(rule.semantic_cluster.as_str());
    }
    for cluster in eligible_clusters.keys() {
        if !emitted_clusters.contains(cluster.as_str()) {
            anyhow::bail!(
                "strong semantic patch optimizer omitted eligible profile cluster {cluster:?}"
            );
        }
    }

    let hard_negative_examples = optimizer_input
        .get("failure_cluster_evidence")
        .and_then(|value| value.get("hard_negative_examples"))
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let required_guard_rule_ids = hard_negative_examples
        .iter()
        .flat_map(|example| {
            example
                .get("related_rule_ids")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
                .filter_map(Value::as_str)
        })
        .collect::<BTreeSet<_>>();
    if !required_guard_rule_ids.is_empty() && plan.negative_guards.is_empty() {
        anyhow::bail!("strong semantic patch optimizer omitted required negative guards");
    }

    let mut guard_ids = BTreeSet::new();
    let mut guarded_rule_ids = BTreeSet::new();
    for guard in &plan.negative_guards {
        if !guard_ids.insert(guard.guard_id.as_str()) {
            anyhow::bail!("duplicate semantic negative guard_id {:?}", guard.guard_id);
        }
        if !rule_ids.contains(guard.rule_id.as_str()) {
            anyhow::bail!(
                "semantic negative guard {:?} references unknown rule_id {:?}",
                guard.guard_id,
                guard.rule_id
            );
        }
        guarded_rule_ids.insert(guard.rule_id.as_str());
        Regex::new(&guard.pattern).with_context(|| {
            format!(
                "semantic negative guard {:?} has invalid regex",
                guard.guard_id
            )
        })?;
    }
    for rule_id in required_guard_rule_ids {
        if !guarded_rule_ids.contains(rule_id) {
            anyhow::bail!(
                "strong semantic patch optimizer omitted negative guard for evidence-related rule_id {rule_id:?}"
            );
        }
    }
    Ok(())
}

async fn run_exact_memo_variant(
    config: &ExperimentConfig,
    experiment_dir: &Path,
    strong: Arc<dyn EffectHandler>,
) -> Result<()> {
    let memo = build_exact_memo_table(config)?;
    let output_schema: Value = read_json(&config.schemas.output_schema)?;
    let task_spec = experiment_task_spec(config)?;
    let events = read_jsonl_values(&config.data.splits_dir.join("heldout_test.events.jsonl"))?;
    let file_name = format!("{}.heldout.jsonl", ExperimentVariant::CpsExactMemo.as_str());
    let prediction_path = experiment_dir.join("predictions").join(&file_name);
    if prediction_path.exists() {
        std::fs::remove_file(&prediction_path)
            .with_context(|| format!("failed to remove {}", prediction_path.display()))?;
    }
    let store = PredictionStore::new(experiment_dir.join("predictions"));
    let run_id = uuid::Uuid::new_v4().to_string();
    let strong = attributed_handler(
        strong,
        run_id.clone(),
        config.workflow_id.clone(),
        ExperimentVariant::CpsExactMemo.as_str(),
    );
    for event in events {
        let event_id = event_id_or_generate(&event);
        let (output, schema_valid, fast_path, model_calls) =
            if let Some(output) = memo.lookup(&event) {
                let output = output_with_event_id(output.clone(), &event_id);
                let schema_valid = validate_value(&output_schema, &output).is_ok();
                (
                    output,
                    schema_valid,
                    FastPathPredictionMetadata {
                        hit: true,
                        rule_id: Some(normalized_text_hash(&event)),
                        kind: Some("exact_memo".to_owned()),
                    },
                    ModelCallCounts::default(),
                )
            } else {
                let (output, schema_valid) = call_direct_model_event(
                    ExperimentVariant::CpsExactMemo.as_str(),
                    &task_spec,
                    event,
                    output_schema.clone(),
                    ModelStrength::Strong,
                    Arc::clone(&strong),
                )
                .await?;
                (
                    output,
                    schema_valid,
                    FastPathPredictionMetadata::miss(),
                    ModelCallCounts {
                        weak: 0,
                        strong_think: 0,
                        strong_task: 1,
                    },
                )
            };
        store.append(
            &file_name,
            &PredictionRecord {
                event_id,
                variant: ExperimentVariant::CpsExactMemo.as_str().to_owned(),
                program_version: "exact_memo_v1".to_owned(),
                output,
                schema_valid,
                trace_run_id: Some(run_id.clone()),
                fast_path,
                model_calls,
            },
        )?;
    }
    Ok(())
}

async fn run_semantic_patch_variant(
    config: &ExperimentConfig,
    experiment_dir: &Path,
    split_name: &str,
    variant: ExperimentVariant,
    semantic_weak_enabled: bool,
    weak: Arc<dyn EffectHandler>,
    strong: Arc<dyn EffectHandler>,
) -> Result<()> {
    let plan = read_semantic_patch_plan_from_artifacts(experiment_dir)?;
    let rules = plan.rules;
    let rules_by_id = rules
        .iter()
        .map(|rule| (rule.rule_id.as_str(), rule))
        .collect::<BTreeMap<_, _>>();
    let output_schema: Value = read_json(&config.schemas.output_schema)?;
    let task_spec = experiment_task_spec(config)?;
    let events = read_jsonl_values(
        &config
            .data
            .splits_dir
            .join(format!("{split_name}.events.jsonl")),
    )?;
    let suffix = if split_name == "heldout_test" {
        "heldout"
    } else {
        "adversarial"
    };
    let file_name = format!("{}.{}.jsonl", variant.as_str(), suffix);
    let prediction_path = experiment_dir.join("predictions").join(&file_name);
    if prediction_path.exists() {
        std::fs::remove_file(&prediction_path)
            .with_context(|| format!("failed to remove {}", prediction_path.display()))?;
    }
    let store = PredictionStore::new(experiment_dir.join("predictions"));
    let run_id = uuid::Uuid::new_v4().to_string();
    let phase = if split_name == "heldout_test" {
        variant.as_str().to_owned()
    } else {
        format!("{}.adversarial", variant.as_str())
    };
    let weak = attributed_handler(
        weak,
        run_id.clone(),
        config.workflow_id.clone(),
        phase.clone(),
    );
    let strong = attributed_handler(strong, run_id.clone(), config.workflow_id.clone(), phase);
    for event in events {
        let event_id = event_id_or_generate(&event);
        let semantic_match = if semantic_weak_enabled {
            semantic_match_event(&event, &rules, Arc::clone(&weak)).await?
        } else {
            None
        };
        let (output, schema_valid, fast_path, model_calls) =
            if let Some(match_output) = semantic_match {
                if match_output.matched && match_output.confidence >= 0.95 {
                    if let Some(rule) = rules_by_id.get(match_output.rule_id.as_str()) {
                        let output = output_from_semantic_rule(rule, &event_id);
                        let schema_valid = validate_value(&output_schema, &output).is_ok();
                        (
                            output,
                            schema_valid,
                            FastPathPredictionMetadata {
                                hit: true,
                                rule_id: Some(rule.rule_id.clone()),
                                kind: Some("weak_semantic_fast_path".to_owned()),
                            },
                            ModelCallCounts {
                                weak: 1,
                                strong_think: 0,
                                strong_task: 0,
                            },
                        )
                    } else {
                        fallback_strong_prediction(
                            variant,
                            &task_spec,
                            event,
                            output_schema.clone(),
                            Arc::clone(&strong),
                            semantic_weak_enabled,
                        )
                        .await?
                    }
                } else {
                    fallback_strong_prediction(
                        variant,
                        &task_spec,
                        event,
                        output_schema.clone(),
                        Arc::clone(&strong),
                        semantic_weak_enabled,
                    )
                    .await?
                }
            } else {
                fallback_strong_prediction(
                    variant,
                    &task_spec,
                    event,
                    output_schema.clone(),
                    Arc::clone(&strong),
                    semantic_weak_enabled,
                )
                .await?
            };
        store.append(
            &file_name,
            &PredictionRecord {
                event_id,
                variant: variant.as_str().to_owned(),
                program_version: if semantic_weak_enabled {
                    "semantic_patch_v1".to_owned()
                } else {
                    "semantic_patch_no_weak_v1".to_owned()
                },
                output,
                schema_valid,
                trace_run_id: Some(run_id.clone()),
                fast_path,
                model_calls,
            },
        )?;
    }
    Ok(())
}

async fn run_installed_patch_variant(
    config: &ExperimentConfig,
    state_dir: &StateDir,
    experiment_dir: &Path,
    split_name: &str,
    variant: ExperimentVariant,
    weak: Arc<dyn EffectHandler>,
    strong: Arc<dyn EffectHandler>,
) -> Result<()> {
    let suffix = if split_name == "heldout_test" {
        "heldout"
    } else {
        "adversarial"
    };
    let file_name = format!("{}.{}.jsonl", variant.as_str(), suffix);
    let prediction_path = experiment_dir.join("predictions").join(&file_name);
    if prediction_path.exists() {
        std::fs::remove_file(&prediction_path)
            .with_context(|| format!("failed to remove {}", prediction_path.display()))?;
    }
    let phase = if split_name == "heldout_test" {
        variant.as_str().to_owned()
    } else {
        format!("{}.adversarial", variant.as_str())
    };
    let weak = attributed_handler(
        weak,
        uuid::Uuid::new_v4().to_string(),
        config.workflow_id.clone(),
        phase.clone(),
    );
    let strong = attributed_handler(
        strong,
        uuid::Uuid::new_v4().to_string(),
        config.workflow_id.clone(),
        phase,
    );
    let source = JsonlEventSource::from_path(
        &config
            .data
            .splits_dir
            .join(format!("{split_name}.events.jsonl")),
    )?;
    let summary = run_stream_with_options(
        state_dir.clone(),
        &config.workflow_id,
        source,
        weak,
        strong,
        false,
        RunStreamOptions {
            predictions: Some(PredictionWriteOptions {
                store: PredictionStore::new(experiment_dir.join("predictions")),
                file_name,
                variant: variant.as_str().to_owned(),
            }),
        },
    )
    .await?;
    if summary.program_version == "v0001" {
        anyhow::bail!(
            "installed patch variant requires a patched program, but latest was {}",
            summary.program_version
        );
    }
    Ok(())
}

async fn run_patch_validation_phase(
    config: &ExperimentConfig,
    state_dir: &StateDir,
    experiment_dir: &Path,
    weak: Arc<dyn EffectHandler>,
    strong: Arc<dyn EffectHandler>,
) -> Result<()> {
    ensure_optimization_artifacts(config, experiment_dir, Arc::clone(&strong)).await?;
    let validation_dir = experiment_dir.join("artifacts").join("patch_validation");
    std::fs::create_dir_all(&validation_dir)
        .with_context(|| format!("failed to create {}", validation_dir.display()))?;
    let plan = read_semantic_patch_plan_from_artifacts(experiment_dir)?;
    let rules = plan.rules.clone();
    let output_schema: Value = read_json(&config.schemas.output_schema)?;
    let programs = FileProgramRegistry::new(state_dir.clone());
    let latest = programs.latest_version(&config.workflow_id)?;
    let base = programs.load_version(&config.workflow_id, &latest)?;
    let patch = semantic_program_patch(&config.workflow_id, &plan, output_schema);
    let patched = validate_patch(&base, &patch).context("patch validation failed")?;
    let patch_dir = experiment_dir.join("artifacts").join("patch_records");
    std::fs::create_dir_all(&patch_dir)
        .with_context(|| format!("failed to create {}", patch_dir.display()))?;
    write_json_pretty(&patch_dir.join("semantic_patch_v1.json"), &patch)?;

    let task_spec = experiment_task_spec(config)?;
    let strong_path = run_direct_split_predictions(
        config,
        experiment_dir,
        "patch_validation",
        "strong_direct",
        ModelStrength::Strong,
        &task_spec,
        Arc::clone(&strong),
    )
    .await?;
    let exact_path = run_exact_memo_split_predictions(
        config,
        experiment_dir,
        "patch_validation",
        Arc::clone(&strong),
    )
    .await?;
    let candidate_path = run_program_split_predictions(
        config,
        experiment_dir,
        "patch_validation",
        "cps_generalized_patch",
        patched,
        Arc::clone(&weak),
        Arc::clone(&strong),
    )
    .await?;
    let quality_dir = validation_dir.join("quality");
    std::fs::create_dir_all(&quality_dir)
        .with_context(|| format!("failed to create {}", quality_dir.display()))?;
    let gold_path = config.data.splits_dir.join("patch_validation.gold.jsonl");
    let strong_quality = evaluate_quality_files(
        &strong_path,
        &gold_path,
        &quality_dir.join("strong_direct.json"),
    )?;
    let exact_quality = evaluate_quality_files(
        &exact_path,
        &gold_path,
        &quality_dir.join("cps_exact_memo.json"),
    )?;
    let candidate_quality = evaluate_quality_files(
        &candidate_path,
        &gold_path,
        &quality_dir.join("cps_generalized_patch.json"),
    )?;
    let gold = read_gold_labels(&gold_path)?;
    let strong_row = row_from_prediction_file(
        "strong_direct",
        &strong_path,
        &gold,
        &strong_quality,
        None,
        0.0,
    )?;
    let exact_row = row_from_prediction_file(
        "cps_exact_memo",
        &exact_path,
        &gold,
        &exact_quality,
        None,
        0.0,
    )?;
    let candidate_row = row_from_prediction_file(
        "cps_generalized_patch",
        &candidate_path,
        &gold,
        &candidate_quality,
        workflow_continuation_frame_p95(state_dir, &config.workflow_id)?,
        0.0,
    )?;
    let gate = evaluate_patch_gate_from_rows(
        config,
        &rules,
        &strong_row,
        &exact_row,
        &candidate_row,
        workflow_continuation_frame_p95(state_dir, &config.workflow_id)?,
    );
    write_json_pretty(&validation_dir.join("patch_gate_decision.json"), &gate)?;
    write_json_pretty(
        &experiment_dir
            .join("artifacts")
            .join("patch_gate_decision.json"),
        &gate,
    )?;
    let patch_report = install_semantic_patch_if_gate_accepted(
        config,
        state_dir,
        experiment_dir,
        &[strong_row, exact_row, candidate_row],
        &gate,
    )?;
    write_json_pretty(&validation_dir.join("patch_install.json"), &patch_report)?;
    if !gate.accepted {
        anyhow::bail!("semantic patch failed patch gate: {:?}", gate.reasons);
    }
    Ok(())
}

async fn run_direct_split_predictions(
    config: &ExperimentConfig,
    experiment_dir: &Path,
    split_name: &str,
    variant: &str,
    strength: ModelStrength,
    task_spec: &str,
    handler: Arc<dyn EffectHandler>,
) -> Result<PathBuf> {
    let output_schema: Value = read_json(&config.schemas.output_schema)?;
    let file_name = format!("{}.{}.jsonl", variant, split_suffix(split_name));
    let prediction_path = experiment_dir.join("predictions").join(&file_name);
    if prediction_path.exists() {
        std::fs::remove_file(&prediction_path)
            .with_context(|| format!("failed to remove {}", prediction_path.display()))?;
    }
    let store = PredictionStore::new(experiment_dir.join("predictions"));
    let run_id = uuid::Uuid::new_v4().to_string();
    let handler = attributed_handler(
        handler,
        run_id.clone(),
        config.workflow_id.clone(),
        format!("{variant}.{}", split_suffix(split_name)),
    );
    for event in read_jsonl_values(
        &config
            .data
            .splits_dir
            .join(format!("{split_name}.events.jsonl")),
    )? {
        let event_id = event_id_or_generate(&event);
        let (output, schema_valid) = call_direct_model_event(
            variant,
            task_spec,
            event,
            output_schema.clone(),
            strength,
            Arc::clone(&handler),
        )
        .await?;
        store.append(
            &file_name,
            &PredictionRecord {
                event_id,
                variant: variant.to_owned(),
                program_version: variant.to_owned(),
                output,
                schema_valid,
                trace_run_id: Some(run_id.clone()),
                fast_path: FastPathPredictionMetadata::miss(),
                model_calls: match strength {
                    ModelStrength::Weak => ModelCallCounts {
                        weak: 1,
                        strong_think: 0,
                        strong_task: 0,
                    },
                    ModelStrength::Strong => ModelCallCounts {
                        weak: 0,
                        strong_think: 0,
                        strong_task: 1,
                    },
                },
            },
        )?;
    }
    Ok(prediction_path)
}

async fn run_exact_memo_split_predictions(
    config: &ExperimentConfig,
    experiment_dir: &Path,
    split_name: &str,
    strong: Arc<dyn EffectHandler>,
) -> Result<PathBuf> {
    let memo = build_exact_memo_table(config)?;
    let output_schema: Value = read_json(&config.schemas.output_schema)?;
    let task_spec = experiment_task_spec(config)?;
    let file_name = format!("cps_exact_memo.{}.jsonl", split_suffix(split_name));
    let prediction_path = experiment_dir.join("predictions").join(&file_name);
    if prediction_path.exists() {
        std::fs::remove_file(&prediction_path)
            .with_context(|| format!("failed to remove {}", prediction_path.display()))?;
    }
    let store = PredictionStore::new(experiment_dir.join("predictions"));
    let run_id = uuid::Uuid::new_v4().to_string();
    let strong = attributed_handler(
        strong,
        run_id.clone(),
        config.workflow_id.clone(),
        format!("cps_exact_memo.{}", split_suffix(split_name)),
    );
    for event in read_jsonl_values(
        &config
            .data
            .splits_dir
            .join(format!("{split_name}.events.jsonl")),
    )? {
        let event_id = event_id_or_generate(&event);
        let (output, schema_valid, fast_path, model_calls) =
            if let Some(output) = memo.lookup(&event) {
                let output = output_with_event_id(output.clone(), &event_id);
                let schema_valid = validate_value(&output_schema, &output).is_ok();
                (
                    output,
                    schema_valid,
                    FastPathPredictionMetadata {
                        hit: true,
                        rule_id: Some(normalized_text_hash(&event)),
                        kind: Some("exact_memo".to_owned()),
                    },
                    ModelCallCounts::default(),
                )
            } else {
                let (output, schema_valid) = call_direct_model_event(
                    "cps_exact_memo",
                    &task_spec,
                    event,
                    output_schema.clone(),
                    ModelStrength::Strong,
                    Arc::clone(&strong),
                )
                .await?;
                (
                    output,
                    schema_valid,
                    FastPathPredictionMetadata::miss(),
                    ModelCallCounts {
                        weak: 0,
                        strong_think: 0,
                        strong_task: 1,
                    },
                )
            };
        store.append(
            &file_name,
            &PredictionRecord {
                event_id,
                variant: "cps_exact_memo".to_owned(),
                program_version: "exact_memo_v1".to_owned(),
                output,
                schema_valid,
                trace_run_id: Some(run_id.clone()),
                fast_path,
                model_calls,
            },
        )?;
    }
    Ok(prediction_path)
}

async fn run_program_split_predictions(
    config: &ExperimentConfig,
    experiment_dir: &Path,
    split_name: &str,
    variant: &str,
    program: Program,
    weak: Arc<dyn EffectHandler>,
    strong: Arc<dyn EffectHandler>,
) -> Result<PathBuf> {
    let file_name = format!("{}.{}.jsonl", variant, split_suffix(split_name));
    let prediction_path = experiment_dir.join("predictions").join(&file_name);
    if prediction_path.exists() {
        std::fs::remove_file(&prediction_path)
            .with_context(|| format!("failed to remove {}", prediction_path.display()))?;
    }
    let store = PredictionStore::new(experiment_dir.join("predictions"));
    let phase = format!("{variant}.{}", split_suffix(split_name));
    let weak = attributed_handler(
        weak,
        uuid::Uuid::new_v4().to_string(),
        config.workflow_id.clone(),
        phase.clone(),
    );
    let strong = attributed_handler(
        strong,
        uuid::Uuid::new_v4().to_string(),
        config.workflow_id.clone(),
        phase,
    );
    for event in read_jsonl_values(
        &config
            .data
            .splits_dir
            .join(format!("{split_name}.events.jsonl")),
    )? {
        let event_id = event_id_or_generate(&event);
        let trace = TraceCollector::default();
        let runtime = Runtime::new(Arc::clone(&weak), Arc::clone(&strong), trace.clone());
        let result = runtime
            .run_program_with_result_and_trace_id(program.clone(), event, event_id.clone())
            .await;
        let trace_events = trace.events();
        let (fast_path, model_calls) = prediction_metadata_from_trace(&trace_events);
        let (output, schema_valid) = match result {
            Ok(result) => {
                let schema_valid = validate_value(&program.output_schema, &result.output).is_ok();
                (result.output, schema_valid)
            }
            Err(err) => (
                json!({
                    "error": err.to_string()
                }),
                false,
            ),
        };
        store.append(
            &file_name,
            &PredictionRecord {
                event_id,
                variant: variant.to_owned(),
                program_version: program.version.clone(),
                output,
                schema_valid,
                trace_run_id: None,
                fast_path,
                model_calls,
            },
        )?;
    }
    Ok(prediction_path)
}

fn split_suffix(split_name: &str) -> &'static str {
    match split_name {
        "patch_validation" => "patch_validation",
        "heldout_test" => "heldout",
        "adversarial_test" => "adversarial",
        _ => "split",
    }
}

fn evaluate_patch_gate_from_rows(
    config: &ExperimentConfig,
    rules: &[SemanticRule],
    strong: &VariantReportRow,
    exact: &VariantReportRow,
    generalized: &VariantReportRow,
    continuation_frame_p95: Option<u64>,
) -> PatchGateDecision {
    let metadata = generalized_semantic_metadata(
        rules
            .iter()
            .map(|rule| rule.semantic_cluster.clone())
            .collect(),
        Vec::new(),
    );
    let gate = config
        .patch_gate
        .as_ref()
        .map(patch_gate_config_from_yaml)
        .unwrap_or_default();
    evaluate_patch_gate(
        &gate,
        &PatchGateInput {
            baseline_quality: strong.quality,
            candidate_quality: generalized.quality,
            baseline_critical_miss_rate: strong.critical_miss,
            candidate_critical_miss_rate: generalized.critical_miss,
            false_fast_path_rate: generalized.false_fast_path_rate.unwrap_or(0.0),
            fast_path_hit_rate_lift: generalized.fast_path_hit_rate - exact.fast_path_hit_rate,
            strong_think_rate_reduction: if strong.strong_calls_per_event == 0.0 {
                0.0
            } else {
                (strong.strong_calls_per_event - generalized.strong_calls_per_event)
                    / strong.strong_calls_per_event
            },
            continuation_frame_p95_bytes: continuation_frame_p95.unwrap_or(0),
            metadata,
        },
    )
}

async fn fallback_strong_prediction(
    variant: ExperimentVariant,
    task_spec: &str,
    event: Value,
    output_schema: Value,
    strong: Arc<dyn EffectHandler>,
    semantic_weak_attempted: bool,
) -> Result<(Value, bool, FastPathPredictionMetadata, ModelCallCounts)> {
    let (output, schema_valid) = call_direct_model_event(
        variant.as_str(),
        task_spec,
        event,
        output_schema,
        ModelStrength::Strong,
        strong,
    )
    .await?;
    Ok((
        output,
        schema_valid,
        FastPathPredictionMetadata::miss(),
        ModelCallCounts {
            weak: u64::from(semantic_weak_attempted),
            strong_think: 0,
            strong_task: 1,
        },
    ))
}

async fn semantic_match_event(
    event: &Value,
    rules: &[SemanticRule],
    weak: Arc<dyn EffectHandler>,
) -> Result<Option<SemanticMatchOutput>> {
    if rules.is_empty() {
        return Ok(None);
    }
    let schema = semantic_match_schema();
    let request = HandlerRequest {
        run_id: None,
        workflow_id: None,
        phase: None,
        effect: EffectCall::ModelTask {
            strength: ModelStrength::Weak,
            task: ModelTaskSpec {
                name: "semantic_fast_path_match".to_owned(),
                instructions: format!(
                    "Match the event to at most one semantic rule. Return matched=true only for a close semantic fit to a rule's positive examples and kind. Treat negative_examples as contraindications. Return matched=false for vague, adversarial, or merely keyword-overlapping events. When matched=true, fill slots.kind, slots.title, and slots.datetime_hint with the action draft to emit; use the rule's kind/title/datetime_hint unless the event clearly supplies a safer slot value. Rules JSON: {}",
                    serde_json::to_string(rules)?
                ),
            },
        },
        input: json!({
            "event": event,
            "rules": rules,
        }),
        expected_schema: schema,
        continuation_summary: None,
        effect_frame: None,
        observations: Vec::new(),
        budget: HandlerBudget {
            effect_depth: 0,
            effects_remaining: RuntimeBudget::default().max_effects,
            handler_reentries_remaining: RuntimeBudget::default().max_handler_reentries,
        },
    };
    match weak.handle(request).await {
        Ok(HandlerDecision::ReturnValue { value, .. }) => {
            let output: SemanticMatchOutput = serde_json::from_value(value)
                .context("weak semantic matcher returned invalid value")?;
            Ok(Some(output))
        }
        Ok(_) | Err(_) => Ok(None),
    }
}

fn semantic_match_schema() -> Value {
    json!({
        "type": "object",
        "additionalProperties": false,
        "required": ["matched", "rule_id", "slots", "confidence", "rationale"],
        "properties": {
            "matched": { "type": "boolean" },
            "rule_id": { "type": "string" },
            "slots": {
                "type": "object",
                "additionalProperties": false,
                "required": ["kind", "title", "datetime_hint"],
                "properties": {
                    "kind": { "type": "string" },
                    "title": { "type": "string" },
                    "datetime_hint": {
                        "anyOf": [
                            { "type": "string" },
                            { "type": "null" }
                        ]
                    }
                }
            },
            "confidence": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 1.0
            },
            "rationale": { "type": "string" }
        }
    })
}

async fn call_direct_model_event(
    task_name: &str,
    task_spec: &str,
    event: Value,
    output_schema: Value,
    strength: ModelStrength,
    handler: Arc<dyn EffectHandler>,
) -> Result<(Value, bool)> {
    let request = HandlerRequest {
        run_id: None,
        workflow_id: None,
        phase: None,
        effect: EffectCall::ModelTask {
            strength,
            task: ModelTaskSpec {
                name: task_name.to_owned(),
                instructions: task_spec.to_owned(),
            },
        },
        input: event,
        expected_schema: output_schema.clone(),
        continuation_summary: None,
        effect_frame: None,
        observations: Vec::new(),
        budget: HandlerBudget {
            effect_depth: 0,
            effects_remaining: RuntimeBudget::default().max_effects,
            handler_reentries_remaining: RuntimeBudget::default().max_handler_reentries,
        },
    };
    match handler.handle(request).await {
        Ok(HandlerDecision::ReturnValue { value, .. }) => {
            let schema_valid = validate_value(&output_schema, &value).is_ok();
            Ok((value, schema_valid))
        }
        Ok(other) => Ok((
            json!({
                "error": format!("handler returned {}", other.decision_name())
            }),
            false,
        )),
        Err(err) => Ok((
            json!({
                "error": err.to_string()
            }),
            false,
        )),
    }
}

fn build_exact_memo_table(config: &ExperimentConfig) -> Result<ExactMemoTable> {
    let events = read_jsonl_values(&config.data.splits_dir.join("profile_train.events.jsonl"))?;
    let gold = read_gold_labels(&config.data.splits_dir.join("profile_train.gold.jsonl"))?;
    let gold_by_id = gold
        .iter()
        .map(|label| (label.event_id.as_str(), label))
        .collect::<BTreeMap<_, _>>();
    let mut entries = BTreeMap::new();
    for event in events {
        let Some(event_id) = event.get("event_id").and_then(Value::as_str) else {
            continue;
        };
        let Some(label) = gold_by_id.get(event_id) else {
            continue;
        };
        let hash = normalized_text_hash(&event);
        entries.insert(
            hash.clone(),
            ExactMemoEntry {
                normalized_text_hash: hash,
                output: output_from_gold_label(label, event_id),
            },
        );
    }
    Ok(ExactMemoTable { entries })
}

fn label_kind_can_fast_path(label: &GoldLabel) -> bool {
    label.kind != "needs_review"
}

fn output_kind_for_label(label: &GoldLabel) -> String {
    if label.kind == "no_action" {
        "ignore".to_owned()
    } else {
        label.kind.clone()
    }
}

fn output_from_semantic_rule(rule: &SemanticRule, event_id: &str) -> Value {
    json!({
        "event_id": event_id,
        "kind": rule.kind,
        "title": rule.title.clone().unwrap_or_else(|| "no action".to_owned()),
        "datetime_hint": rule.datetime_hint,
    })
}

fn output_from_gold_label(label: &GoldLabel, event_id: &str) -> Value {
    json!({
        "event_id": event_id,
        "kind": label.kind,
        "title": label.title_canonical.clone().unwrap_or_else(|| "no action".to_owned()),
        "datetime_hint": label.datetime_canonical,
    })
}

fn output_with_event_id(mut output: Value, event_id: &str) -> Value {
    if let Value::Object(object) = &mut output {
        object.insert("event_id".to_owned(), Value::String(event_id.to_owned()));
    }
    output
}

async fn run_shadow_audit(
    config: &ExperimentConfig,
    experiment_dir: &Path,
    strong: Arc<dyn EffectHandler>,
) -> Result<()> {
    let output_schema: Value = read_json(&config.schemas.output_schema)?;
    let task_spec = experiment_task_spec(config)?;
    let mut source_predictions = Vec::new();
    let mut events_by_id = BTreeMap::new();
    let mut critical_by_id = BTreeMap::new();
    for split in ["heldout_test", "adversarial_test"] {
        for event in
            read_jsonl_values(&config.data.splits_dir.join(format!("{split}.events.jsonl")))?
        {
            if let Some(event_id) = event.get("event_id").and_then(Value::as_str) {
                events_by_id.insert(event_id.to_owned(), event);
            }
        }
        for label in read_gold_labels(&config.data.splits_dir.join(format!("{split}.gold.jsonl")))?
        {
            critical_by_id.insert(
                label.event_id,
                label.criticality == "critical" && label.is_actionable,
            );
        }
    }
    for file_name in [
        "cps_generalized_patch.heldout.jsonl",
        "cps_generalized_patch.adversarial.jsonl",
    ] {
        let path = experiment_dir.join("predictions").join(file_name);
        source_predictions.extend(PredictionStore::read(&path)?);
    }
    let shadow_dir = experiment_dir.join("shadow");
    std::fs::create_dir_all(&shadow_dir)
        .with_context(|| format!("failed to create {}", shadow_dir.display()))?;
    let records_path = shadow_dir.join("shadow_records.jsonl");
    if records_path.exists() {
        std::fs::remove_file(&records_path)
            .with_context(|| format!("failed to remove {}", records_path.display()))?;
    }
    let mut records = Vec::new();
    let strong = attributed_handler(
        strong,
        uuid::Uuid::new_v4().to_string(),
        config.workflow_id.clone(),
        "shadow_audit",
    );
    for prediction in source_predictions
        .iter()
        .filter(|prediction| prediction.fast_path.hit)
    {
        let Some(event) = events_by_id.get(&prediction.event_id).cloned() else {
            continue;
        };
        let (shadow_output, _) = call_direct_model_event(
            "shadow_strong_direct",
            &task_spec,
            event,
            output_schema.clone(),
            ModelStrength::Strong,
            Arc::clone(&strong),
        )
        .await?;
        let critical = critical_by_id
            .get(&prediction.event_id)
            .copied()
            .unwrap_or(false);
        let (_, record) =
            shadow_execution_does_not_change_output(prediction, shadow_output, critical);
        append_jsonl(&records_path, &record)?;
        records.push(record);
    }
    let metrics = summarize_shadow_audit(&source_predictions, &records);
    write_json_pretty(&shadow_dir.join("shadow_metrics.json"), &metrics)?;
    Ok(())
}

fn write_experiment_report(
    config: &ExperimentConfig,
    experiment_dir: &Path,
    state_dir: &StateDir,
) -> Result<()> {
    let budget_store = FileBudgetStore::new(state_dir.clone());
    let budget = budget_store.report(&config.budget.budget_config())?;
    let model_calls = FileModelCallStore::new(state_dir.clone()).list_all()?;
    let api_spend = api_spend_by_variant(&model_calls);
    let continuation_frame_p95 = workflow_continuation_frame_p95(state_dir, &config.workflow_id)?;
    let quality_dir = experiment_dir.join("quality");
    let mut quality = BTreeMap::new();
    let mut variants = Vec::new();
    if quality_dir.exists() {
        for entry in std::fs::read_dir(&quality_dir)? {
            let entry = entry?;
            if !entry.file_type()?.is_file() {
                continue;
            }
            let variant = entry
                .path()
                .file_stem()
                .and_then(|value| value.to_str())
                .unwrap_or("unknown")
                .to_owned();
            let metrics: crate::experiment::quality::QualityMetrics = read_json(&entry.path())?;
            quality.insert(variant.clone(), metrics.clone());
            variants.push(variant_row_from_artifacts(
                experiment_dir,
                &config.data.splits_dir,
                &variant,
                &metrics,
                continuation_frame_p95,
                api_spend.get(&variant).copied().unwrap_or(0.0),
            )?);
        }
    }
    variants.sort_by(|left, right| left.variant.cmp(&right.variant));
    let generalization_breakdown =
        build_generalization_breakdown(experiment_dir, &config.data.splits_dir)?;
    let continuation_frames = workflow_continuation_frame_summary(state_dir, &config.workflow_id)?;
    let patch_gate = read_patch_gate_decision(experiment_dir)?;
    let patch = installed_semantic_patch_report(config, state_dir, experiment_dir)?;
    let pass_fail = build_report_pass_fail(
        &variants,
        &budget,
        patch
            .as_ref()
            .and_then(|patch| patch.get("installed_program_version"))
            .and_then(Value::as_str)
            .is_some_and(|version| version != "v0001"),
        patch_gate.accepted,
        generalized_hits_multiple_clusters(&generalization_breakdown),
    );
    let report = ExperimentReport {
        experiment_id: config.experiment_id.clone(),
        budget,
        variants,
        quality,
        generalization_breakdown,
        patch,
        patch_gate: Some(serde_json::to_value(&patch_gate)?),
        continuation_frames,
        pass_fail,
    };
    write_report(&report, &experiment_dir.join("report.md"))
}

fn patch_gate_config_from_yaml(config: &PatchGateConfigYaml) -> PatchGateConfig {
    PatchGateConfig {
        max_quality_drop: config.max_quality_drop,
        max_critical_miss_delta: config.max_critical_miss_delta,
        max_false_fast_path_rate: config.max_false_fast_path_rate,
        min_fast_path_hit_rate_lift: config.min_fast_path_hit_rate_lift,
        min_strong_think_rate_reduction: config.min_strong_think_rate_reduction,
        max_continuation_frame_p95_bytes: config.max_continuation_frame_p95_bytes,
        require_non_exact_generalization: config.require_non_exact_generalization,
    }
}

fn read_patch_gate_decision(experiment_dir: &Path) -> Result<PatchGateDecision> {
    let path = experiment_dir
        .join("artifacts")
        .join("patch_validation")
        .join("patch_gate_decision.json");
    if path.exists() {
        return read_json(&path);
    }
    Ok(PatchGateDecision {
        accepted: false,
        reasons: vec!["patch_validation phase did not write patch gate decision".to_owned()],
    })
}

fn installed_semantic_patch_report(
    config: &ExperimentConfig,
    state_dir: &StateDir,
    experiment_dir: &Path,
) -> Result<Option<Value>> {
    let path = state_dir
        .workflow_dir(&config.workflow_id)?
        .join("patches")
        .join("installed")
        .join("semantic_patch_v1.json");
    if !path.exists() {
        return Ok(Some(json!({
            "patch_id": "semantic_patch_v1",
            "installed": false,
        })));
    }
    let record: Value = read_json(&path)?;
    let plan = read_semantic_patch_plan_from_artifacts(experiment_dir)?;
    let positive_clusters = plan
        .rules
        .iter()
        .map(|rule| rule.semantic_cluster.clone())
        .collect::<Vec<_>>();
    Ok(Some(json!({
        "patch_id": "semantic_patch_v1",
        "installed": true,
        "installed_program_version": record
            .get("metadata")
            .and_then(|metadata| metadata.get("target_program_version"))
            .and_then(Value::as_str),
        "source": record
            .get("metadata")
            .and_then(|metadata| metadata.get("source"))
            .and_then(Value::as_str),
        "optimizer_source": plan.optimizer_source,
        "raw_optimizer_response_is_final_plan": true,
        "negative_guards": plan.negative_guards.len(),
        "positive_clusters": positive_clusters,
    })))
}

fn install_semantic_patch_if_gate_accepted(
    config: &ExperimentConfig,
    state_dir: &StateDir,
    experiment_dir: &Path,
    variants: &[VariantReportRow],
    decision: &PatchGateDecision,
) -> Result<Option<Value>> {
    if !decision.accepted {
        return Ok(Some(json!({
            "patch_id": "semantic_patch_v1",
            "installed": false,
            "gate_accepted": false,
            "reasons": decision.reasons,
        })));
    }
    let plan = read_semantic_patch_plan_from_artifacts(experiment_dir)?;
    let positive_clusters = plan
        .rules
        .iter()
        .map(|rule| rule.semantic_cluster.clone())
        .collect::<Vec<_>>();
    let output_schema: Value = read_json(&config.schemas.output_schema)?;
    let patch = semantic_program_patch(&config.workflow_id, &plan, output_schema);
    let patch_dir = experiment_dir.join("artifacts").join("patch_records");
    std::fs::create_dir_all(&patch_dir)
        .with_context(|| format!("failed to create {}", patch_dir.display()))?;
    write_json_pretty(&patch_dir.join("semantic_patch_v1.json"), &patch)?;
    let programs = FileProgramRegistry::new(state_dir.clone());
    let patches = FilePatchRegistry::new(state_dir.clone());
    let latest = programs.latest_version(&config.workflow_id)?;
    let versions = programs.list_versions(&config.workflow_id)?;
    if let Some(existing) = versions
        .iter()
        .find(|version| version.patch_id.as_deref() == Some(patch.patch_id.as_str()))
    {
        return Ok(Some(json!({
            "patch_id": patch.patch_id,
            "installed": true,
            "installed_program_version": existing.version,
            "source": "optimizer_strong_model",
            "optimizer_source": plan.optimizer_source,
            "raw_optimizer_response_is_final_plan": true,
            "negative_guards": plan.negative_guards.len(),
            "positive_clusters": positive_clusters,
        })));
    }
    let base = programs.load_version(&config.workflow_id, &latest)?;
    let metadata = PatchMetadata {
        patch_id: patch.patch_id.clone(),
        workflow_id: config.workflow_id.clone(),
        target_program_version: latest.clone(),
        status: PatchStatus::Proposed,
        source: PatchSource::OptimizerStrongModel,
        created_at: now_string(),
        evaluated_at: Some(now_string()),
        installed_at: None,
        rationale: patch.rationale.clone(),
        metrics_delta: Some(patch_metrics_from_variants(variants)),
    };
    patches.record_proposed(&config.workflow_id, patch.clone(), metadata.clone())?;
    let patched = match validate_patch(&base, &patch) {
        Ok(patched) => patched,
        Err(err) => {
            patches.mark_rejected(
                &config.workflow_id,
                patch.clone(),
                metadata,
                &err.to_string(),
            )?;
            return Err(err).context("semantic patch validation failed");
        }
    };
    patches.mark_validated(&config.workflow_id, patch.clone(), metadata.clone())?;
    let installed_version = programs.install_version(
        &config.workflow_id,
        patched,
        ProgramMetadata {
            workflow_id: config.workflow_id.clone(),
            program_id: base.program_id,
            version: String::new(),
            created_at: now_string(),
            source: ProgramSource::PatchInstall,
            parent_version: Some(latest),
            patch_id: Some(patch.patch_id.clone()),
            task_hash: patch.patch_id.clone(),
        },
    )?;
    patches.mark_installed(
        &config.workflow_id,
        patch.clone(),
        metadata,
        &installed_version,
    )?;
    Ok(Some(json!({
        "patch_id": patch.patch_id,
        "installed": true,
        "installed_program_version": installed_version,
        "source": "optimizer_strong_model",
        "optimizer_source": plan.optimizer_source,
        "raw_optimizer_response_is_final_plan": true,
        "negative_guards": plan.negative_guards.len(),
        "positive_clusters": positive_clusters,
    })))
}

fn semantic_patch_validators_value(
    declared_rule_ids: &[String],
    negative_guards: &[SemanticNegativeGuard],
) -> Value {
    let mut validators = vec![json!({
        "validator_id": "semantic_action_slots",
        "predicates": [
            {
                "op": "json_schema_valid",
                "schema": {
                    "type": "object",
                    "additionalProperties": false,
                    "required": [
                        "semantic_match",
                        "event"
                    ],
                    "properties": {
                        "semantic_match": {
                            "type": "object",
                            "additionalProperties": false,
                            "required": [
                                "matched",
                                "rule_id",
                                "slots",
                                "confidence",
                                "rationale"
                            ],
                            "properties": {
                                "matched": { "const": true },
                                "rule_id": {
                                    "type": "string",
                                    "enum": declared_rule_ids
                                },
                                "slots": {
                                    "type": "object",
                                    "additionalProperties": false,
                                    "required": ["kind", "title", "datetime_hint"],
                                    "properties": {
                                        "kind": { "type": "string", "minLength": 1 },
                                        "title": { "type": "string" },
                                        "datetime_hint": {
                                            "anyOf": [
                                                { "type": "string" },
                                                { "type": "null" }
                                            ]
                                        }
                                    }
                                },
                                "confidence": { "type": "number", "minimum": 0.95, "maximum": 1.0 },
                                "rationale": { "type": "string" }
                            }
                        },
                        "event": {
                            "type": "object",
                            "required": ["text"],
                            "properties": {
                                "text": {
                                    "type": "string",
                                    "minLength": 1
                                }
                            },
                            "additionalProperties": true
                        }
                    }
                }
            }
        ]
    })];

    validators.extend(negative_guards.iter().map(|guard| {
        json!({
            "validator_id": format!("semantic_negative_guard_{}", guard.guard_id),
            "predicates": [
                {
                    "op": "not",
                    "term": {
                        "op": "and",
                        "terms": [
                            {
                                "op": "field_equals",
                                "path": ["semantic_match", "rule_id"],
                                "value": guard.rule_id
                            },
                            {
                                "op": "regex_match",
                                "path": ["event", "text"],
                                "pattern": guard.pattern
                            }
                        ]
                    }
                }
            ]
        })
    }));

    Value::Array(validators)
}

fn semantic_program_patch(
    program_id: &str,
    plan: &SemanticPatchPlan,
    output_schema: Value,
) -> ProgramPatch {
    let rules = &plan.rules;
    let positive_clusters = rules
        .iter()
        .map(|rule| rule.semantic_cluster.clone())
        .collect::<Vec<_>>();
    let declared_rule_ids = rules
        .iter()
        .map(|rule| rule.rule_id.clone())
        .collect::<Vec<_>>();
    let rules_value = serde_json::to_value(rules).unwrap_or_else(|_| json!([]));
    let validators_value =
        semantic_patch_validators_value(&declared_rule_ids, &plan.negative_guards);
    ProgramPatch {
        target_program_id: program_id.to_owned(),
        patch_id: plan.patch_id.clone(),
        rationale: plan.rationale.clone(),
        generalization: Some(generalized_semantic_metadata(
            positive_clusters.to_vec(),
            Vec::new(),
        )),
        operations: vec![
            PatchOp::AddEffectPermission {
                permission: EffectPermission::LocalTool {
                    tool_name: "validator_apply".to_owned(),
                },
            },
            PatchOp::AddEffectPermission {
                permission: EffectPermission::LocalTool {
                    tool_name: "template_emit".to_owned(),
                },
            },
            PatchOp::InsertInstruction {
                function: "main".to_owned(),
                pc: 0,
                instr: Instr::Perform {
                    out: "semantic_match".to_owned(),
                    effect: EffectCall::ModelTask {
                        strength: ModelStrength::Weak,
                        task: ModelTaskSpec {
                            name: SEMANTIC_FAST_PATH_TASK.to_owned(),
                        instructions: "Match input.event to at most one object from input.rules. Return matched=true only when the selected rule_id exactly equals a rule_id in input.rules and the event is a close semantic fit to that rule's examples. Treat negative_examples as contraindications. When matched=true, fill slots.kind, slots.title, and slots.datetime_hint from the selected rule unless the event supplies a safer value. Return matched=false for vague, adversarial, unknown-rule, or merely keyword-overlapping events."
                                .to_owned(),
                        },
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
                                    value: rules_value,
                                },
                            ),
                        ]),
                    },
                    expected_schema: semantic_match_schema(),
                    acceptance: AcceptancePolicy {
                        min_confidence: None,
                        require_schema_valid: false,
                        on_failure: FailureHandler::Abort {
                            reason: "semantic matcher failed".to_owned(),
                        },
                    },
                },
            },
            PatchOp::InsertInstruction {
                function: "main".to_owned(),
                pc: 1,
                instr: Instr::Branch {
                    condition: GuardExpr::FieldIsTruthy {
                        var: "semantic_match".to_owned(),
                        path: vec!["matched".to_owned()],
                    },
                    then_pc: 2,
                    else_pc: 6,
                },
            },
            PatchOp::InsertInstruction {
                function: "main".to_owned(),
                pc: 2,
                instr: Instr::Perform {
                    out: "semantic_validation".to_owned(),
                    effect: EffectCall::LocalTool {
                        tool_name: "validator_apply".to_owned(),
                        args_schema: json!({ "type": "object" }),
                    },
                    input: JsonExpr::Object {
                        fields: BTreeMap::from([
                            (
                                "value".to_owned(),
                                JsonExpr::Object {
                                    fields: BTreeMap::from([
                                        (
                                            "semantic_match".to_owned(),
                                            JsonExpr::Var {
                                                name: "semantic_match".to_owned(),
                                            },
                                        ),
                                        (
                                            "event".to_owned(),
                                            JsonExpr::Var {
                                                name: "event".to_owned(),
                                            },
                                        ),
                                    ]),
                                },
                            ),
                            (
                                "validators".to_owned(),
                                JsonExpr::Literal {
                                    value: validators_value,
                                },
                            ),
                        ]),
                    },
                    expected_schema: json!({
                        "type": "object",
                        "additionalProperties": false,
                        "required": ["passed", "failed_validator_ids"],
                        "properties": {
                            "passed": { "type": "boolean" },
                            "failed_validator_ids": {
                                "type": "array",
                                "items": { "type": "string" }
                            }
                        }
                    }),
                    acceptance: AcceptancePolicy {
                        min_confidence: Some(1.0),
                        require_schema_valid: true,
                        on_failure: FailureHandler::Abort {
                            reason: "semantic validator failed".to_owned(),
                        },
                    },
                },
            },
            PatchOp::InsertInstruction {
                function: "main".to_owned(),
                pc: 3,
                instr: Instr::Branch {
                    condition: GuardExpr::FieldIsTruthy {
                        var: "semantic_validation".to_owned(),
                        path: vec!["passed".to_owned()],
                    },
                    then_pc: 4,
                    else_pc: 6,
                },
            },
            PatchOp::InsertInstruction {
                function: "main".to_owned(),
                pc: 4,
                instr: Instr::Perform {
                    out: "fast_action".to_owned(),
                    effect: EffectCall::LocalTool {
                        tool_name: "template_emit".to_owned(),
                        args_schema: json!({ "type": "object" }),
                    },
                    input: JsonExpr::Object {
                        fields: BTreeMap::from([
                            (
                                "input".to_owned(),
                                JsonExpr::Object {
                                    fields: BTreeMap::from([
                                        (
                                            "event".to_owned(),
                                            JsonExpr::Var {
                                                name: "event".to_owned(),
                                            },
                                        ),
                                        (
                                            "semantic_match".to_owned(),
                                            JsonExpr::Var {
                                                name: "semantic_match".to_owned(),
                                            },
                                        ),
                                    ]),
                                },
                            ),
                            (
                                "template".to_owned(),
                                JsonExpr::Literal {
                                    value: json!({
                                        "event_id": { "path": ["event", "event_id"] },
                                        "kind": { "path": ["semantic_match", "slots", "kind"] },
                                        "title": { "path": ["semantic_match", "slots", "title"] },
                                        "datetime_hint": { "path": ["semantic_match", "slots", "datetime_hint"] }
                                    }),
                                },
                            ),
                        ]),
                    },
                    expected_schema: output_schema,
                    acceptance: AcceptancePolicy {
                        min_confidence: Some(1.0),
                        require_schema_valid: true,
                        on_failure: FailureHandler::Abort {
                            reason: "semantic template failed".to_owned(),
                        },
                    },
                },
            },
            PatchOp::InsertInstruction {
                function: "main".to_owned(),
                pc: 5,
                instr: Instr::Return {
                    value: JsonExpr::Var {
                        name: "fast_action".to_owned(),
                    },
                },
            },
        ],
    }
}

fn patch_metrics_from_variants(variants: &[VariantReportRow]) -> PatchEvaluationMetrics {
    let strong = variants
        .iter()
        .find(|row| row.variant == "strong_direct")
        .map(|row| row.strong_calls_per_event)
        .unwrap_or(0.0);
    let generalized = variants
        .iter()
        .find(|row| row.variant == "cps_generalized_patch")
        .cloned()
        .unwrap_or_else(|| VariantReportRow {
            variant: "cps_generalized_patch".to_owned(),
            quality: 0.0,
            critical_miss: 0.0,
            strong_calls_per_event: 0.0,
            weak_calls_per_event: 0.0,
            fast_path_hit_rate: 0.0,
            false_fast_path_rate: Some(0.0),
            shadow_disagreement_rate: None,
            p95_frame_bytes: None,
            api_spend_usd: 0.0,
        });
    PatchEvaluationMetrics {
        base_strong_think_calls: (strong * 160.0).round() as u64,
        patched_strong_think_calls: (generalized.strong_calls_per_event * 160.0).round() as u64,
        base_fast_path_hits: 0,
        patched_fast_path_hits: (generalized.fast_path_hit_rate * 160.0).round() as u64,
    }
}

fn read_semantic_patch_plan_from_artifacts(experiment_dir: &Path) -> Result<SemanticPatchPlan> {
    read_json(
        &experiment_dir
            .join("artifacts")
            .join("semantic_patch_plan.json"),
    )
}

fn build_generalization_breakdown(experiment_dir: &Path, splits_dir: &Path) -> Result<Vec<Value>> {
    let generalized = PredictionStore::read(
        &experiment_dir
            .join("predictions")
            .join("cps_generalized_patch.heldout.jsonl"),
    )?;
    let exact = PredictionStore::read(
        &experiment_dir
            .join("predictions")
            .join("cps_exact_memo.heldout.jsonl"),
    )?;
    let gold = read_gold_labels(&splits_dir.join("heldout_test.gold.jsonl"))?;
    let gold_by_id = gold
        .iter()
        .map(|label| (label.event_id.as_str(), label))
        .collect::<BTreeMap<_, _>>();
    let exact_by_id = exact
        .iter()
        .map(|prediction| (prediction.event_id.as_str(), prediction))
        .collect::<BTreeMap<_, _>>();
    let mut rows: BTreeMap<String, (u64, u64, u64, u64)> = BTreeMap::new();
    for prediction in &generalized {
        let Some(label) = gold_by_id.get(prediction.event_id.as_str()) else {
            continue;
        };
        let row = rows
            .entry(label.semantic_cluster.clone())
            .or_insert((0, 0, 0, 0));
        row.0 += 1;
        if exact_by_id
            .get(prediction.event_id.as_str())
            .is_some_and(|prediction| prediction.fast_path.hit)
        {
            row.1 += 1;
        }
        if prediction.fast_path.hit {
            row.2 += 1;
            if label.hard_negative && prediction_is_actionable(prediction) {
                row.3 += 1;
            }
        }
    }
    Ok(rows
        .into_iter()
        .map(
            |(cluster, (events, exact_hits, generalized_hits, false_hits))| {
                json!({
                    "semantic_cluster": cluster,
                    "events": events,
                    "exact_memo_hits": exact_hits,
                    "generalized_patch_hits": generalized_hits,
                    "false_fast_path_hits": false_hits,
                })
            },
        )
        .collect())
}

fn generalized_hits_multiple_clusters(rows: &[Value]) -> bool {
    rows.iter()
        .filter(|row| {
            row.get("generalized_patch_hits")
                .and_then(Value::as_u64)
                .unwrap_or(0)
                > 0
        })
        .count()
        >= 2
}

fn workflow_continuation_frame_summary(
    state_dir: &StateDir,
    workflow_id: &str,
) -> Result<Option<Value>> {
    let metrics_dir = state_dir.workflow_dir(workflow_id)?.join("metrics");
    if !metrics_dir.exists() {
        return Ok(None);
    }
    let mut frames_total = 0_u64;
    let mut p50 = 0_u64;
    let mut p95 = 0_u64;
    let mut max = 0_u64;
    for entry in std::fs::read_dir(metrics_dir)? {
        let entry = entry?;
        if !entry.file_type()?.is_file() {
            continue;
        }
        let metrics: Value = read_json(&entry.path())?;
        frames_total += metrics
            .get("continuation_frames_total")
            .and_then(Value::as_u64)
            .unwrap_or(0);
        p50 = p50.max(
            metrics
                .get("continuation_frame_bytes_p50")
                .and_then(Value::as_u64)
                .unwrap_or(0),
        );
        p95 = p95.max(
            metrics
                .get("continuation_frame_bytes_p95")
                .and_then(Value::as_u64)
                .unwrap_or(0),
        );
        max = max.max(
            metrics
                .get("continuation_frame_bytes_max")
                .and_then(Value::as_u64)
                .unwrap_or(0),
        );
    }
    Ok(Some(json!({
        "frames_total": frames_total,
        "p50_bytes": p50,
        "p95_bytes": p95,
        "max_bytes": max,
        "limit_bytes": 16384,
        "over_limit": u64::from(p95 > 16384),
    })))
}

fn variant_row_from_artifacts(
    experiment_dir: &Path,
    splits_dir: &Path,
    variant: &str,
    metrics: &crate::experiment::quality::QualityMetrics,
    continuation_frame_p95: Option<u64>,
    api_spend_usd: f64,
) -> Result<VariantReportRow> {
    let (predictions_file, gold_file) = variant
        .strip_suffix(".adversarial")
        .map(|base| {
            (
                format!("{base}.adversarial.jsonl"),
                "adversarial_test.gold.jsonl",
            )
        })
        .unwrap_or_else(|| {
            (
                format!("{variant}.heldout.jsonl"),
                "heldout_test.gold.jsonl",
            )
        });
    let predictions_path = experiment_dir.join("predictions").join(predictions_file);
    let predictions = PredictionStore::read(&predictions_path)?;
    let gold = read_gold_labels(&splits_dir.join(gold_file))?;
    let shadow_disagreement_rate = if variant == "cps_generalized_patch" {
        let metrics_path = experiment_dir.join("shadow").join("shadow_metrics.json");
        if metrics_path.exists() {
            let metrics: crate::experiment::shadow::ShadowMetrics = read_json(&metrics_path)?;
            Some(metrics.shadow_disagreement_rate)
        } else {
            None
        }
    } else {
        None
    };
    let mut row = row_from_predictions(
        variant,
        &predictions,
        &gold,
        metrics,
        if variant.starts_with("cps_") {
            continuation_frame_p95
        } else {
            None
        },
        api_spend_usd,
    );
    row.shadow_disagreement_rate = shadow_disagreement_rate;
    Ok(row)
}

fn row_from_prediction_file(
    variant: &str,
    predictions_path: &Path,
    gold: &[GoldLabel],
    metrics: &QualityMetrics,
    continuation_frame_p95: Option<u64>,
    api_spend_usd: f64,
) -> Result<VariantReportRow> {
    let predictions = PredictionStore::read(predictions_path)?;
    Ok(row_from_predictions(
        variant,
        &predictions,
        gold,
        metrics,
        continuation_frame_p95,
        api_spend_usd,
    ))
}

fn row_from_predictions(
    variant: &str,
    predictions: &[PredictionRecord],
    gold: &[GoldLabel],
    metrics: &QualityMetrics,
    continuation_frame_p95: Option<u64>,
    api_spend_usd: f64,
) -> VariantReportRow {
    let events_total = predictions.len().max(1) as f64;
    let strong_calls = predictions
        .iter()
        .map(|prediction| prediction.model_calls.strong_task + prediction.model_calls.strong_think)
        .sum::<u64>() as f64;
    let weak_calls = predictions
        .iter()
        .map(|prediction| prediction.model_calls.weak)
        .sum::<u64>() as f64;
    let fast_hits = predictions
        .iter()
        .filter(|prediction| prediction.fast_path.hit)
        .count() as f64;
    VariantReportRow {
        variant: variant.to_owned(),
        quality: metrics.intent_accuracy,
        critical_miss: metrics.critical_miss_rate,
        strong_calls_per_event: strong_calls / events_total,
        weak_calls_per_event: weak_calls / events_total,
        fast_path_hit_rate: fast_hits / events_total,
        false_fast_path_rate: Some(false_fast_path_rate(predictions, gold)),
        shadow_disagreement_rate: None,
        p95_frame_bytes: continuation_frame_p95,
        api_spend_usd,
    }
}

fn api_spend_by_variant(records: &[ModelCallRecord]) -> BTreeMap<String, f64> {
    let mut by_variant = BTreeMap::new();
    for record in records {
        let Some(variant) = variant_for_model_call(record) else {
            continue;
        };
        *by_variant.entry(variant.to_owned()).or_insert(0.0) += record.cost.total_usd;
    }
    by_variant
}

fn variant_for_model_call(record: &ModelCallRecord) -> Option<&str> {
    if let Some(phase) = record.phase.as_deref() {
        if phase.ends_with(".patch_validation") {
            return None;
        }
        return match phase {
            "weak_only" => Some("weak_only"),
            "strong_direct" => Some("strong_direct"),
            "cps_unoptimized_profile" | "cps_unoptimized_heldout" => Some("cps_unoptimized"),
            "cps_exact_memo" => Some("cps_exact_memo"),
            "cps_generalized_patch" => Some("cps_generalized_patch"),
            "cps_generalized_patch.adversarial" => Some("cps_generalized_patch.adversarial"),
            "cps_generalized_patch_no_semantic_weak" => {
                Some("cps_generalized_patch_no_semantic_weak")
            }
            "shadow_audit" => None,
            _ => None,
        };
    }
    match record.task_name.as_deref()? {
        "weak_only" => Some("weak_only"),
        "strong_direct" => Some("strong_direct"),
        "draft_action_from_event" => Some("cps_unoptimized"),
        "cps_exact_memo" => Some("cps_exact_memo"),
        "semantic_fast_path_match" | "cps_generalized_patch" => Some("cps_generalized_patch"),
        "cps_generalized_patch_no_semantic_weak" => Some("cps_generalized_patch_no_semantic_weak"),
        _ => None,
    }
}

fn workflow_continuation_frame_p95(state_dir: &StateDir, workflow_id: &str) -> Result<Option<u64>> {
    let metrics_dir = state_dir.workflow_dir(workflow_id)?.join("metrics");
    if !metrics_dir.exists() {
        return Ok(None);
    }
    let mut p95 = None;
    for entry in std::fs::read_dir(metrics_dir)? {
        let entry = entry?;
        if !entry.file_type()?.is_file() {
            continue;
        }
        let metrics: Value = read_json(&entry.path())?;
        if let Some(value) = metrics
            .get("continuation_frame_bytes_p95")
            .and_then(Value::as_u64)
        {
            p95 = Some(p95.unwrap_or(0).max(value));
        }
    }
    Ok(p95)
}

fn false_fast_path_rate(predictions: &[PredictionRecord], gold: &[GoldLabel]) -> f64 {
    let gold_by_id = gold
        .iter()
        .map(|label| (label.event_id.as_str(), label))
        .collect::<BTreeMap<_, _>>();
    let false_hits = predictions
        .iter()
        .filter(|prediction| prediction.fast_path.hit)
        .filter(|prediction| {
            let Some(label) = gold_by_id.get(prediction.event_id.as_str()) else {
                return false;
            };
            label.hard_negative && prediction_is_actionable(prediction)
        })
        .count() as f64;
    false_hits / predictions.len().max(1) as f64
}

fn prediction_is_actionable(prediction: &PredictionRecord) -> bool {
    let kind = prediction
        .output
        .get("kind")
        .and_then(Value::as_str)
        .unwrap_or("none");
    kind != "none" && kind != "no_action" && kind != "ignore"
}

fn build_report_pass_fail(
    variants: &[VariantReportRow],
    budget: &BudgetReport,
    program_version_advanced: bool,
    patch_gate_accepted: bool,
    multi_cluster_improvement: bool,
) -> ExperimentPassFail {
    let strong = variants.iter().find(|row| row.variant == "strong_direct");
    let exact = variants.iter().find(|row| row.variant == "cps_exact_memo");
    let generalized = variants
        .iter()
        .find(|row| row.variant == "cps_generalized_patch");
    let adversarial = variants
        .iter()
        .find(|row| row.variant == "cps_generalized_patch.adversarial");
    match (strong, exact, generalized) {
        (Some(strong), Some(exact), Some(generalized)) => {
            let mut pass_fail = build_pass_fail(strong, exact, generalized, budget);
            pass_fail.adversarial_false_fast_path_under_2_percent = adversarial
                .and_then(|row| row.false_fast_path_rate)
                .is_some_and(|rate| rate <= 0.02);
            pass_fail.program_version_advanced = program_version_advanced;
            pass_fail.multi_cluster_improvement = multi_cluster_improvement;
            pass_fail.patch_gate_accepted = patch_gate_accepted;
            pass_fail.experiment_passed = pass_fail.quality_noninferior
                && pass_fail.critical_miss_not_worse
                && pass_fail.strong_calls_reduced_by_50_percent
                && pass_fail.generalized_patch_beats_exact_memo
                && pass_fail.false_fast_path_under_1_percent
                && pass_fail.adversarial_false_fast_path_under_2_percent
                && pass_fail.continuation_frame_p95_under_limit
                && pass_fail.program_version_advanced
                && pass_fail.multi_cluster_improvement
                && pass_fail.patch_gate_accepted
                && pass_fail.within_100_usd_budget;
            pass_fail
        }
        _ => ExperimentPassFail {
            quality_noninferior: false,
            critical_miss_not_worse: false,
            strong_calls_reduced_by_50_percent: false,
            generalized_patch_beats_exact_memo: false,
            false_fast_path_under_1_percent: false,
            adversarial_false_fast_path_under_2_percent: false,
            continuation_frame_p95_under_limit: false,
            program_version_advanced: false,
            multi_cluster_improvement: false,
            patch_gate_accepted: false,
            within_100_usd_budget: budget.spent_usd <= budget.hard_cap_usd,
            experiment_passed: false,
        },
    }
}

fn append_jsonl<T: Serialize>(path: &Path, value: &T) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }
    let mut raw = serde_json::to_vec(value)?;
    raw.push(b'\n');
    use std::io::Write;
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .with_context(|| format!("failed to open {}", path.display()))?;
    file.write_all(&raw)
        .with_context(|| format!("failed to append {}", path.display()))
}

fn experiment_task_spec(config: &ExperimentConfig) -> Result<String> {
    if let Some(workflow) = &config.workflow {
        return std::fs::read_to_string(&workflow.task)
            .with_context(|| format!("failed to read task file {}", workflow.task.display()));
    }
    Ok("Return one action draft for the event.".to_owned())
}

fn evaluate_present_predictions(experiment_dir: &Path, splits_dir: &Path) -> Result<()> {
    let predictions_dir = experiment_dir.join("predictions");
    if !predictions_dir.exists() {
        return Ok(());
    }
    let quality_dir = experiment_dir.join("quality");
    for entry in std::fs::read_dir(&predictions_dir)? {
        let entry = entry?;
        if !entry.file_type()?.is_file() {
            continue;
        }
        let file_name = entry.file_name().to_string_lossy().to_string();
        let (variant, gold_file) = if file_name.ends_with(".heldout.jsonl") {
            (
                file_name.trim_end_matches(".heldout.jsonl").to_owned(),
                "heldout_test.gold.jsonl",
            )
        } else if file_name.ends_with(".adversarial.jsonl") {
            (
                format!(
                    "{}.adversarial",
                    file_name.trim_end_matches(".adversarial.jsonl")
                ),
                "adversarial_test.gold.jsonl",
            )
        } else {
            continue;
        };
        evaluate_quality_files(
            &entry.path(),
            &splits_dir.join(gold_file),
            &quality_dir.join(format!("{variant}.json")),
        )?;
    }
    Ok(())
}

pub fn dry_run_estimate(
    config: &ExperimentConfig,
    catalog: &PriceCatalog,
) -> Result<DryRunCostEstimate> {
    let events_total = count_jsonl(&config.data.all_events).unwrap_or(400).max(1) as u64;
    let weak_calls = events_total * 3;
    let strong_calls = events_total + events_total / 2;
    let optimizer_calls = 4;
    let weak_usage = crate::store::model_call_store::ModelUsage {
        input_tokens: 1_000 * weak_calls,
        cached_input_tokens: 0,
        output_tokens: 200 * weak_calls,
        reasoning_tokens: 0,
        total_tokens: 1_200 * weak_calls,
    };
    let strong_usage = crate::store::model_call_store::ModelUsage {
        input_tokens: 2_500 * strong_calls + 25_000 * optimizer_calls,
        cached_input_tokens: 0,
        output_tokens: 300 * strong_calls + 4_000 * optimizer_calls,
        reasoning_tokens: 0,
        total_tokens: 2_800 * strong_calls + 29_000 * optimizer_calls,
    };
    let weak_cost = catalog
        .estimate_cost(&config.models.weak_model, &weak_usage, true)?
        .total_usd;
    let strong_cost = catalog
        .estimate_cost(&config.models.strong_model, &strong_usage, true)?
        .total_usd;
    let estimated_total_usd = weak_cost + strong_cost;
    let fits_budget = estimated_total_usd <= config.budget.hard_cap_usd;
    Ok(DryRunCostEstimate {
        estimated_total_usd,
        hard_cap_usd: config.budget.hard_cap_usd,
        fits_budget,
        largest_phase: "shadow_audit".to_owned(),
        recommendation: if fits_budget {
            "proceed".to_owned()
        } else {
            "reduce scale or shadow sample rate".to_owned()
        },
    })
}

pub fn experiment_dir(state_dir: &StateDir, experiment_id: &str) -> PathBuf {
    state_dir.root().join("experiments").join(experiment_id)
}

fn write_phase_marker(experiment_dir: &Path, phase: &str, status: &str) -> Result<()> {
    let phase_dir = experiment_dir.join("artifacts").join("phases");
    std::fs::create_dir_all(&phase_dir)
        .with_context(|| format!("failed to create {}", phase_dir.display()))?;
    std::fs::write(
        phase_dir.join(format!("{phase}.json")),
        serde_json::to_vec_pretty(&json!({
            "phase": phase,
            "status": status,
            "completed_at": now_string(),
        }))?,
    )
    .with_context(|| format!("failed to write phase marker {phase}"))
}

fn write_skipped_phase_marker(experiment_dir: &Path, phase: &str, reason: &str) -> Result<()> {
    let phase_dir = experiment_dir.join("artifacts").join("phases");
    std::fs::create_dir_all(&phase_dir)
        .with_context(|| format!("failed to create {}", phase_dir.display()))?;
    std::fs::write(
        phase_dir.join(format!("{phase}.json")),
        serde_json::to_vec_pretty(&json!({
            "phase": phase,
            "status": "skipped",
            "reason": reason,
            "updated_at": now_string(),
        }))?,
    )
    .with_context(|| format!("failed to write skipped phase marker {phase}"))
}

fn write_minimal_report(
    experiment_dir: &Path,
    config: &ExperimentConfig,
    budget: &BudgetReport,
) -> Result<()> {
    let report = json!({
        "experiment_id": config.experiment_id,
        "workflow_id": config.workflow_id,
        "budget": budget,
        "status": "phases_completed",
    });
    write_json_pretty(&experiment_dir.join("report.json"), &report)?;
    std::fs::write(
        experiment_dir.join("report.md"),
        format!(
            "# {}\n\nCompleted phases: {}\n\nActual API spend: ${:.4}\n",
            config.experiment_id,
            config.phases.join(", "),
            budget.spent_usd
        ),
    )
    .with_context(|| "failed to write experiment markdown report")
}

fn count_jsonl(path: &Path) -> Result<usize> {
    let raw = std::fs::read_to_string(path)
        .with_context(|| format!("failed to read {}", path.display()))?;
    Ok(raw.lines().filter(|line| !line.trim().is_empty()).count())
}

fn read_jsonl_values(path: &Path) -> Result<Vec<Value>> {
    let raw = std::fs::read_to_string(path)
        .with_context(|| format!("failed to read {}", path.display()))?;
    raw.lines()
        .enumerate()
        .filter(|(_, line)| !line.trim().is_empty())
        .map(|(index, line)| {
            serde_json::from_str(line).with_context(|| {
                format!("invalid JSONL record at {}:{}", path.display(), index + 1)
            })
        })
        .collect()
}

fn event_id_or_generate(event: &Value) -> String {
    event
        .get("event_id")
        .and_then(Value::as_str)
        .map(ToOwned::to_owned)
        .unwrap_or_else(|| uuid::Uuid::new_v4().to_string())
}
