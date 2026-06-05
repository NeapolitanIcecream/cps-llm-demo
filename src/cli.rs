use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{Context, Result, anyhow};
use clap::{Parser, Subcommand};
use serde_json::{Value, json};

use crate::config::{DEFAULT_BASE_URL, DEFAULT_STRONG_MODEL, DEFAULT_WEAK_MODEL, ModelConfig};
use crate::effects::{HandlerBudget, HandlerDecision, HandlerRequest, RuntimeBudget};
use crate::engine::event_source::JsonlEventSource;
use crate::engine::run_coordinator::run_stream;
use crate::evaluation::compare_runs::compare_runs;
use crate::evaluation::strong_direct_baseline::baseline_strong_direct;
use crate::experiment::exact_memo::write_exact_memo_patch;
use crate::experiment::quality::evaluate_quality_files;
use crate::experiment::runner::{experiment_dir, run_experiment_from_file};
use crate::experiment::split::{SplitCounts, SplitStrategy, split_events_files};
use crate::model_cache::{ModelCache, ModelCacheMode};
use crate::models::{EffectHandler, FixtureModelHandler, ResponsesStrongModel, ResponsesWeakModel};
use crate::observability::report::build_metrics_report;
use crate::optimizer::patch_installer::install_fixture_patch;
use crate::optimizer::patch_optimizer::{OptimizerContext, optimize_from_profile};
use crate::pricing::price_catalog::PriceCatalog;
use crate::program::{EffectCall, ModelStrength, ModelTaskSpec, Program, ProgramPatch};
use crate::responses_client::{ModelCallRuntime, ResponsesClient, ResponsesClientConfig};
use crate::runtime::Runtime;
use crate::schema::{action_drafts_schema, message_events_schema, program_schema, schema_bundle};
use crate::store::budget_store::FileBudgetStore;
use crate::store::metrics_store::{FileMetricsStore, RunMetrics};
use crate::store::model_call_store::FileModelCallStore;
use crate::store::program_registry::{
    FileProgramRegistry, ProgramMetadata, ProgramSource, fixture_program_metadata,
};
use crate::store::state_dir::{StateDir, now_string, stable_hash_bytes};
use crate::trace::{TraceCollector, parse_trace_jsonl, replay_trace_events};
use crate::validator::validate_program;

#[derive(Debug, Parser)]
#[command(version, about = "CPS-style LLM typed-effect runtime demo")]
pub struct Cli {
    #[command(subcommand)]
    pub command: Command,
}

#[derive(Debug, Subcommand)]
pub enum Command {
    CompileRun {
        #[arg(long)]
        task: PathBuf,

        #[arg(long)]
        input: PathBuf,

        #[arg(long, env = "OPENAI_BASE_URL", default_value = DEFAULT_BASE_URL)]
        base_url: String,

        #[arg(long, env = "OPENAI_API_KEY")]
        api_key: Option<String>,

        #[arg(long, env = "CPS_WEAK_MODEL", default_value = DEFAULT_WEAK_MODEL)]
        weak_model: String,

        #[arg(long, env = "CPS_STRONG_MODEL", default_value = DEFAULT_STRONG_MODEL)]
        strong_model: String,

        #[arg(long)]
        trace_json: bool,
    },
    RunProgram {
        #[arg(long)]
        program: PathBuf,

        #[arg(long)]
        input: PathBuf,

        #[arg(long, env = "OPENAI_BASE_URL", default_value = DEFAULT_BASE_URL)]
        base_url: String,

        #[arg(long, env = "OPENAI_API_KEY")]
        api_key: Option<String>,

        #[arg(long, env = "CPS_WEAK_MODEL", default_value = DEFAULT_WEAK_MODEL)]
        weak_model: String,

        #[arg(long, env = "CPS_STRONG_MODEL", default_value = DEFAULT_STRONG_MODEL)]
        strong_model: String,

        #[arg(long)]
        trace_json: bool,
    },
    ValidateProgram {
        #[arg(long)]
        program: PathBuf,
    },
    InitWorkflow {
        #[arg(long)]
        workflow: String,

        #[arg(long)]
        task: Option<PathBuf>,

        #[arg(long)]
        program: Option<PathBuf>,

        #[arg(long, default_value = ".cps-llm-demo")]
        state_dir: PathBuf,

        #[arg(long, env = "OPENAI_BASE_URL", default_value = DEFAULT_BASE_URL)]
        base_url: String,

        #[arg(long, env = "OPENAI_API_KEY")]
        api_key: Option<String>,

        #[arg(long, env = "CPS_WEAK_MODEL", default_value = DEFAULT_WEAK_MODEL)]
        weak_model: String,

        #[arg(long, env = "CPS_STRONG_MODEL", default_value = DEFAULT_STRONG_MODEL)]
        strong_model: String,
    },
    RunStream {
        #[arg(long)]
        workflow: String,

        #[arg(long)]
        events: PathBuf,

        #[arg(long, default_value = ".cps-llm-demo")]
        state_dir: PathBuf,

        #[arg(long, env = "OPENAI_BASE_URL", default_value = DEFAULT_BASE_URL)]
        base_url: String,

        #[arg(long, env = "OPENAI_API_KEY")]
        api_key: Option<String>,

        #[arg(long, env = "CPS_WEAK_MODEL", default_value = DEFAULT_WEAK_MODEL)]
        weak_model: String,

        #[arg(long, env = "CPS_STRONG_MODEL", default_value = DEFAULT_STRONG_MODEL)]
        strong_model: String,

        #[arg(long)]
        trace_json: bool,
    },
    Optimize {
        #[arg(long)]
        workflow: String,

        #[arg(long, default_value = ".cps-llm-demo")]
        state_dir: PathBuf,

        #[arg(long)]
        patch: Option<PathBuf>,

        #[arg(long, default_value_t = 3)]
        max_patches: usize,

        #[arg(long, env = "OPENAI_BASE_URL", default_value = DEFAULT_BASE_URL)]
        base_url: String,

        #[arg(long, env = "OPENAI_API_KEY")]
        api_key: Option<String>,

        #[arg(long, env = "CPS_WEAK_MODEL", default_value = DEFAULT_WEAK_MODEL)]
        weak_model: String,

        #[arg(long, env = "CPS_STRONG_MODEL", default_value = DEFAULT_STRONG_MODEL)]
        strong_model: String,
    },
    BaselineStrongDirect {
        #[arg(long)]
        workflow: String,

        #[arg(long)]
        task: PathBuf,

        #[arg(long)]
        events: PathBuf,

        #[arg(long, default_value = ".cps-llm-demo")]
        state_dir: PathBuf,

        #[arg(long, env = "OPENAI_BASE_URL", default_value = DEFAULT_BASE_URL)]
        base_url: String,

        #[arg(long, env = "OPENAI_API_KEY")]
        api_key: Option<String>,

        #[arg(long, env = "CPS_STRONG_MODEL", default_value = DEFAULT_STRONG_MODEL)]
        strong_model: String,
    },
    MetricsReport {
        #[arg(long)]
        workflow: String,

        #[arg(long, default_value = ".cps-llm-demo")]
        state_dir: PathBuf,
    },
    CompareRuns {
        #[arg(long)]
        baseline_run: String,

        #[arg(long)]
        before_run: String,

        #[arg(long)]
        after_run: String,

        #[arg(long, default_value = ".cps-llm-demo")]
        state_dir: PathBuf,
    },
    BudgetReport {
        #[arg(long, default_value = ".cps-llm-demo")]
        state_dir: PathBuf,
    },
    BudgetReset {
        #[arg(long, default_value = ".cps-llm-demo")]
        state_dir: PathBuf,

        #[arg(long)]
        confirm: bool,
    },
    SplitEvents {
        #[arg(long)]
        events: PathBuf,

        #[arg(long)]
        gold: PathBuf,

        #[arg(long)]
        out: PathBuf,

        #[arg(long, default_value = "time-cluster")]
        strategy: String,

        #[arg(long)]
        profile_train: usize,

        #[arg(long)]
        patch_validation: usize,

        #[arg(long)]
        heldout_test: usize,

        #[arg(long)]
        adversarial_test: usize,
    },
    EvaluateQuality {
        #[arg(long)]
        predictions: PathBuf,

        #[arg(long)]
        gold: PathBuf,

        #[arg(long)]
        out: PathBuf,
    },
    RunExperiment {
        #[arg(long)]
        config: PathBuf,

        #[arg(long, default_value = ".cps-real-exp")]
        state_dir: PathBuf,

        #[arg(long)]
        dry_run_cost: bool,
    },
    ExperimentReport {
        #[arg(long)]
        experiment: String,

        #[arg(long, default_value = ".cps-real-exp")]
        state_dir: PathBuf,

        #[arg(long)]
        out: PathBuf,
    },
    BuildExactMemoPatch {
        #[arg(long)]
        workflow: String,

        #[arg(long)]
        from_events: PathBuf,

        #[arg(long, default_value = ".cps-real-exp")]
        state_dir: PathBuf,
    },
    Replay {
        #[arg(long)]
        trace: PathBuf,
    },
    Schema,
    ProbeModels {
        #[arg(long, env = "OPENAI_BASE_URL", default_value = DEFAULT_BASE_URL)]
        base_url: String,

        #[arg(long, env = "OPENAI_API_KEY")]
        api_key: Option<String>,

        #[arg(long, env = "CPS_WEAK_MODEL", default_value = DEFAULT_WEAK_MODEL)]
        weak_model: String,

        #[arg(long, env = "CPS_STRONG_MODEL", default_value = DEFAULT_STRONG_MODEL)]
        strong_model: String,
    },
}

pub async fn run() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Command::CompileRun {
            task,
            input,
            base_url,
            api_key,
            weak_model,
            strong_model,
            trace_json,
        } => {
            let config = ModelConfig::new(base_url, api_key, weak_model, strong_model)?;
            let task_spec = read_task(&task)?;
            let input = read_json(&input)?;
            let trace = TraceCollector::default();
            let client = config.responses_client();
            let weak = ResponsesWeakModel::new(client.clone(), config.weak_model);
            let strong = ResponsesStrongModel::new(client, config.strong_model);

            let input_schema = message_events_schema();
            let output_schema = action_drafts_schema();
            let program = compile_program(&strong, &task_spec, input_schema, output_schema).await?;
            let runtime = Runtime::new(weak, strong, trace.clone());
            let output = runtime.run_program(program, input).await?;

            println!("{}", serde_json::to_string_pretty(&output)?);
            emit_trace(trace, trace_json)?;
        }
        Command::RunProgram {
            program,
            input,
            base_url,
            api_key,
            weak_model,
            strong_model,
            trace_json,
        } => {
            let config = ModelConfig::new(base_url, api_key, weak_model, strong_model)?;
            let program = read_program(&program)?;
            validate_program(&program).context("program validation failed")?;
            let input = read_json(&input)?;
            let trace = TraceCollector::default();
            let client = config.responses_client();
            let weak = ResponsesWeakModel::new(client.clone(), config.weak_model);
            let strong = ResponsesStrongModel::new(client, config.strong_model);
            let runtime = Runtime::new(weak, strong, trace.clone());
            let output = runtime.run_program(program, input).await?;

            println!("{}", serde_json::to_string_pretty(&output)?);
            emit_trace(trace, trace_json)?;
        }
        Command::ValidateProgram { program } => {
            let program = read_program(&program)?;
            validate_program(&program).context("program validation failed")?;
            println!("{}", json!({ "ok": true, "status": "OK" }));
        }
        Command::InitWorkflow {
            workflow,
            task,
            program,
            state_dir,
            base_url,
            api_key,
            weak_model,
            strong_model,
        } => {
            let state = StateDir::new(state_dir);
            let registry = FileProgramRegistry::new(state.clone());
            let (program, metadata) = match (task, program) {
                (Some(task), None) => {
                    let config = ModelConfig::new(base_url, api_key, weak_model, strong_model)?;
                    let task_spec = read_task(&task)?;
                    let client = responses_client_for_state(&config, &state);
                    let strong = ResponsesStrongModel::new(client, config.strong_model);
                    let program =
                        compile_program(&strong, &task_spec, json!({}), json!({})).await?;
                    let metadata = ProgramMetadata {
                        workflow_id: workflow.clone(),
                        program_id: program.program_id.clone(),
                        version: program.version.clone(),
                        created_at: now_string(),
                        source: ProgramSource::StrongCompile,
                        parent_version: None,
                        patch_id: None,
                        task_hash: stable_hash_bytes(task_spec.as_bytes()),
                    };
                    (program, metadata)
                }
                (None, Some(program_path)) => {
                    let program = read_program(&program_path)?;
                    validate_program(&program).context("program validation failed")?;
                    let metadata = fixture_program_metadata(&workflow, &program);
                    (program, metadata)
                }
                (Some(_), Some(_)) => {
                    return Err(anyhow!(
                        "init-workflow accepts either --task or --program, not both"
                    ));
                }
                (None, None) => {
                    return Err(anyhow!("init-workflow requires --task or --program"));
                }
            };
            let compiled_by_strong = matches!(metadata.source, ProgramSource::StrongCompile);
            let compiled_program_id = program.program_id.clone();
            registry.init_workflow(&workflow, program, metadata)?;
            if compiled_by_strong {
                let mut metrics = RunMetrics::new(
                    uuid::Uuid::new_v4().to_string(),
                    workflow.clone(),
                    "compile".to_owned(),
                    compiled_program_id,
                    "v0001".to_owned(),
                    now_string(),
                );
                metrics.program_compile_calls = 1;
                metrics.estimated_model_calls = 1;
                metrics.finished_at = now_string();
                FileMetricsStore::new(state).write(&metrics)?;
            }
            println!(
                "{}",
                serde_json::to_string_pretty(&json!({
                    "ok": true,
                    "workflow_id": workflow,
                    "latest_program_version": "v0001"
                }))?
            );
        }
        Command::RunStream {
            workflow,
            events,
            state_dir,
            base_url,
            api_key,
            weak_model,
            strong_model,
            trace_json,
        } => {
            let state = StateDir::new(state_dir);
            let event_source = JsonlEventSource::from_path(&events)?;
            let (weak, strong) =
                handler_pair_for_state(base_url, api_key, weak_model, strong_model, &state)?;
            let summary =
                run_stream(state, &workflow, event_source, weak, strong, trace_json).await?;
            println!("{}", serde_json::to_string_pretty(&summary)?);
        }
        Command::Optimize {
            workflow,
            state_dir,
            patch,
            max_patches,
            base_url,
            api_key,
            weak_model,
            strong_model,
        } => {
            let state = StateDir::new(state_dir);
            let programs = FileProgramRegistry::new(state.clone());
            let patches = crate::store::patch_registry::FilePatchRegistry::new(state.clone());
            let traces = crate::store::trace_store::FileTraceStore::new(state.clone());
            let profiles = crate::store::profile_store::FileProfileStore::new(state.clone());
            let (weak, strong) =
                handler_pair_for_state(base_url, api_key, weak_model, strong_model, &state)?;
            let installed_version = if let Some(patch_path) = patch {
                let patch = read_patch(&patch_path)?;
                install_fixture_patch(&programs, &patches, &traces, &workflow, patch, weak, strong)
                    .await?
            } else {
                optimize_from_profile(
                    OptimizerContext {
                        programs: &programs,
                        patches: &patches,
                        traces: &traces,
                        profiles: &profiles,
                        weak,
                        strong,
                    },
                    &workflow,
                    max_patches,
                )
                .await?
            };
            println!(
                "{}",
                serde_json::to_string_pretty(&json!({
                    "ok": true,
                    "workflow_id": workflow,
                    "installed_program_version": installed_version
                }))?
            );
        }
        Command::BaselineStrongDirect {
            workflow,
            task,
            events,
            state_dir,
            base_url,
            api_key,
            strong_model,
        } => {
            let state = StateDir::new(state_dir);
            let event_source = JsonlEventSource::from_path(&events)?;
            let task_spec = read_task(&task)?;
            let strong = strong_handler_for_state(base_url, api_key, strong_model, &state)?;
            let summary =
                baseline_strong_direct(state, &workflow, task_spec, event_source, strong).await?;
            println!("{}", serde_json::to_string_pretty(&summary)?);
        }
        Command::MetricsReport {
            workflow,
            state_dir,
        } => {
            let state = StateDir::new(state_dir);
            let metrics = FileMetricsStore::new(state.clone()).list(&workflow)?;
            let programs = FileProgramRegistry::new(state);
            let latest = programs.latest_version(&workflow).ok();
            let versions = programs.list_versions(&workflow).unwrap_or_default();
            let report = build_metrics_report(&workflow, latest, metrics, &versions);
            println!("{}", serde_json::to_string_pretty(&report)?);
        }
        Command::CompareRuns {
            baseline_run,
            before_run,
            after_run,
            state_dir,
        } => {
            let metrics = FileMetricsStore::new(StateDir::new(state_dir));
            let baseline = metrics.find_run(&baseline_run)?;
            let before = metrics.find_run(&before_run)?;
            let after = metrics.find_run(&after_run)?;
            let comparison = compare_runs(&baseline, &before, &after, 16_384);
            println!("{}", serde_json::to_string_pretty(&comparison)?);
        }
        Command::BudgetReport { state_dir } => {
            let store = FileBudgetStore::new(StateDir::new(state_dir));
            let config = store.read_config()?;
            let report = store.report(&config)?;
            println!("{}", serde_json::to_string_pretty(&report)?);
        }
        Command::BudgetReset { state_dir, confirm } => {
            if !confirm {
                return Err(anyhow!("budget-reset requires --confirm"));
            }
            let store = FileBudgetStore::new(StateDir::new(state_dir));
            store.reset()?;
            println!("{}", serde_json::to_string_pretty(&json!({ "ok": true }))?);
        }
        Command::SplitEvents {
            events,
            gold,
            out,
            strategy,
            profile_train,
            patch_validation,
            heldout_test,
            adversarial_test,
        } => {
            if strategy != "time-cluster" {
                return Err(anyhow!("unsupported split strategy {strategy}"));
            }
            let result = split_events_files(
                &events,
                &gold,
                &out,
                SplitStrategy::TimeCluster,
                SplitCounts {
                    profile_train,
                    patch_validation,
                    heldout_test,
                    adversarial_test,
                },
            )?;
            println!("{}", serde_json::to_string_pretty(&result.leakage_report)?);
        }
        Command::EvaluateQuality {
            predictions,
            gold,
            out,
        } => {
            let metrics = evaluate_quality_files(&predictions, &gold, &out)?;
            println!("{}", serde_json::to_string_pretty(&metrics)?);
        }
        Command::RunExperiment {
            config,
            state_dir,
            dry_run_cost,
        } => {
            let output =
                run_experiment_from_file(&config, StateDir::new(state_dir), dry_run_cost).await?;
            println!("{}", serde_json::to_string_pretty(&output)?);
        }
        Command::ExperimentReport {
            experiment,
            state_dir,
            out,
        } => {
            let dir = experiment_dir(&StateDir::new(state_dir), &experiment);
            let report_md = dir.join("report.md");
            let report_json = dir.join("report.json");
            if let Some(parent) = out.parent() {
                std::fs::create_dir_all(parent)
                    .with_context(|| format!("failed to create {}", parent.display()))?;
            }
            std::fs::copy(&report_md, &out)
                .with_context(|| format!("failed to copy {}", report_md.display()))?;
            if report_json.exists() {
                std::fs::copy(&report_json, out.with_extension("json"))
                    .with_context(|| format!("failed to copy {}", report_json.display()))?;
            }
            println!(
                "{}",
                serde_json::to_string_pretty(&json!({ "ok": true, "out": out }))?
            );
        }
        Command::BuildExactMemoPatch {
            workflow,
            from_events,
            state_dir,
        } => {
            let state = StateDir::new(state_dir);
            let program_id = FileProgramRegistry::new(state.clone())
                .load_latest(&workflow)
                .map(|program| program.program_id)
                .unwrap_or_else(|_| workflow.clone());
            let _events = JsonlEventSource::from_path(&from_events)?;
            let out = state
                .root()
                .join("experiments")
                .join("artifacts")
                .join(format!("{workflow}.exact_memo.patch.json"));
            let patch = write_exact_memo_patch(&out, &program_id)?;
            println!(
                "{}",
                serde_json::to_string_pretty(&json!({
                    "ok": true,
                    "patch_id": patch.patch_id,
                    "path": out,
                }))?
            );
        }
        Command::Replay { trace } => {
            let raw = std::fs::read_to_string(&trace)
                .with_context(|| format!("failed to read trace file {}", trace.display()))?;
            let events = parse_trace_jsonl(&raw)?;
            let report = replay_trace_events(&events)?;
            println!(
                "{}",
                json!({
                    "ok": true,
                    "captures": report.captures,
                    "resumes": report.resumes,
                    "aborts": report.aborts,
                })
            );
        }
        Command::Schema => {
            println!("{}", serde_json::to_string_pretty(&schema_bundle())?);
        }
        Command::ProbeModels {
            base_url,
            api_key,
            weak_model,
            strong_model,
        } => {
            let config = ModelConfig::new(base_url, api_key, weak_model, strong_model)?;
            let client = config.responses_client();
            let weak = ResponsesWeakModel::new(client.clone(), config.weak_model);
            let strong = ResponsesStrongModel::new(client, config.strong_model);

            let task = ModelTaskSpec {
                name: "probe_json_transform".to_owned(),
                instructions: "Given one JSON object, return one JSON object.".to_owned(),
            };
            let input = json!({
                "event_id": "probe-1",
                "text": "sample input"
            });
            let request = HandlerRequest {
                run_id: None,
                workflow_id: None,
                phase: Some("probe_models".to_owned()),
                effect: EffectCall::ModelTask {
                    strength: ModelStrength::Weak,
                    task,
                },
                input,
                expected_schema: json!({ "type": "object" }),
                continuation_summary: None,
                effect_frame: None,
                observations: Vec::new(),
                budget: HandlerBudget {
                    effect_depth: 0,
                    effects_remaining: RuntimeBudget::default().max_effects,
                    handler_reentries_remaining: RuntimeBudget::default().max_handler_reentries,
                },
            };
            let weak_decision = weak.handle(request).await?;
            let strong_request = HandlerRequest {
                run_id: None,
                workflow_id: None,
                phase: Some("probe_models".to_owned()),
                effect: EffectCall::Think {
                    reason: "probe".to_owned(),
                },
                input: serde_json::to_value(&weak_decision)?,
                expected_schema: json!({ "type": "object" }),
                continuation_summary: None,
                effect_frame: None,
                observations: Vec::new(),
                budget: HandlerBudget {
                    effect_depth: 0,
                    effects_remaining: RuntimeBudget::default().max_effects,
                    handler_reentries_remaining: RuntimeBudget::default().max_handler_reentries,
                },
            };
            let strong_decision = strong.handle(strong_request).await?;
            let ok = matches!(
                weak_decision,
                HandlerDecision::ReturnValue { .. }
                    | HandlerDecision::RequestEffect { .. }
                    | HandlerDecision::ReturnProgramFragment { .. }
                    | HandlerDecision::Abort { .. }
                    | HandlerDecision::ReturnProgram { .. }
                    | HandlerDecision::ReturnProgramPatch { .. }
            ) && matches!(
                strong_decision,
                HandlerDecision::ReturnValue { .. }
                    | HandlerDecision::RequestEffect { .. }
                    | HandlerDecision::ReturnProgramPatch { .. }
                    | HandlerDecision::Abort { .. }
                    | HandlerDecision::ReturnProgram { .. }
                    | HandlerDecision::ReturnProgramFragment { .. }
            );

            println!("{}", json!({ "ok": ok, "status": "OK" }));
        }
    }

    Ok(())
}

pub async fn compile_program<H>(
    strong: &H,
    task_spec: &str,
    input_schema: Value,
    output_schema: Value,
) -> Result<Program>
where
    H: EffectHandler,
{
    let request = HandlerRequest {
        run_id: None,
        workflow_id: None,
        phase: None,
        effect: EffectCall::CompileProgram {
            strength: ModelStrength::Strong,
            task_spec: task_spec.to_owned(),
            input_schema: input_schema.clone(),
            output_schema: output_schema.clone(),
        },
        input: json!({}),
        expected_schema: program_schema(),
        continuation_summary: None,
        effect_frame: None,
        observations: Vec::new(),
        budget: HandlerBudget {
            effect_depth: 0,
            effects_remaining: RuntimeBudget::default().max_effects,
            handler_reentries_remaining: RuntimeBudget::default().max_handler_reentries,
        },
    };

    match strong.handle(request).await? {
        HandlerDecision::ReturnProgram { mut program, .. } => {
            program.input_schema = input_schema;
            program.output_schema = output_schema;
            validate_program(&program).context("compiled program validation failed")?;
            Ok(program)
        }
        other => Err(anyhow!(
            "strong compiler returned {}, expected return_program",
            other.decision_name()
        )),
    }
}

fn read_task(path: &PathBuf) -> Result<String> {
    std::fs::read_to_string(path)
        .with_context(|| format!("failed to read task file {}", path.display()))
}

fn read_program(path: &PathBuf) -> Result<Program> {
    let raw = std::fs::read_to_string(path)
        .with_context(|| format!("failed to read program file {}", path.display()))?;
    serde_json::from_str(&raw)
        .with_context(|| format!("invalid Program JSON in {}", path.display()))
}

fn read_patch(path: &PathBuf) -> Result<ProgramPatch> {
    let raw = std::fs::read_to_string(path)
        .with_context(|| format!("failed to read patch file {}", path.display()))?;
    serde_json::from_str(&raw)
        .with_context(|| format!("invalid ProgramPatch JSON in {}", path.display()))
}

fn read_json(path: &PathBuf) -> Result<Value> {
    let raw = std::fs::read_to_string(path)
        .with_context(|| format!("failed to read input file {}", path.display()))?;
    serde_json::from_str(&raw).with_context(|| format!("invalid JSON in {}", path.display()))
}

fn emit_trace(trace: TraceCollector, trace_json: bool) -> Result<()> {
    if trace_json {
        for event in trace.events() {
            eprintln!("{}", serde_json::to_string(&event)?);
        }
    }
    Ok(())
}

fn handler_pair_for_state(
    base_url: String,
    api_key: Option<String>,
    weak_model: String,
    strong_model: String,
    state: &StateDir,
) -> Result<(Arc<dyn EffectHandler>, Arc<dyn EffectHandler>)> {
    if api_key
        .as_ref()
        .is_some_and(|value| !value.trim().is_empty())
    {
        let config = ModelConfig::new(base_url, api_key, weak_model, strong_model)?;
        let client = responses_client_for_state(&config, state);
        Ok((
            Arc::new(ResponsesWeakModel::new(client.clone(), config.weak_model)),
            Arc::new(ResponsesStrongModel::new(client, config.strong_model)),
        ))
    } else {
        Ok((
            Arc::new(FixtureModelHandler::weak()),
            Arc::new(FixtureModelHandler::strong()),
        ))
    }
}

fn strong_handler_for_state(
    base_url: String,
    api_key: Option<String>,
    strong_model: String,
    state: &StateDir,
) -> Result<Arc<dyn EffectHandler>> {
    if api_key
        .as_ref()
        .is_some_and(|value| !value.trim().is_empty())
    {
        let config = ModelConfig::new(
            base_url,
            api_key,
            DEFAULT_WEAK_MODEL.to_owned(),
            strong_model,
        )?;
        let client = responses_client_for_state(&config, state);
        Ok(Arc::new(ResponsesStrongModel::new(
            client,
            config.strong_model,
        )))
    } else {
        Ok(Arc::new(FixtureModelHandler::strong()))
    }
}

fn responses_client_for_state(config: &ModelConfig, state: &StateDir) -> ResponsesClient {
    let call_store = FileModelCallStore::new(state.clone());
    let budget_store = FileBudgetStore::new(state.clone());
    let budget_config = budget_store.read_config().unwrap_or_default();
    let catalog = PriceCatalog::default_openai();
    let runtime = ModelCallRuntime::new(call_store, catalog)
        .with_budget(budget_store, budget_config)
        .with_cache(ModelCache::new(
            state.root().join("model_cache"),
            ModelCacheMode::ReadWrite,
        ));
    ResponsesClient::new(ResponsesClientConfig {
        base_url: config.base_url.clone(),
        api_key: config.api_key.clone(),
        runtime: Some(Arc::new(runtime)),
    })
}
