use std::path::PathBuf;

use anyhow::{Context, Result, anyhow};
use clap::{Parser, Subcommand};
use serde_json::{Value, json};

use crate::config::{DEFAULT_BASE_URL, DEFAULT_STRONG_MODEL, DEFAULT_WEAK_MODEL, ModelConfig};
use crate::effects::{HandlerBudget, HandlerDecision, HandlerRequest, RuntimeBudget};
use crate::models::{EffectHandler, ResponsesStrongModel, ResponsesWeakModel};
use crate::program::{EffectCall, ModelStrength, ModelTaskSpec, Program};
use crate::runtime::{RunContext, Runtime};
use crate::schema::{action_drafts_schema, message_events_schema, program_schema, schema_bundle};
use crate::trace::{TraceCollector, parse_trace_jsonl, replay_trace_events};
use crate::validator::validate_program;
use crate::value_demo::baseline::run_strong_direct_baseline;
use crate::value_demo::continuation_compaction::{ContinuationCompactionConfig, ValueStore};
use crate::value_demo::event_source::{EventEnvelope, EventSource, JsonlEventSource};
use crate::value_demo::local_tools::LocalToolRegistry;
use crate::value_demo::metrics::{EventRunResult, MetricsCollector, RunMetrics, RunMode};
use crate::value_demo::patch_evaluator::PatchEvaluator;
use crate::value_demo::patch_registry::{PatchEnvelope, PatchRegistry, PatchStatus};
use crate::value_demo::program_registry::ProgramRegistry;
use crate::value_demo::report::build_metrics_report;
use crate::value_demo::state_dir::StateDir;

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
    ValueDemo {
        #[arg(long)]
        program_id: Option<String>,

        #[arg(long)]
        program: Option<PathBuf>,

        #[arg(long)]
        events: PathBuf,

        #[arg(long, default_value = ".local-cps-demo")]
        state_dir: PathBuf,

        #[arg(long)]
        metrics_out: Option<PathBuf>,

        #[arg(long)]
        install_accepted_patches: bool,

        #[arg(long)]
        eval_events: Option<PathBuf>,

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
    BaselineStrongDirect {
        #[arg(long)]
        task: PathBuf,

        #[arg(long)]
        events: PathBuf,

        #[arg(long)]
        metrics_out: Option<PathBuf>,

        #[arg(long, default_value = ".local-cps-demo")]
        state_dir: PathBuf,

        #[arg(long, env = "OPENAI_BASE_URL", default_value = DEFAULT_BASE_URL)]
        base_url: String,

        #[arg(long, env = "OPENAI_API_KEY")]
        api_key: Option<String>,

        #[arg(long, env = "CPS_STRONG_MODEL", default_value = DEFAULT_STRONG_MODEL)]
        strong_model: String,

        #[arg(long)]
        trace_json: bool,
    },
    MetricsReport {
        #[arg(long)]
        baseline: PathBuf,

        #[arg(long)]
        round1: PathBuf,

        #[arg(long)]
        round2: PathBuf,
    },
    ListPrograms {
        #[arg(long, default_value = ".local-cps-demo")]
        state_dir: PathBuf,
    },
    ListPatches {
        #[arg(long, default_value = ".local-cps-demo")]
        state_dir: PathBuf,

        #[arg(long)]
        program_id: Option<String>,
    },
    ValidateProgram {
        #[arg(long)]
        program: PathBuf,
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
            let runtime = Runtime::with_local_tools(
                weak,
                strong,
                trace.clone(),
                LocalToolRegistry::with_default_tools(),
            );
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
            let runtime = Runtime::with_local_tools(
                weak,
                strong,
                trace.clone(),
                LocalToolRegistry::with_default_tools(),
            );
            let output = runtime.run_program(program, input).await?;

            println!("{}", serde_json::to_string_pretty(&output)?);
            emit_trace(trace, trace_json)?;
        }
        Command::ValueDemo {
            program_id,
            program,
            events,
            state_dir,
            metrics_out,
            install_accepted_patches,
            eval_events,
            base_url,
            api_key,
            weak_model,
            strong_model,
            trace_json,
        } => {
            let state_dir = StateDir::open_or_create(state_dir)?;
            let program_registry = ProgramRegistry::new(state_dir.clone());
            let patch_registry = PatchRegistry::new(state_dir.clone());
            let run_id = format!("run-{}", uuid::Uuid::new_v4());
            let mut active_program = match (program, program_id) {
                (Some(program_path), _) => {
                    let program = read_program(&program_path)?;
                    validate_program(&program).context("program validation failed")?;
                    program_registry.install_initial(&program, &run_id)?;
                    program
                }
                (None, Some(program_id)) => program_registry.load_latest(&program_id)?,
                (None, None) => {
                    return Err(anyhow!("value-demo requires --program or --program-id"));
                }
            };

            let events = read_events_jsonl(&events)?;
            let config = ModelConfig::new(base_url, api_key, weak_model, strong_model)?;
            let client = config.responses_client();
            let weak = ResponsesWeakModel::new(client.clone(), config.weak_model);
            let strong = ResponsesStrongModel::new(client, config.strong_model);
            let local_tools = LocalToolRegistry::with_default_tools();
            let trace = TraceCollector::default();
            let runtime = Runtime::with_local_tools(
                weak.clone(),
                strong.clone(),
                trace.clone(),
                local_tools.clone(),
            );
            let value_store = ValueStore::new(run_id.clone(), state_dir.value_store_dir(&run_id))?;
            let mut collector = MetricsCollector::new(run_id.clone(), RunMode::ValueDemo);
            collector.metrics_mut().program_id = Some(active_program.program_id.clone());
            collector.metrics_mut().program_version = Some(active_program.version.clone());
            let mut outputs = Vec::new();
            let mut proposed_patches = Vec::new();

            for event in &events {
                let report = runtime
                    .run_program_with_report(
                        active_program.clone(),
                        event.as_program_input(),
                        Some(RunContext {
                            run_id: run_id.clone(),
                            event_id: Some(event.event_id.clone()),
                            value_store: Some(value_store.clone()),
                            continuation_compaction: Some(ContinuationCompactionConfig::default()),
                        }),
                    )
                    .await;
                match report {
                    Ok(report) => {
                        outputs.push(report.output);
                        proposed_patches.extend(report.pending_patches);
                        collector.observe_event_result(&EventRunResult { succeeded: true });
                    }
                    Err(err) => {
                        outputs.push(json!({
                            "event_id": event.event_id,
                            "error": err.to_string(),
                        }));
                        collector.observe_event_result(&EventRunResult { succeeded: false });
                    }
                }
            }

            let mut installed_count = 0;
            let mut rejected_count = 0;
            proposed_patches.sort_by(|left, right| left.patch_id.cmp(&right.patch_id));
            proposed_patches.dedup_by(|left, right| left.patch_id == right.patch_id);

            for patch in &proposed_patches {
                patch_registry.save_pending(&PatchEnvelope {
                    patch: patch.clone(),
                    status: PatchStatus::Pending,
                    proposed_by_run_id: run_id.clone(),
                    proposed_for_program_version: active_program.version.clone(),
                    failure_fingerprint: None,
                    evaluation: None,
                })?;
            }

            if install_accepted_patches {
                let eval_events = match eval_events {
                    Some(path) => read_events_jsonl(&path)?,
                    None => events.clone(),
                };
                let evaluator = PatchEvaluator::default();
                for patch in proposed_patches {
                    let outcome = evaluator
                        .evaluate(
                            &active_program,
                            &patch,
                            &eval_events,
                            weak.clone(),
                            strong.clone(),
                            local_tools.clone(),
                        )
                        .await?;
                    if outcome.report.accepted {
                        patch_registry.mark_validated(&patch.patch_id, outcome.report.clone())?;
                        let patched_program = outcome
                            .patched_program
                            .expect("accepted evaluation includes patched program");
                        program_registry.install_patched(
                            &patched_program,
                            &patch.patch_id,
                            &run_id,
                        )?;
                        patch_registry.mark_installed(&patch.patch_id, &patched_program.version)?;
                        active_program = patched_program;
                        installed_count += 1;
                    } else {
                        patch_registry.mark_rejected(&patch.patch_id, &outcome.report.reason)?;
                        rejected_count += 1;
                    }
                }
            }

            collector.metrics_mut().patches_installed += installed_count;
            collector.metrics_mut().patches_rejected += rejected_count;
            for event in trace.events() {
                collector.observe_trace_event(&event);
            }
            let metrics = collector.finish();
            write_metrics(&state_dir, metrics_out.as_ref(), &metrics)?;
            write_trace(&state_dir, &run_id, &trace)?;

            println!("{}", serde_json::to_string_pretty(&outputs)?);
            emit_trace(trace, trace_json)?;
        }
        Command::BaselineStrongDirect {
            task,
            events,
            metrics_out,
            state_dir,
            base_url,
            api_key,
            strong_model,
            trace_json,
        } => {
            let state_dir = StateDir::open_or_create(state_dir)?;
            let run_id = format!("baseline-{}", uuid::Uuid::new_v4());
            let task_spec = read_task(&task)?;
            let events = read_events_jsonl(&events)?;
            let config = ModelConfig::new(
                base_url,
                api_key,
                DEFAULT_WEAK_MODEL.to_owned(),
                strong_model,
            )?;
            let client = config.responses_client();
            let strong = ResponsesStrongModel::new(client, config.strong_model);
            let trace = TraceCollector::default();
            let metrics =
                run_strong_direct_baseline(run_id.clone(), &task_spec, &events, &strong, &trace)
                    .await?;
            write_metrics(&state_dir, metrics_out.as_ref(), &metrics)?;
            write_trace(&state_dir, &run_id, &trace)?;
            println!("{}", serde_json::to_string_pretty(&metrics)?);
            emit_trace(trace, trace_json)?;
        }
        Command::MetricsReport {
            baseline,
            round1,
            round2,
        } => {
            let baseline = read_metrics(&baseline)?;
            let round1 = read_metrics(&round1)?;
            let round2 = read_metrics(&round2)?;
            println!(
                "{}",
                serde_json::to_string_pretty(&build_metrics_report(&baseline, &round1, &round2))?
            );
        }
        Command::ListPrograms { state_dir } => {
            let state_dir = StateDir::open_or_create(state_dir)?;
            let registry = ProgramRegistry::new(state_dir);
            println!(
                "{}",
                serde_json::to_string_pretty(&registry.list_programs()?)?
            );
        }
        Command::ListPatches {
            state_dir,
            program_id,
        } => {
            let state_dir = StateDir::open_or_create(state_dir)?;
            let registry = PatchRegistry::new(state_dir);
            let mut patches = registry.list(None)?;
            if let Some(program_id) = program_id {
                patches.retain(|patch| patch.patch.target_program_id == program_id);
            }
            println!("{}", serde_json::to_string_pretty(&patches)?);
        }
        Command::ValidateProgram { program } => {
            let program = read_program(&program)?;
            validate_program(&program).context("program validation failed")?;
            println!("{}", json!({ "ok": true, "status": "OK" }));
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
                name: "classify_and_extract_action_draft".to_owned(),
                instructions: "Given one message event, return one action draft.".to_owned(),
            };
            let input = json!({
                "event_id": "probe-calendar",
                "text": "Friday 3pm product review meeting"
            });
            let request = HandlerRequest {
                effect: EffectCall::ModelTask {
                    strength: ModelStrength::Weak,
                    task,
                },
                input,
                expected_schema: crate::schema::action_draft_schema(),
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
                effect: EffectCall::Think {
                    reason: "probe".to_owned(),
                },
                input: serde_json::to_value(&weak_decision)?,
                expected_schema: crate::schema::action_draft_schema(),
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

fn read_json(path: &PathBuf) -> Result<Value> {
    let raw = std::fs::read_to_string(path)
        .with_context(|| format!("failed to read input file {}", path.display()))?;
    serde_json::from_str(&raw).with_context(|| format!("invalid JSON in {}", path.display()))
}

fn read_events_jsonl(path: &PathBuf) -> Result<Vec<EventEnvelope>> {
    let mut source = JsonlEventSource::from_path(path)?;
    let mut events = Vec::new();
    while let Some(event) = source.next_event()? {
        events.push(event);
    }
    Ok(events)
}

fn read_metrics(path: &PathBuf) -> Result<RunMetrics> {
    let raw = std::fs::read(path)
        .with_context(|| format!("failed to read metrics file {}", path.display()))?;
    serde_json::from_slice(&raw).with_context(|| format!("invalid metrics JSON {}", path.display()))
}

fn write_metrics(
    state_dir: &StateDir,
    metrics_out: Option<&PathBuf>,
    metrics: &RunMetrics,
) -> Result<()> {
    let state_metrics_path = state_dir
        .metrics_dir()
        .join(format!("{}.metrics.json", metrics.run_id));
    write_pretty_json(&state_metrics_path, metrics)?;
    if let Some(metrics_out) = metrics_out {
        write_pretty_json(metrics_out, metrics)?;
    }
    Ok(())
}

fn write_trace(state_dir: &StateDir, run_id: &str, trace: &TraceCollector) -> Result<()> {
    let path = state_dir.traces_dir().join(format!("{run_id}.trace.jsonl"));
    let mut raw = String::new();
    for event in trace.events() {
        raw.push_str(&serde_json::to_string(&event)?);
        raw.push('\n');
    }
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(path, raw)?;
    Ok(())
}

fn write_pretty_json(path: &PathBuf, value: &impl serde::Serialize) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(path, serde_json::to_vec_pretty(value)?)?;
    Ok(())
}

fn emit_trace(trace: TraceCollector, trace_json: bool) -> Result<()> {
    if trace_json {
        for event in trace.events() {
            eprintln!("{}", serde_json::to_string(&event)?);
        }
    }
    Ok(())
}
