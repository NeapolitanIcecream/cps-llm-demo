use std::path::PathBuf;

use anyhow::{Context, Result, anyhow};
use clap::{Parser, Subcommand};
use serde_json::{Value, json};

use crate::config::{DEFAULT_BASE_URL, DEFAULT_STRONG_MODEL, DEFAULT_WEAK_MODEL, ModelConfig};
use crate::effects::{HandlerBudget, HandlerDecision, HandlerRequest, RuntimeBudget};
use crate::models::{EffectHandler, ResponsesStrongModel, ResponsesWeakModel};
use crate::program::{EffectCall, ModelStrength, ModelTaskSpec, Program};
use crate::runtime::Runtime;
use crate::schema::{action_drafts_schema, message_events_schema, program_schema, schema_bundle};
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

fn emit_trace(trace: TraceCollector, trace_json: bool) -> Result<()> {
    if trace_json {
        for event in trace.events() {
            eprintln!("{}", serde_json::to_string(&event)?);
        }
    }
    Ok(())
}
