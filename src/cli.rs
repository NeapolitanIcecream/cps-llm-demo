use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use serde_json::{Value, json};

use crate::config::{DEFAULT_BASE_URL, DEFAULT_STRONG_MODEL, DEFAULT_WEAK_MODEL, ModelConfig};
use crate::effects::{Continuation, EffectFrame, ThinkDecision};
use crate::models::{ResponsesStrongModel, ResponsesWeakModel, StrongModel, WeakModel};
use crate::program::{Program, WeakTaskSpec};
use crate::runtime::Runtime;
use crate::schema::{action_draft_schema, message_event_schema, schema_bundle};
use crate::trace::TraceCollector;

#[derive(Debug, Parser)]
#[command(version, about = "CPS-style LLM program runtime demo")]
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
            let inputs = read_inputs(&input)?;
            let trace = TraceCollector::default();
            let client = config.responses_client();
            let weak = ResponsesWeakModel::new(client.clone(), config.weak_model);
            let strong = ResponsesStrongModel::new(client, config.strong_model);

            let input_schema = message_event_schema();
            let output_schema = action_draft_schema();
            let program = strong
                .compile_program(&task_spec, &input_schema, &output_schema)
                .await?;
            let program = constrain_compiled_program_contract(program, input_schema, output_schema);
            let runtime = Runtime::new(weak, strong, trace.clone());
            let outputs = run_inputs(&runtime, program, inputs).await?;

            println!("{}", serde_json::to_string_pretty(&outputs)?);
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
            let inputs = read_inputs(&input)?;
            let trace = TraceCollector::default();
            let client = config.responses_client();
            let weak = ResponsesWeakModel::new(client.clone(), config.weak_model);
            let strong = ResponsesStrongModel::new(client, config.strong_model);
            let runtime = Runtime::new(weak, strong, trace.clone());
            let outputs = run_inputs(&runtime, program, inputs).await?;

            println!("{}", serde_json::to_string_pretty(&outputs)?);
            emit_trace(trace, trace_json)?;
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

            let task = WeakTaskSpec {
                name: "classify_and_extract_action_draft".to_owned(),
                instructions: "Given one message event, return one action draft.".to_owned(),
            };
            let input = json!({
                "event_id": "probe-calendar",
                "text": "Friday 3pm product review meeting"
            });
            let weak_result = weak
                .run_weak_task(&task, &input, &action_draft_schema())
                .await?;

            let mut env = serde_json::Map::new();
            env.insert("$input".to_owned(), input);
            let effect = EffectFrame {
                effect_id: uuid::Uuid::new_v4().to_string(),
                reason: "probe".to_owned(),
                failed_instruction: None,
                continuation: Continuation {
                    program_id: "probe".to_owned(),
                    pc: 1,
                    resume_var: Some("draft".to_owned()),
                    env,
                    expected_schema: action_draft_schema(),
                },
                observations: vec![serde_json::to_value(&weak_result)?],
            };
            let decision = strong.think(&effect).await?;
            let ok = matches!(
                decision,
                ThinkDecision::ResumeWithValue { .. }
                    | ThinkDecision::RequestWeakProbe { .. }
                    | ThinkDecision::Abort { .. }
            );

            println!("{}", json!({ "ok": ok, "status": "OK" }));
        }
    }

    Ok(())
}

fn constrain_compiled_program_contract(
    mut program: Program,
    input_schema: Value,
    output_schema: Value,
) -> Program {
    program.input_schema = input_schema;
    program.output_schema = output_schema;
    program
}

async fn run_inputs<W, S>(
    runtime: &Runtime<W, S>,
    program: Program,
    inputs: Vec<Value>,
) -> Result<Vec<Value>>
where
    W: WeakModel,
    S: StrongModel,
{
    let mut outputs = Vec::with_capacity(inputs.len());
    for input in inputs {
        outputs.push(runtime.run_program(program.clone(), input).await?);
    }
    Ok(outputs)
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

fn read_inputs(path: &PathBuf) -> Result<Vec<Value>> {
    let raw = std::fs::read_to_string(path)
        .with_context(|| format!("failed to read input file {}", path.display()))?;
    let value: Value = serde_json::from_str(&raw)
        .with_context(|| format!("invalid JSON in {}", path.display()))?;

    match value {
        Value::Array(items) => Ok(items),
        Value::Object(_) => Ok(vec![value]),
        _ => Err(anyhow::anyhow!(
            "input must be a JSON object or an array of JSON objects"
        )),
    }
}

fn emit_trace(trace: TraceCollector, trace_json: bool) -> Result<()> {
    if trace_json {
        for event in trace.events() {
            eprintln!("{}", serde_json::to_string(&event)?);
        }
    }
    Ok(())
}
