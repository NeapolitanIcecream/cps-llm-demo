use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use serde_json::json;

use crate::config::{
    DEFAULT_BASE_URL, DEFAULT_STRONG_MODEL, DEFAULT_THRESHOLD, DEFAULT_WEAK_MODEL, ModelConfig,
};
use crate::domain::{DecisionSource, MessageEvent, ResolvedIntent};
use crate::effects::ThinkDecision;
use crate::models::{ResponsesStrongModel, ResponsesWeakModel, StrongModel, WeakModel};
use crate::runtime::{CapturePolicy, Runtime, make_effect_frame};
use crate::schema::schema_bundle;
use crate::trace::TraceCollector;

#[derive(Debug, Parser)]
#[command(version, about = "CPS-style LLM runtime demo")]
pub struct Cli {
    #[command(subcommand)]
    pub command: Command,
}

#[derive(Debug, Subcommand)]
pub enum Command {
    Run {
        input: PathBuf,

        #[arg(long, env = "OPENAI_BASE_URL", default_value = DEFAULT_BASE_URL)]
        base_url: String,

        #[arg(long, env = "OPENAI_API_KEY")]
        api_key: Option<String>,

        #[arg(long, env = "CPS_WEAK_MODEL", default_value = DEFAULT_WEAK_MODEL)]
        weak_model: String,

        #[arg(long, env = "CPS_STRONG_MODEL", default_value = DEFAULT_STRONG_MODEL)]
        strong_model: String,

        #[arg(long, env = "CPS_THRESHOLD", default_value_t = DEFAULT_THRESHOLD)]
        threshold: f32,

        #[arg(
            long,
            value_enum,
            default_value_t = CapturePolicyArg::ConfidenceOnly
        )]
        capture_policy: CapturePolicyArg,

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

#[derive(Debug, Clone, Copy, clap::ValueEnum)]
pub enum CapturePolicyArg {
    ConfidenceOnly,
    AlwaysAfterWeak,
}

impl From<CapturePolicyArg> for CapturePolicy {
    fn from(value: CapturePolicyArg) -> Self {
        match value {
            CapturePolicyArg::ConfidenceOnly => Self::ConfidenceOnly,
            CapturePolicyArg::AlwaysAfterWeak => Self::AlwaysAfterWeak,
        }
    }
}

pub async fn run() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Command::Run {
            input,
            base_url,
            api_key,
            weak_model,
            strong_model,
            threshold,
            capture_policy,
            trace_json,
        } => {
            let config = ModelConfig::new(base_url, api_key, weak_model, strong_model, threshold)?;
            let messages = read_messages(&input)?;
            let trace = TraceCollector::default();
            let client = config.responses_client();
            let weak = ResponsesWeakModel::new(client.clone(), config.weak_model);
            let strong = ResponsesStrongModel::new(client, config.strong_model);
            let runtime = Runtime::new(
                weak,
                strong,
                config.threshold,
                capture_policy.into(),
                trace.clone(),
            );

            let mut outputs = Vec::with_capacity(messages.len());
            for message in messages {
                outputs.push(runtime.run_one(message).await?);
            }

            println!("{}", serde_json::to_string_pretty(&outputs)?);
            if trace_json {
                for event in trace.events() {
                    eprintln!("{}", serde_json::to_string(&event)?);
                }
            }
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
            let config = ModelConfig::new(
                base_url,
                api_key,
                weak_model,
                strong_model,
                DEFAULT_THRESHOLD,
            )?;
            let client = config.responses_client();
            let weak = ResponsesWeakModel::new(client.clone(), config.weak_model);
            let strong = ResponsesStrongModel::new(client, config.strong_model);

            let event = MessageEvent {
                event_id: "probe-calendar".to_owned(),
                text: "Friday 3pm product review meeting".to_owned(),
            };
            let weak_guess = weak.classify_message(&event).await?;

            let effect = make_effect_frame(
                MessageEvent {
                    event_id: "probe-think".to_owned(),
                    text: "Should we continue with this direction?".to_owned(),
                },
                Some(weak_guess),
                None,
                "probe",
            );
            let decision = strong.think(&effect).await?;
            let ok = matches!(
                decision,
                ThinkDecision::Value(ResolvedIntent {
                    source: DecisionSource::StrongThink,
                    ..
                }) | ThinkDecision::Abort { .. }
            );

            println!("{}", json!({ "ok": ok, "status": "OK" }));
        }
    }

    Ok(())
}

fn read_messages(input: &PathBuf) -> Result<Vec<MessageEvent>> {
    let raw = std::fs::read_to_string(input)
        .with_context(|| format!("failed to read input file {}", input.display()))?;
    serde_json::from_str(&raw).with_context(|| format!("invalid JSON in {}", input.display()))
}
