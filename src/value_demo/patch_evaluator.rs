use anyhow::{Context, Result};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::models::EffectHandler;
use crate::program::{Program, ProgramPatch};
use crate::runtime::Runtime;
use crate::trace::TraceCollector;
use crate::validator::validate_patch;
use crate::value_demo::event_source::EventEnvelope;
use crate::value_demo::local_tools::LocalToolRegistry;
use crate::value_demo::metrics::{EventRunResult, MetricsCollector, RunMetrics, RunMode};

#[derive(Debug, Clone)]
pub struct PatchEvaluator {
    pub min_fast_path_hit_rate_delta: f64,
    pub max_strong_think_rate_delta: f64,
}

impl Default for PatchEvaluator {
    fn default() -> Self {
        Self {
            min_fast_path_hit_rate_delta: 0.0,
            max_strong_think_rate_delta: 0.0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
pub struct PatchEvaluationReport {
    pub patch_id: String,
    pub accepted: bool,
    pub reason: String,
    pub original_metrics: Option<RunMetrics>,
    pub patched_metrics: Option<RunMetrics>,
    pub installed_version: Option<String>,
}

impl PatchEvaluationReport {
    pub fn rejected(patch_id: &str, reason: &str) -> Self {
        Self {
            patch_id: patch_id.to_owned(),
            accepted: false,
            reason: reason.to_owned(),
            original_metrics: None,
            patched_metrics: None,
            installed_version: None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct PatchEvaluationOutcome {
    pub patched_program: Option<Program>,
    pub report: PatchEvaluationReport,
}

impl PatchEvaluator {
    pub async fn evaluate<W, S>(
        &self,
        original: &Program,
        patch: &ProgramPatch,
        events: &[EventEnvelope],
        weak: W,
        strong: S,
        local_tools: LocalToolRegistry,
    ) -> Result<PatchEvaluationOutcome>
    where
        W: EffectHandler + Clone + 'static,
        S: EffectHandler + Clone + 'static,
    {
        let patched = match validate_patch(original, patch) {
            Ok(program) => program,
            Err(err) => {
                return Ok(PatchEvaluationOutcome {
                    patched_program: None,
                    report: PatchEvaluationReport::rejected(
                        &patch.patch_id,
                        &format!("patch validation failed: {err}"),
                    ),
                });
            }
        };

        let original_metrics = run_eval_program(
            "patch-eval-original",
            original.clone(),
            events,
            weak.clone(),
            strong.clone(),
            local_tools.clone(),
        )
        .await
        .context("failed to evaluate original program")?;
        let patched_metrics = run_eval_program(
            "patch-eval-patched",
            patched.clone(),
            events,
            weak,
            strong,
            local_tools,
        )
        .await
        .context("failed to evaluate patched program")?;

        let fast_path_delta =
            patched_metrics.fast_path_hit_rate() - original_metrics.fast_path_hit_rate();
        let strong_think_delta =
            patched_metrics.strong_think_rate() - original_metrics.strong_think_rate();
        let no_new_failures = patched_metrics.events_failed <= original_metrics.events_failed;
        let improved = fast_path_delta > self.min_fast_path_hit_rate_delta
            || strong_think_delta < self.max_strong_think_rate_delta;
        let accepted = no_new_failures && improved;
        let reason = if accepted {
            "patched program improved fast-path coverage or strong Think rate".to_owned()
        } else if !no_new_failures {
            "patched program introduced new event failures".to_owned()
        } else {
            "patched program did not improve fast-path coverage or strong Think rate".to_owned()
        };

        Ok(PatchEvaluationOutcome {
            patched_program: accepted.then_some(patched),
            report: PatchEvaluationReport {
                patch_id: patch.patch_id.clone(),
                accepted,
                reason,
                original_metrics: Some(original_metrics),
                patched_metrics: Some(patched_metrics),
                installed_version: None,
            },
        })
    }
}

async fn run_eval_program<W, S>(
    run_id: &str,
    program: Program,
    events: &[EventEnvelope],
    weak: W,
    strong: S,
    local_tools: LocalToolRegistry,
) -> Result<RunMetrics>
where
    W: EffectHandler + Clone + 'static,
    S: EffectHandler + Clone + 'static,
{
    let trace = TraceCollector::default();
    let runtime = Runtime::with_local_tools(weak, strong, trace.clone(), local_tools);
    let mut collector = MetricsCollector::new(run_id, RunMode::ValueDemo);
    collector.metrics_mut().program_id = Some(program.program_id.clone());
    collector.metrics_mut().program_version = Some(program.version.clone());

    for event in events {
        let result = runtime
            .run_program(program.clone(), event.as_program_input())
            .await;
        collector.observe_event_result(&EventRunResult {
            succeeded: result.is_ok(),
        });
    }
    for event in trace.events() {
        collector.observe_trace_event(&event);
    }
    Ok(collector.finish())
}
