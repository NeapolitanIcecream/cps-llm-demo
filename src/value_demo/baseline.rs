use anyhow::Result;
use serde_json::json;

use crate::effects::{HandlerBudget, HandlerDecision, HandlerRequest, RuntimeBudget};
use crate::models::EffectHandler;
use crate::program::{EffectCall, ModelStrength, ModelTaskSpec};
use crate::schema::action_draft_schema;
use crate::trace::TraceCollector;
use crate::value_demo::event_source::EventEnvelope;
use crate::value_demo::metrics::{RunMetrics, RunMode};

pub async fn run_strong_direct_baseline<H>(
    run_id: impl Into<String>,
    task_spec: &str,
    events: &[EventEnvelope],
    strong: &H,
    trace: &TraceCollector,
) -> Result<RunMetrics>
where
    H: EffectHandler,
{
    let run_id = run_id.into();
    let mut metrics = RunMetrics {
        run_id,
        mode: RunMode::BaselineStrongDirect,
        events_total: events.len() as u64,
        strong_direct_calls: events.len() as u64,
        strong_model_calls: events.len() as u64,
        ..RunMetrics::default()
    };

    for event in events {
        trace.emit(
            "handler_request",
            &event.event_id,
            json!({
                "handler": "strong_model",
                "effect": "model_task",
                "strength": "strong",
                "task": "strong_direct_full_task",
                "source": "strong_model",
                "depth": 0,
            }),
        );
        let request = HandlerRequest {
            effect: EffectCall::ModelTask {
                strength: ModelStrength::Strong,
                task: ModelTaskSpec {
                    name: "strong_direct_full_task".to_owned(),
                    instructions: task_spec.to_owned(),
                },
            },
            input: event.as_program_input(),
            expected_schema: action_draft_schema(),
            continuation_summary: None,
            effect_frame: None,
            observations: Vec::new(),
            budget: HandlerBudget {
                effect_depth: 0,
                effects_remaining: RuntimeBudget::default().max_effects,
                handler_reentries_remaining: RuntimeBudget::default().max_handler_reentries,
            },
        };
        match strong.handle(request).await {
            Ok(HandlerDecision::ReturnValue { .. }) => metrics.events_succeeded += 1,
            Ok(_) | Err(_) => metrics.events_failed += 1,
        }
    }

    Ok(metrics)
}
