use std::sync::Arc;

use anyhow::Result;
use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::effects::{HandlerBudget, HandlerDecision, HandlerRequest, RuntimeBudget};
use crate::engine::event_source::EventSource;
use crate::models::EffectHandler;
use crate::program::{EffectCall, ModelStrength, ModelTaskSpec};
use crate::store::metrics_store::{FileMetricsStore, RunMetrics};
use crate::store::program_registry::FileProgramRegistry;
use crate::store::state_dir::{StateDir, now_string};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BaselineSummary {
    pub run_id: String,
    pub workflow_id: String,
    pub events_total: u64,
    pub strong_model_task_calls: u64,
    pub weak_model_calls: u64,
    pub fast_path_hits: u64,
}

pub async fn baseline_strong_direct<S>(
    state: StateDir,
    workflow_id: &str,
    task_spec: String,
    mut event_source: S,
    strong: Arc<dyn EffectHandler>,
) -> Result<BaselineSummary>
where
    S: EventSource,
{
    state.ensure_workflow_layout(workflow_id)?;
    let expected_schema = FileProgramRegistry::new(state.clone())
        .load_latest(workflow_id)
        .map(|program| program.output_schema)
        .unwrap_or_else(|_| json!({ "type": "object" }));
    let run_id = uuid::Uuid::new_v4().to_string();
    let mut metrics = RunMetrics::new(
        run_id.clone(),
        workflow_id.to_owned(),
        "strong_direct".to_owned(),
        "strong_direct".to_owned(),
        "baseline".to_owned(),
        now_string(),
    );

    while let Some(event) = event_source.next_event()? {
        metrics.events_total += 1;
        let request = HandlerRequest {
            run_id: Some(run_id.clone()),
            budget_scope_id: None,
            workflow_id: Some(workflow_id.to_owned()),
            phase: Some("strong_direct".to_owned()),
            effect: EffectCall::ModelTask {
                strength: ModelStrength::Strong,
                task: ModelTaskSpec {
                    name: "strong_direct_baseline".to_owned(),
                    instructions: task_spec.clone(),
                },
            },
            input: event,
            expected_schema: expected_schema.clone(),
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
            Ok(HandlerDecision::Abort { .. }) | Err(_) => metrics.events_failed += 1,
            Ok(_) => metrics.events_succeeded += 1,
        }
        metrics.strong_model_task_calls += 1;
        metrics.estimated_model_calls += 1;
    }

    metrics.finished_at = now_string();
    FileMetricsStore::new(state).write(&metrics)?;
    Ok(BaselineSummary {
        run_id,
        workflow_id: workflow_id.to_owned(),
        events_total: metrics.events_total,
        strong_model_task_calls: metrics.strong_model_task_calls,
        weak_model_calls: metrics.weak_model_calls,
        fast_path_hits: metrics.fast_path_hits,
    })
}
