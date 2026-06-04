use serde::{Deserialize, Serialize};
use std::sync::Arc;
use uuid::Uuid;

use anyhow::Result;
use serde_json::Value;

use crate::effects::RuntimeBudget;
use crate::local_tools::LocalToolRegistry;
use crate::models::EffectHandler;
use crate::observability::metrics::MetricsAccumulator;
use crate::program::{EffectCall, Instr, PatchOp, Program, ProgramPatch};
use crate::runtime::Runtime;
use crate::store::continuation_store::{FileEffectFrameEncoder, FrameEncodingConfig};
use crate::store::metrics_store::RunMetrics;
use crate::store::state_dir::{StateDir, now_string};
use crate::trace::TraceCollector;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PatchEvaluationReport {
    pub accepted: bool,
    pub reason: String,
}

pub fn evaluate_metrics(
    base: &RunMetrics,
    patched: &RunMetrics,
    patch_has_fast_path: bool,
    continuation_frame_limit: u64,
) -> PatchEvaluationReport {
    if patched.events_failed > 0 {
        return rejected("patched run failed events");
    }
    if patched.strong_think_calls > base.strong_think_calls {
        return rejected("patched run increased StrongThink calls");
    }
    if patched.continuation_frame_bytes_p95 > continuation_frame_limit {
        return rejected("patched run exceeded continuation frame size limit");
    }
    if patch_has_fast_path && patched.fast_path_hits <= base.fast_path_hits {
        return rejected("fast-path patch did not increase fast-path hits");
    }
    PatchEvaluationReport {
        accepted: true,
        reason: "accepted".to_owned(),
    }
}

pub async fn evaluate_program_on_events(
    workflow_id: &str,
    program: Program,
    events: &[Value],
    weak: Arc<dyn EffectHandler>,
    strong: Arc<dyn EffectHandler>,
) -> Result<RunMetrics> {
    let run_id = format!("eval-{}", Uuid::new_v4());
    let eval_state = StateDir::new(std::env::temp_dir().join(format!("{run_id}-state")));
    let frame_encoder = Arc::new(FileEffectFrameEncoder::new(
        eval_state.clone(),
        workflow_id.to_owned(),
        FrameEncodingConfig::default(),
    ));
    let mut metrics = RunMetrics::new(
        run_id.clone(),
        workflow_id.to_owned(),
        "patch_evaluation".to_owned(),
        program.program_id.clone(),
        program.version.clone(),
        now_string(),
    );
    let mut accumulator = MetricsAccumulator::default();

    for event in events {
        metrics.events_total += 1;
        let trace = TraceCollector::default();
        let runtime = Runtime::with_frame_encoder(
            Arc::clone(&weak),
            Arc::clone(&strong),
            LocalToolRegistry::default(),
            frame_encoder.clone(),
            trace.clone(),
            RuntimeBudget::default(),
        );
        let result = runtime.run_program(program.clone(), event.clone()).await;
        if result.is_ok() {
            metrics.events_succeeded += 1;
        } else {
            metrics.events_failed += 1;
        }
        accumulator.update_from_trace(&mut metrics, &trace.events());
    }

    accumulator.finalize(&mut metrics);
    metrics.finished_at = now_string();
    let _ = std::fs::remove_dir_all(eval_state.root());
    Ok(metrics)
}

pub fn patch_has_fast_path(patch: &ProgramPatch) -> bool {
    patch.operations.iter().any(operation_has_fast_path)
}

fn operation_has_fast_path(operation: &PatchOp) -> bool {
    match operation {
        PatchOp::ReplaceInstruction { instr, .. } | PatchOp::InsertInstruction { instr, .. } => {
            instr_has_fast_path(instr)
        }
        PatchOp::AddFunction { function, .. } => function.body.iter().any(instr_has_fast_path),
        PatchOp::UpdateAcceptancePolicy { .. } => false,
    }
}

fn instr_has_fast_path(instr: &Instr) -> bool {
    matches!(
        instr,
        Instr::Perform {
            effect: EffectCall::LocalTool { tool_name, .. },
            ..
        } if tool_name == "fast_path_apply"
    )
}

fn rejected(reason: &str) -> PatchEvaluationReport {
    PatchEvaluationReport {
        accepted: false,
        reason: reason.to_owned(),
    }
}
