use serde::{Deserialize, Serialize};

use crate::store::metrics_store::RunMetrics;

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

fn rejected(reason: &str) -> PatchEvaluationReport {
    PatchEvaluationReport {
        accepted: false,
        reason: reason.to_owned(),
    }
}
