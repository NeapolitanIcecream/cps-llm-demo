use serde::{Deserialize, Serialize};

use crate::store::metrics_store::RunMetrics;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RunComparison {
    pub strong_calls_per_event_decreased: bool,
    pub fast_path_coverage_increased: bool,
    pub continuation_frame_size_bounded: bool,
    pub program_version_advanced: bool,
}

pub fn compare_runs(
    baseline: &RunMetrics,
    before: &RunMetrics,
    after: &RunMetrics,
    frame_limit: u64,
) -> RunComparison {
    let baseline_rate = strong_calls_per_event(baseline);
    let after_rate = strong_calls_per_event(after);
    RunComparison {
        strong_calls_per_event_decreased: after_rate < baseline_rate,
        fast_path_coverage_increased: after.fast_path_hit_rate() > before.fast_path_hit_rate(),
        continuation_frame_size_bounded: after.continuation_frame_bytes_p95 <= frame_limit,
        program_version_advanced: before.program_version != after.program_version,
    }
}

fn strong_calls_per_event(metrics: &RunMetrics) -> f64 {
    if metrics.events_total == 0 {
        return 0.0;
    }
    (metrics.strong_model_task_calls + metrics.strong_think_calls + metrics.program_compile_calls)
        as f64
        / metrics.events_total as f64
}
