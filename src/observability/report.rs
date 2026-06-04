use serde::{Deserialize, Serialize};

use crate::store::metrics_store::RunMetrics;
use crate::store::program_registry::ProgramMetadata;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MetricsReport {
    pub workflow_id: String,
    pub latest_program_version: Option<String>,
    pub runs: Vec<RunMetrics>,
    pub summary: MetricsSummary,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MetricsSummary {
    pub events_total: u64,
    pub strong_direct_calls: u64,
    pub cps_strong_compile_calls: u64,
    pub cps_strong_think_calls_round1: u64,
    pub cps_strong_think_calls_round2: u64,
    pub fast_path_hit_rate_round1: f64,
    pub fast_path_hit_rate_round2: f64,
    pub strong_think_rate_round1: f64,
    pub strong_think_rate_round2: f64,
    pub strong_call_reduction_vs_baseline: f64,
    pub continuation_frame_bytes_p95: u64,
    pub program_version_advanced: bool,
}

pub fn build_metrics_report(
    workflow_id: &str,
    latest_program_version: Option<String>,
    mut runs: Vec<RunMetrics>,
    versions: &[ProgramMetadata],
) -> MetricsReport {
    runs.sort_by(|left, right| {
        left.started_at
            .cmp(&right.started_at)
            .then_with(|| left.program_version.cmp(&right.program_version))
            .then_with(|| left.run_id.cmp(&right.run_id))
    });
    let stream_runs = runs
        .iter()
        .filter(|run| run.mode == "stream")
        .collect::<Vec<_>>();
    let round1 = stream_runs.first().copied();
    let round2 = stream_runs
        .last()
        .copied()
        .filter(|_| stream_runs.len() > 1);
    let baseline = runs.iter().find(|run| run.mode == "strong_direct");

    let cps_strong_calls = runs
        .iter()
        .map(|run| run.program_compile_calls)
        .sum::<u64>()
        + stream_runs
            .iter()
            .map(|run| run.strong_think_calls + run.strong_model_task_calls)
            .sum::<u64>();
    let strong_direct_calls = baseline
        .map(|run| run.strong_model_task_calls + run.strong_think_calls)
        .unwrap_or(0);
    let stream_events_total = stream_runs.iter().map(|run| run.events_total).sum::<u64>();
    let baseline_events_total = baseline.map(|run| run.events_total).unwrap_or(0);
    let strong_call_reduction_vs_baseline =
        if strong_direct_calls == 0 || stream_events_total == 0 || baseline_events_total == 0 {
            0.0
        } else {
            let cps_rate = cps_strong_calls as f64 / stream_events_total as f64;
            let baseline_rate = strong_direct_calls as f64 / baseline_events_total as f64;
            1.0 - (cps_rate / baseline_rate)
        };

    let summary = MetricsSummary {
        events_total: stream_events_total,
        strong_direct_calls,
        cps_strong_compile_calls: runs.iter().map(|run| run.program_compile_calls).sum(),
        cps_strong_think_calls_round1: round1.map(|run| run.strong_think_calls).unwrap_or(0),
        cps_strong_think_calls_round2: round2.map(|run| run.strong_think_calls).unwrap_or(0),
        fast_path_hit_rate_round1: round1.map(|run| run.fast_path_hit_rate()).unwrap_or(0.0),
        fast_path_hit_rate_round2: round2.map(|run| run.fast_path_hit_rate()).unwrap_or(0.0),
        strong_think_rate_round1: round1.map(|run| run.strong_think_rate()).unwrap_or(0.0),
        strong_think_rate_round2: round2.map(|run| run.strong_think_rate()).unwrap_or(0.0),
        strong_call_reduction_vs_baseline,
        continuation_frame_bytes_p95: stream_runs
            .iter()
            .map(|run| run.continuation_frame_bytes_p95)
            .max()
            .unwrap_or(0),
        program_version_advanced: versions.len() > 1,
    };

    MetricsReport {
        workflow_id: workflow_id.to_owned(),
        latest_program_version,
        runs,
        summary,
    }
}
