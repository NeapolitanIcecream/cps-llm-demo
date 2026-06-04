use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::value_demo::metrics::RunMetrics;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MetricsComparisonReport {
    pub baseline_strong_calls: u64,
    pub round1_strong_think_rate: f64,
    pub round2_strong_think_rate: f64,
    pub round1_fast_path_hit_rate: f64,
    pub round2_fast_path_hit_rate: f64,
    pub strong_calls_avoided_round2_vs_baseline: i64,
    pub value_claim_passed: bool,
    pub claims: Value,
}

pub fn build_metrics_report(
    baseline: &RunMetrics,
    round1: &RunMetrics,
    round2: &RunMetrics,
) -> MetricsComparisonReport {
    let high_frequency_events = round2.events_total >= 20;
    let strong_think_not_every_event = round2.strong_think_calls < round2.events_total;
    let fast_path_hit_rate_increased = round2.fast_path_hit_rate() > round1.fast_path_hit_rate();
    let strong_think_rate_decreased = round2.strong_think_rate() < round1.strong_think_rate();
    let patch_installed = round1.patches_installed > 0;
    let continuation_frames_small = round2
        .continuation_frame_bytes_p95
        .map(|p95| p95 <= 16 * 1024)
        .unwrap_or(true);
    let strong_calls_avoided = baseline.strong_direct_calls as i64
        - (round2.strong_think_calls + round2.program_compile_calls) as i64;
    let strong_calls_avoided_vs_baseline = strong_calls_avoided > 0;
    let value_claim_passed = high_frequency_events
        && strong_think_not_every_event
        && fast_path_hit_rate_increased
        && strong_think_rate_decreased
        && patch_installed
        && continuation_frames_small
        && strong_calls_avoided_vs_baseline;

    MetricsComparisonReport {
        baseline_strong_calls: baseline.strong_direct_calls,
        round1_strong_think_rate: round1.strong_think_rate(),
        round2_strong_think_rate: round2.strong_think_rate(),
        round1_fast_path_hit_rate: round1.fast_path_hit_rate(),
        round2_fast_path_hit_rate: round2.fast_path_hit_rate(),
        strong_calls_avoided_round2_vs_baseline: strong_calls_avoided,
        value_claim_passed,
        claims: json!({
            "high_frequency_events": high_frequency_events,
            "amortized_strong_compile": round2.program_compile_calls <= 1,
            "strong_think_not_every_event": strong_think_not_every_event,
            "fast_path_hit_rate_increased": fast_path_hit_rate_increased,
            "strong_think_rate_decreased": strong_think_rate_decreased,
            "patch_installed": patch_installed,
            "continuation_frames_small": continuation_frames_small,
            "strong_calls_avoided_vs_baseline": strong_calls_avoided_vs_baseline,
        }),
    }
}
