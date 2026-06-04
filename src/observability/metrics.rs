use serde_json::Value;

use crate::observability::cost_model::rough_token_estimate;
use crate::store::metrics_store::RunMetrics;
use crate::trace::TraceEvent;

#[derive(Debug, Default)]
pub struct MetricsAccumulator {
    frame_bytes: Vec<u64>,
}

impl MetricsAccumulator {
    pub fn update_from_trace(&mut self, metrics: &mut RunMetrics, events: &[TraceEvent]) {
        for event in events {
            match event.event.as_str() {
                "handler_request" => match event.detail.get("handler").and_then(Value::as_str) {
                    Some("weak_model") => metrics.weak_model_calls += 1,
                    Some("strong_model") => {
                        match event.detail.get("effect").and_then(Value::as_str) {
                            Some("think") => metrics.strong_think_calls += 1,
                            Some("model_task") => metrics.strong_model_task_calls += 1,
                            Some("compile_program") => metrics.program_compile_calls += 1,
                            _ => {}
                        }
                    }
                    Some("local_tool") => metrics.local_tool_calls += 1,
                    _ => {}
                },
                "perform_effect" => metrics.effects_total += 1,
                "capture_continuation" => metrics.effects_captured += 1,
                "request_nested_effect" => {
                    metrics.nested_effect_requests += 1;
                    metrics.probe_requests += 1;
                    match event.detail.get("to_handler").and_then(Value::as_str) {
                        Some("weak_model") => metrics.probe_weak_model_calls += 1,
                        Some("local_tool") => metrics.probe_local_tool_calls += 1,
                        _ => {}
                    }
                }
                "reenter_handler" => metrics.probe_reentries += 1,
                "fast_path_hit" => metrics.fast_path_hits += 1,
                "fast_path_miss" => metrics.fast_path_misses += 1,
                "patch_proposed" => metrics.patches_proposed += 1,
                "patch_validated" => metrics.patches_validated += 1,
                "patch_installed" => metrics.patches_installed += 1,
                "patch_rejected" | "patch_invalid" => metrics.patches_rejected += 1,
                "continuation_frame_encoded" => {
                    metrics.continuation_frames_total += 1;
                    if let Some(bytes) = event.detail.get("encoded_bytes").and_then(Value::as_u64) {
                        self.frame_bytes.push(bytes);
                    }
                    if let Some(bytes) = event.detail.get("original_bytes").and_then(Value::as_u64)
                    {
                        metrics.estimated_input_tokens += rough_token_estimate(bytes as usize);
                    }
                }
                _ => {}
            }
        }
        metrics.estimated_model_calls =
            metrics.weak_model_calls + metrics.strong_model_task_calls + metrics.strong_think_calls;
    }

    pub fn finalize(&mut self, metrics: &mut RunMetrics) {
        metrics.effects_accepted_without_capture = metrics
            .effects_total
            .saturating_sub(metrics.effects_captured);
        self.frame_bytes.sort_unstable();
        if self.frame_bytes.is_empty() {
            return;
        }
        let sum: u64 = self.frame_bytes.iter().sum();
        metrics.continuation_frame_bytes_avg = sum as f64 / self.frame_bytes.len() as f64;
        metrics.continuation_frame_bytes_p50 = percentile(&self.frame_bytes, 0.50);
        metrics.continuation_frame_bytes_p95 = percentile(&self.frame_bytes, 0.95);
        metrics.continuation_frame_bytes_max = *self.frame_bytes.last().unwrap_or(&0);
    }
}

fn percentile(values: &[u64], percentile: f64) -> u64 {
    if values.is_empty() {
        return 0;
    }
    let index = ((values.len() - 1) as f64 * percentile).ceil() as usize;
    values[index.min(values.len() - 1)]
}
