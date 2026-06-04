use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::trace::TraceEvent;

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
pub struct RunMetrics {
    pub run_id: String,
    pub mode: RunMode,

    pub program_id: Option<String>,
    pub program_version: Option<String>,

    pub events_total: u64,
    pub events_succeeded: u64,
    pub events_failed: u64,

    pub program_compile_calls: u64,

    pub weak_model_calls: u64,
    pub strong_model_calls: u64,
    pub strong_think_calls: u64,
    pub strong_direct_calls: u64,

    pub effects_total: u64,
    pub effects_model_task_weak: u64,
    pub effects_model_task_strong: u64,
    pub effects_think: u64,
    pub effects_local_tool: u64,

    pub effects_accepted_without_capture: u64,
    pub effects_captured: u64,

    pub fast_path_hits: u64,
    pub fast_path_misses: u64,

    pub patches_proposed: u64,
    pub patches_validated: u64,
    pub patches_installed: u64,
    pub patches_rejected: u64,

    pub continuation_frames_total: u64,
    pub continuation_frame_bytes_total: u64,
    pub continuation_frame_bytes_max: u64,
    pub continuation_frame_bytes_p95: Option<u64>,

    pub estimated_strong_calls_avoided_vs_direct: i64,
}

impl Default for RunMetrics {
    fn default() -> Self {
        Self {
            run_id: String::new(),
            mode: RunMode::ValueDemo,
            program_id: None,
            program_version: None,
            events_total: 0,
            events_succeeded: 0,
            events_failed: 0,
            program_compile_calls: 0,
            weak_model_calls: 0,
            strong_model_calls: 0,
            strong_think_calls: 0,
            strong_direct_calls: 0,
            effects_total: 0,
            effects_model_task_weak: 0,
            effects_model_task_strong: 0,
            effects_think: 0,
            effects_local_tool: 0,
            effects_accepted_without_capture: 0,
            effects_captured: 0,
            fast_path_hits: 0,
            fast_path_misses: 0,
            patches_proposed: 0,
            patches_validated: 0,
            patches_installed: 0,
            patches_rejected: 0,
            continuation_frames_total: 0,
            continuation_frame_bytes_total: 0,
            continuation_frame_bytes_max: 0,
            continuation_frame_bytes_p95: None,
            estimated_strong_calls_avoided_vs_direct: 0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum RunMode {
    ValueDemo,
    BaselineStrongDirect,
}

impl Default for RunMode {
    fn default() -> Self {
        RunMode::ValueDemo
    }
}

impl RunMetrics {
    pub fn strong_think_rate(&self) -> f64 {
        if self.events_total == 0 {
            0.0
        } else {
            self.strong_think_calls as f64 / self.events_total as f64
        }
    }

    pub fn fast_path_hit_rate(&self) -> f64 {
        let total = self.fast_path_hits + self.fast_path_misses;
        if total == 0 {
            0.0
        } else {
            self.fast_path_hits as f64 / total as f64
        }
    }

    pub fn weak_or_fast_path_rate(&self) -> f64 {
        if self.events_total == 0 {
            0.0
        } else {
            (self.events_total.saturating_sub(self.strong_think_calls)) as f64
                / self.events_total as f64
        }
    }

    pub fn avg_continuation_frame_bytes(&self) -> Option<f64> {
        if self.continuation_frames_total == 0 {
            None
        } else {
            Some(self.continuation_frame_bytes_total as f64 / self.continuation_frames_total as f64)
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EventRunResult {
    pub succeeded: bool,
}

pub struct MetricsCollector {
    metrics: RunMetrics,
    continuation_frame_sizes: Vec<u64>,
}

impl MetricsCollector {
    pub fn new(run_id: impl Into<String>, mode: RunMode) -> Self {
        Self {
            metrics: RunMetrics {
                run_id: run_id.into(),
                mode,
                ..RunMetrics::default()
            },
            continuation_frame_sizes: Vec::new(),
        }
    }

    pub fn metrics_mut(&mut self) -> &mut RunMetrics {
        &mut self.metrics
    }

    pub fn observe_trace_event(&mut self, event: &TraceEvent) {
        match event.event.as_str() {
            "handler_request" => self.observe_handler_request(&event.detail),
            "perform_effect" => self.observe_perform_effect(&event.detail),
            "capture_continuation" => {
                self.metrics.effects_captured += 1;
                if let Some(bytes) = u64_field(&event.detail, "frame_bytes") {
                    self.observe_frame_size(bytes);
                }
            }
            "handler_decision" => {
                if event.detail["decision"] == "return_program_patch" {
                    self.metrics.patches_proposed += 1;
                }
                if event.detail["decision"] == "return_value" {
                    self.metrics.effects_accepted_without_capture += 1;
                }
            }
            "patch_proposed" => self.metrics.patches_proposed += 1,
            "patch_validated" => self.metrics.patches_validated += 1,
            "patch_installed" => self.metrics.patches_installed += 1,
            "patch_rejected" | "patch_invalid" => self.metrics.patches_rejected += 1,
            "fast_path_result" => {
                if event.detail["hit"].as_bool().unwrap_or(false) {
                    self.metrics.fast_path_hits += 1;
                } else {
                    self.metrics.fast_path_misses += 1;
                }
            }
            "continuation_compacted" => {
                if let Some(bytes) = u64_field(&event.detail, "public_bytes") {
                    self.observe_frame_size(bytes);
                }
            }
            "request_nested_effect" | "nested_effect_result" | "reenter_handler" => {}
            _ => {}
        }
    }

    pub fn observe_event_result(&mut self, result: &EventRunResult) {
        self.metrics.events_total += 1;
        if result.succeeded {
            self.metrics.events_succeeded += 1;
        } else {
            self.metrics.events_failed += 1;
        }
    }

    pub fn finish(mut self) -> RunMetrics {
        self.continuation_frame_sizes.sort_unstable();
        self.metrics.continuation_frames_total = self.continuation_frame_sizes.len() as u64;
        self.metrics.continuation_frame_bytes_total =
            self.continuation_frame_sizes.iter().sum::<u64>();
        self.metrics.continuation_frame_bytes_max = self
            .continuation_frame_sizes
            .iter()
            .copied()
            .max()
            .unwrap_or(0);
        self.metrics.continuation_frame_bytes_p95 = percentile(&self.continuation_frame_sizes, 95);
        self.metrics.estimated_strong_calls_avoided_vs_direct = self.metrics.events_total as i64
            - (self.metrics.strong_think_calls + self.metrics.program_compile_calls) as i64;
        self.metrics
    }

    fn observe_handler_request(&mut self, detail: &Value) {
        match detail.get("handler").and_then(Value::as_str) {
            Some("weak_model") => self.metrics.weak_model_calls += 1,
            Some("strong_model") => {
                self.metrics.strong_model_calls += 1;
                if detail.get("effect").and_then(Value::as_str) == Some("think") {
                    self.metrics.strong_think_calls += 1;
                }
            }
            Some("local_tool") => {}
            _ => {}
        }
        if detail.get("effect").and_then(Value::as_str) == Some("compile_program") {
            self.metrics.program_compile_calls += 1;
        }
    }

    fn observe_perform_effect(&mut self, detail: &Value) {
        self.metrics.effects_total += 1;
        match detail.get("effect").and_then(Value::as_str) {
            Some("model_task") => match detail.get("strength").and_then(Value::as_str) {
                Some("weak") => self.metrics.effects_model_task_weak += 1,
                Some("strong") => self.metrics.effects_model_task_strong += 1,
                _ => {}
            },
            Some("think") => self.metrics.effects_think += 1,
            Some("local_tool") => self.metrics.effects_local_tool += 1,
            _ => {}
        }
    }

    fn observe_frame_size(&mut self, bytes: u64) {
        self.continuation_frame_sizes.push(bytes);
    }
}

fn u64_field(value: &Value, name: &str) -> Option<u64> {
    value.get(name).and_then(Value::as_u64)
}

fn percentile(sorted_values: &[u64], percentile: u64) -> Option<u64> {
    if sorted_values.is_empty() {
        return None;
    }
    let rank = ((percentile as f64 / 100.0) * sorted_values.len() as f64).ceil() as usize;
    let index = rank.saturating_sub(1).min(sorted_values.len() - 1);
    Some(sorted_values[index])
}
