use anyhow::{Result, anyhow};
use serde::{Deserialize, Serialize};

use crate::store::state_dir::{StateDir, read_json, write_json_pretty};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RunMetrics {
    pub run_id: String,
    pub workflow_id: String,
    pub mode: String,
    pub program_id: String,
    pub program_version: String,

    pub events_total: u64,
    pub events_succeeded: u64,
    pub events_failed: u64,

    pub program_compile_calls: u64,

    pub weak_model_calls: u64,
    pub strong_model_task_calls: u64,
    pub strong_think_calls: u64,

    pub local_tool_calls: u64,
    pub fast_path_hits: u64,
    pub fast_path_misses: u64,

    pub effects_total: u64,
    pub effects_accepted_without_capture: u64,
    pub effects_captured: u64,
    pub nested_effect_requests: u64,

    pub probe_requests: u64,
    pub probe_weak_model_calls: u64,
    pub probe_local_tool_calls: u64,
    pub probe_reentries: u64,

    pub patches_proposed: u64,
    pub patches_validated: u64,
    pub patches_installed: u64,
    pub patches_rejected: u64,

    pub continuation_frames_total: u64,
    pub continuation_frame_bytes_avg: f64,
    pub continuation_frame_bytes_p50: u64,
    pub continuation_frame_bytes_p95: u64,
    pub continuation_frame_bytes_max: u64,

    pub estimated_input_tokens: u64,
    pub estimated_output_tokens: u64,
    pub estimated_model_calls: u64,

    pub started_at: String,
    pub finished_at: String,
}

impl RunMetrics {
    pub fn new(
        run_id: String,
        workflow_id: String,
        mode: String,
        program_id: String,
        program_version: String,
        started_at: String,
    ) -> Self {
        Self {
            run_id,
            workflow_id,
            mode,
            program_id,
            program_version,
            events_total: 0,
            events_succeeded: 0,
            events_failed: 0,
            program_compile_calls: 0,
            weak_model_calls: 0,
            strong_model_task_calls: 0,
            strong_think_calls: 0,
            local_tool_calls: 0,
            fast_path_hits: 0,
            fast_path_misses: 0,
            effects_total: 0,
            effects_accepted_without_capture: 0,
            effects_captured: 0,
            nested_effect_requests: 0,
            probe_requests: 0,
            probe_weak_model_calls: 0,
            probe_local_tool_calls: 0,
            probe_reentries: 0,
            patches_proposed: 0,
            patches_validated: 0,
            patches_installed: 0,
            patches_rejected: 0,
            continuation_frames_total: 0,
            continuation_frame_bytes_avg: 0.0,
            continuation_frame_bytes_p50: 0,
            continuation_frame_bytes_p95: 0,
            continuation_frame_bytes_max: 0,
            estimated_input_tokens: 0,
            estimated_output_tokens: 0,
            estimated_model_calls: 0,
            started_at,
            finished_at: String::new(),
        }
    }

    pub fn fast_path_hit_rate(&self) -> f64 {
        let denominator = self.fast_path_hits + self.fast_path_misses;
        if denominator == 0 {
            0.0
        } else {
            self.fast_path_hits as f64 / denominator as f64
        }
    }

    pub fn strong_think_rate(&self) -> f64 {
        if self.events_total == 0 {
            0.0
        } else {
            self.strong_think_calls as f64 / self.events_total as f64
        }
    }
}

#[derive(Debug, Clone)]
pub struct FileMetricsStore {
    state: StateDir,
}

impl FileMetricsStore {
    pub fn new(state: StateDir) -> Self {
        Self { state }
    }

    pub fn write(&self, metrics: &RunMetrics) -> Result<()> {
        self.state.ensure_workflow_layout(&metrics.workflow_id)?;
        write_json_pretty(
            &self
                .state
                .workflow_dir(&metrics.workflow_id)
                .join("metrics")
                .join(format!("{}.json", metrics.run_id)),
            metrics,
        )
    }

    pub fn read(&self, workflow_id: &str, run_id: &str) -> Result<RunMetrics> {
        read_json(
            &self
                .state
                .workflow_dir(workflow_id)
                .join("metrics")
                .join(format!("{run_id}.json")),
        )
    }

    pub fn find_run(&self, run_id: &str) -> Result<RunMetrics> {
        let workflows = self.state.root().join("workflows");
        for workflow in std::fs::read_dir(&workflows)? {
            let workflow = workflow?;
            if !workflow.file_type()?.is_dir() {
                continue;
            }
            let metrics = workflow
                .path()
                .join("metrics")
                .join(format!("{run_id}.json"));
            if metrics.exists() {
                return read_json(&metrics);
            }
        }
        Err(anyhow!("metrics run {run_id} not found"))
    }

    pub fn list(&self, workflow_id: &str) -> Result<Vec<RunMetrics>> {
        let metrics_dir = self.state.workflow_dir(workflow_id).join("metrics");
        let mut runs = Vec::new();
        if !metrics_dir.exists() {
            return Ok(runs);
        }
        for entry in std::fs::read_dir(metrics_dir)? {
            let entry = entry?;
            if entry.file_type()?.is_file() {
                runs.push(read_json(&entry.path())?);
            }
        }
        runs.sort_by(|left: &RunMetrics, right| {
            left.started_at
                .cmp(&right.started_at)
                .then_with(|| left.program_version.cmp(&right.program_version))
                .then_with(|| left.run_id.cmp(&right.run_id))
        });
        Ok(runs)
    }
}
