use std::path::PathBuf;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use crate::store::state_dir::{StateDir, validate_path_component};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ModelUsage {
    pub input_tokens: u64,
    pub cached_input_tokens: u64,
    pub output_tokens: u64,
    pub reasoning_tokens: u64,
    pub total_tokens: u64,
}

impl ModelUsage {
    pub fn zero() -> Self {
        Self {
            input_tokens: 0,
            cached_input_tokens: 0,
            output_tokens: 0,
            reasoning_tokens: 0,
            total_tokens: 0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CostBreakdown {
    pub input_usd: f64,
    pub cached_input_usd: f64,
    pub output_usd: f64,
    pub total_usd: f64,
    pub estimated: bool,
}

impl CostBreakdown {
    pub fn zero(estimated: bool) -> Self {
        Self {
            input_usd: 0.0,
            cached_input_usd: 0.0,
            output_usd: 0.0,
            total_usd: 0.0,
            estimated,
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum CacheStatus {
    Miss,
    Hit,
    Bypass,
    Refresh,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ModelCallRecord {
    pub call_id: String,
    pub run_id: Option<String>,
    pub workflow_id: Option<String>,
    pub event_id: Option<String>,

    pub model: String,
    pub handler: String,
    pub effect_kind: String,
    pub task_name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub phase: Option<String>,

    pub request_hash: String,
    pub prompt_hash: String,
    pub schema_hash: String,
    pub model_config_hash: String,

    pub input_bytes: u64,
    pub output_bytes: u64,

    pub usage: Option<ModelUsage>,
    pub estimated_usage: ModelUsage,
    pub cost: CostBreakdown,

    pub latency_ms: u64,
    pub cache_status: CacheStatus,
    pub success: bool,
    pub error: Option<String>,
    pub created_at: String,
}

#[derive(Debug, Clone)]
pub struct FileModelCallStore {
    state: StateDir,
}

impl FileModelCallStore {
    pub fn new(state: StateDir) -> Self {
        Self { state }
    }

    pub fn append(&self, record: &ModelCallRecord) -> Result<()> {
        append_jsonl(&self.calls_path(), record)?;
        if let Some(run_id) = record.run_id.as_deref() {
            validate_path_component("run_id", run_id)?;
            append_jsonl(&self.by_run_path(run_id), record)?;
        }
        Ok(())
    }

    pub fn list_all(&self) -> Result<Vec<ModelCallRecord>> {
        read_jsonl(&self.calls_path())
    }

    pub fn list_by_run(&self, run_id: &str) -> Result<Vec<ModelCallRecord>> {
        validate_path_component("run_id", run_id)?;
        read_jsonl(&self.by_run_path(run_id))
    }

    fn calls_path(&self) -> PathBuf {
        self.state.root().join("model_calls").join("calls.jsonl")
    }

    fn by_run_path(&self, run_id: &str) -> PathBuf {
        self.state
            .root()
            .join("model_calls")
            .join("by_run")
            .join(format!("{run_id}.jsonl"))
    }
}

pub fn estimate_usage_from_bytes(input_bytes: u64, output_bytes: u64) -> ModelUsage {
    let input_tokens = conservative_bytes_to_tokens(input_bytes);
    let output_tokens = conservative_bytes_to_tokens(output_bytes);
    ModelUsage {
        input_tokens,
        cached_input_tokens: 0,
        output_tokens,
        reasoning_tokens: 0,
        total_tokens: input_tokens + output_tokens,
    }
}

pub fn conservative_bytes_to_tokens(bytes: u64) -> u64 {
    // Conservative for mixed English/Chinese JSON prompts. One token per two
    // bytes overestimates most requests but keeps the budget guard cautious.
    bytes.div_ceil(2).max(1)
}

fn append_jsonl<T: Serialize>(path: &PathBuf, value: &T) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }
    let mut raw = serde_json::to_vec(value)?;
    raw.push(b'\n');
    use std::io::Write;
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .with_context(|| format!("failed to open {}", path.display()))?;
    file.write_all(&raw)
        .with_context(|| format!("failed to append {}", path.display()))
}

fn read_jsonl<T: for<'de> Deserialize<'de>>(path: &PathBuf) -> Result<Vec<T>> {
    if !path.exists() {
        return Ok(Vec::new());
    }
    let raw = std::fs::read_to_string(path)
        .with_context(|| format!("failed to read {}", path.display()))?;
    raw.lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| serde_json::from_str(line).context("invalid JSONL record"))
        .collect()
}
