use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::trace::TraceEvent;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PredictionRecord {
    pub event_id: String,
    pub variant: String,
    pub program_version: String,
    pub output: Value,
    pub schema_valid: bool,
    pub trace_run_id: Option<String>,
    pub fast_path: FastPathPredictionMetadata,
    pub model_calls: ModelCallCounts,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FastPathPredictionMetadata {
    pub hit: bool,
    pub rule_id: Option<String>,
    pub kind: Option<String>,
}

impl FastPathPredictionMetadata {
    pub fn miss() -> Self {
        Self {
            hit: false,
            rule_id: None,
            kind: None,
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct ModelCallCounts {
    pub weak: u64,
    pub strong_think: u64,
    pub strong_task: u64,
}

#[derive(Debug, Clone)]
pub struct PredictionStore {
    dir: PathBuf,
}

impl PredictionStore {
    pub fn new(dir: impl Into<PathBuf>) -> Self {
        Self { dir: dir.into() }
    }

    pub fn append(&self, file_name: &str, record: &PredictionRecord) -> Result<()> {
        let path = self.dir.join(file_name);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("failed to create {}", parent.display()))?;
        }
        let mut raw = serde_json::to_vec(record)?;
        raw.push(b'\n');
        use std::io::Write;
        let mut file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .with_context(|| format!("failed to open {}", path.display()))?;
        file.write_all(&raw)
            .with_context(|| format!("failed to append {}", path.display()))
    }

    pub fn read(path: &Path) -> Result<Vec<PredictionRecord>> {
        read_jsonl(path)
    }
}

pub fn prediction_metadata_from_trace(
    events: &[TraceEvent],
) -> (FastPathPredictionMetadata, ModelCallCounts) {
    let mut fast_path = FastPathPredictionMetadata::miss();
    let mut calls = ModelCallCounts::default();
    for event in events {
        match event.event.as_str() {
            "fast_path_hit" => {
                fast_path.hit = true;
                fast_path.rule_id = event
                    .detail
                    .get("rule_id")
                    .and_then(Value::as_str)
                    .map(ToOwned::to_owned);
                fast_path.kind = Some("weak_semantic_fast_path".to_owned());
            }
            "handler_request" => {
                let handler = event.detail.get("handler").and_then(Value::as_str);
                let effect = event.detail.get("effect").and_then(Value::as_str);
                match (handler, effect) {
                    (Some("weak_model"), Some("model_task")) => calls.weak += 1,
                    (Some("strong_model"), Some("think")) => calls.strong_think += 1,
                    (Some("strong_model"), Some("model_task")) => calls.strong_task += 1,
                    _ => {}
                }
            }
            _ => {}
        }
    }
    (fast_path, calls)
}

fn read_jsonl<T: for<'de> Deserialize<'de>>(path: &Path) -> Result<Vec<T>> {
    if !path.exists() {
        return Ok(Vec::new());
    }
    let raw = std::fs::read_to_string(path)
        .with_context(|| format!("failed to read {}", path.display()))?;
    raw.lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| serde_json::from_str(line).context("invalid prediction JSONL record"))
        .collect()
}
