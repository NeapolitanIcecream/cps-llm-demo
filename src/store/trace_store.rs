use std::collections::BTreeMap;

use anyhow::{Context, Result};
use serde_json::Value;

use crate::store::state_dir::StateDir;
use crate::trace::{TraceEvent, parse_trace_jsonl};

#[derive(Debug, Clone)]
pub struct FileTraceStore {
    state: StateDir,
}

impl FileTraceStore {
    pub fn new(state: StateDir) -> Self {
        Self { state }
    }

    pub fn append_events(
        &self,
        workflow_id: &str,
        run_id: &str,
        events: &[TraceEvent],
    ) -> Result<()> {
        self.state.ensure_workflow_layout(workflow_id)?;
        let path = self
            .state
            .workflow_dir(workflow_id)?
            .join("traces")
            .join(format!("{run_id}.jsonl"));
        let mut raw = String::new();
        if path.exists() {
            raw = std::fs::read_to_string(&path)
                .with_context(|| format!("failed to read {}", path.display()))?;
        }
        for event in events {
            raw.push_str(&serde_json::to_string(event)?);
            raw.push('\n');
        }
        std::fs::write(&path, raw).with_context(|| format!("failed to write {}", path.display()))
    }

    pub fn read_run(&self, workflow_id: &str, run_id: &str) -> Result<Vec<TraceEvent>> {
        let path = self
            .state
            .workflow_dir(workflow_id)?
            .join("traces")
            .join(format!("{run_id}.jsonl"));
        let raw = std::fs::read_to_string(&path)
            .with_context(|| format!("failed to read {}", path.display()))?;
        parse_trace_jsonl(&raw)
    }

    pub fn list_stored_events(&self, workflow_id: &str, limit: usize) -> Result<Vec<Value>> {
        let traces_dir = self.state.workflow_dir(workflow_id)?.join("traces");
        let mut events = BTreeMap::new();
        if !traces_dir.exists() {
            return Ok(Vec::new());
        }
        for entry in std::fs::read_dir(traces_dir)? {
            let entry = entry?;
            if !entry.file_type()?.is_file() {
                continue;
            }
            let raw = std::fs::read_to_string(entry.path())?;
            for event in parse_trace_jsonl(&raw)? {
                if event.event != "stream_event" {
                    continue;
                }
                if let Some(value) = event.detail.get("event").cloned() {
                    events.entry(event.event_id).or_insert(value);
                }
                if events.len() >= limit {
                    return Ok(events.into_values().collect());
                }
            }
        }
        Ok(events.into_values().collect())
    }
}
