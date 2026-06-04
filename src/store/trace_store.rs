use anyhow::{Context, Result};

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
            .workflow_dir(workflow_id)
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
            .workflow_dir(workflow_id)
            .join("traces")
            .join(format!("{run_id}.jsonl"));
        let raw = std::fs::read_to_string(&path)
            .with_context(|| format!("failed to read {}", path.display()))?;
        parse_trace_jsonl(&raw)
    }
}
