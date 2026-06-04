use std::sync::{Arc, Mutex};

use anyhow::{Result, anyhow};
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TraceEvent {
    pub event: String,
    pub event_id: String,
    pub detail: Value,
}

#[derive(Clone, Default)]
pub struct TraceCollector {
    inner: Arc<Mutex<Vec<TraceEvent>>>,
}

impl TraceCollector {
    pub fn emit(&self, event: impl Into<String>, event_id: impl Into<String>, detail: Value) {
        self.inner
            .lock()
            .expect("trace mutex poisoned")
            .push(TraceEvent {
                event: event.into(),
                event_id: event_id.into(),
                detail,
            });
    }

    pub fn events(&self) -> Vec<TraceEvent> {
        self.inner.lock().expect("trace mutex poisoned").clone()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReplayReport {
    pub captures: usize,
    pub resumes: usize,
    pub aborts: usize,
}

pub fn parse_trace_jsonl(raw: &str) -> Result<Vec<TraceEvent>> {
    raw.lines()
        .enumerate()
        .filter(|(_, line)| !line.trim().is_empty())
        .map(|(index, line)| {
            serde_json::from_str::<TraceEvent>(line)
                .map_err(|err| anyhow!("invalid trace JSONL at line {}: {err}", index + 1))
        })
        .collect()
}

pub fn replay_trace_events(events: &[TraceEvent]) -> Result<ReplayReport> {
    let mut captures = std::collections::BTreeSet::new();
    let mut resumes = std::collections::BTreeSet::new();
    let mut aborts = 0;

    for event in events {
        match event.event.as_str() {
            "capture_continuation" => {
                let continuation_id = trace_continuation_id(event)?;
                if !captures.insert(continuation_id.to_owned()) {
                    return Err(anyhow!(
                        "continuation {continuation_id} was captured more than once"
                    ));
                }
            }
            "resume_continuation" => {
                let continuation_id = trace_continuation_id(event)?;
                if !captures.contains(continuation_id) {
                    return Err(anyhow!(
                        "resume_continuation references unknown continuation {continuation_id}"
                    ));
                }
                if !resumes.insert(continuation_id.to_owned()) {
                    return Err(anyhow!(
                        "continuation {continuation_id} was resumed more than once"
                    ));
                }
            }
            "program_aborted" => {
                aborts += 1;
            }
            _ => {}
        }
    }

    for continuation_id in &captures {
        if !resumes.contains(continuation_id) && aborts == 0 {
            return Err(anyhow!(
                "captured continuation {continuation_id} has no resume or abort"
            ));
        }
    }

    Ok(ReplayReport {
        captures: captures.len(),
        resumes: resumes.len(),
        aborts,
    })
}

fn trace_continuation_id(event: &TraceEvent) -> Result<&str> {
    event
        .detail
        .get("continuation_id")
        .and_then(Value::as_str)
        .ok_or_else(|| anyhow!("{} event is missing continuation_id", event.event))
}
