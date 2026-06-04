use std::collections::BTreeSet;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

use anyhow::{Context, Result, anyhow};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
pub struct EventEnvelope {
    pub event_id: String,

    #[serde(default)]
    pub event_type: Option<String>,

    #[serde(default)]
    pub timestamp: Option<String>,

    pub payload: Value,
}

impl EventEnvelope {
    pub fn as_program_input(&self) -> Value {
        serde_json::to_value(self).expect("EventEnvelope serialization should not fail")
    }
}

pub trait EventSource {
    fn next_event(&mut self) -> Result<Option<EventEnvelope>>;
}

pub struct JsonlEventSource<R> {
    reader: R,
    seen_event_ids: BTreeSet<String>,
    line_number: usize,
    event_count: u64,
}

impl JsonlEventSource<BufReader<File>> {
    pub fn from_path(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let file = File::open(path)
            .with_context(|| format!("failed to open event JSONL {}", path.display()))?;
        Ok(Self::new(BufReader::new(file)))
    }
}

impl<R: BufRead> JsonlEventSource<R> {
    pub fn new(reader: R) -> Self {
        Self {
            reader,
            seen_event_ids: BTreeSet::new(),
            line_number: 0,
            event_count: 0,
        }
    }

    pub fn event_count(&self) -> u64 {
        self.event_count
    }
}

impl<R: BufRead> EventSource for JsonlEventSource<R> {
    fn next_event(&mut self) -> Result<Option<EventEnvelope>> {
        let mut line = String::new();
        loop {
            line.clear();
            let bytes = self
                .reader
                .read_line(&mut line)
                .context("failed to read event JSONL line")?;
            if bytes == 0 {
                return Ok(None);
            }
            self.line_number += 1;
            if line.trim().is_empty() {
                continue;
            }

            let event: EventEnvelope = serde_json::from_str(&line).map_err(|err| {
                anyhow!("invalid event JSONL at line {}: {err}", self.line_number)
            })?;
            if !self.seen_event_ids.insert(event.event_id.clone()) {
                return Err(anyhow!(
                    "duplicate event_id {} at line {}",
                    event.event_id,
                    self.line_number
                ));
            }
            self.event_count += 1;
            return Ok(Some(event));
        }
    }
}
