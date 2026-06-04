use std::path::Path;

use anyhow::{Context, Result};
use serde_json::Value;

pub trait EventSource {
    fn next_event(&mut self) -> Result<Option<Value>>;
}

pub struct JsonlEventSource {
    events: std::vec::IntoIter<Value>,
}

impl JsonlEventSource {
    pub fn from_path(path: &Path) -> Result<Self> {
        let raw = std::fs::read_to_string(path)
            .with_context(|| format!("failed to read events JSONL {}", path.display()))?;
        let events = raw
            .lines()
            .enumerate()
            .filter(|(_, line)| !line.trim().is_empty())
            .map(|(index, line)| {
                serde_json::from_str::<Value>(line)
                    .with_context(|| format!("invalid event JSON at line {}", index + 1))
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            events: events.into_iter(),
        })
    }
}

impl EventSource for JsonlEventSource {
    fn next_event(&mut self) -> Result<Option<Value>> {
        Ok(self.events.next())
    }
}

pub struct InMemoryEventSource {
    events: std::vec::IntoIter<Value>,
}

impl InMemoryEventSource {
    pub fn new(events: Vec<Value>) -> Self {
        Self {
            events: events.into_iter(),
        }
    }
}

impl EventSource for InMemoryEventSource {
    fn next_event(&mut self) -> Result<Option<Value>> {
        Ok(self.events.next())
    }
}
