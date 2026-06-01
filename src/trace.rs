use std::sync::{Arc, Mutex};

use serde::Serialize;
use serde_json::Value;

#[derive(Debug, Clone, Serialize, PartialEq)]
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
