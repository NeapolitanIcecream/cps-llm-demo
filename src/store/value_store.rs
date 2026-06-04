use anyhow::Result;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::store::state_dir::{StateDir, read_json, stable_hash_value, write_json_pretty};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ValueRef {
    pub uri: String,
    pub schema_hash: String,
    pub byte_len: usize,
}

#[derive(Debug, Clone)]
pub struct FileValueStore {
    state: StateDir,
    workflow_id: String,
}

impl FileValueStore {
    pub fn new(state: StateDir, workflow_id: impl Into<String>) -> Self {
        Self {
            state,
            workflow_id: workflow_id.into(),
        }
    }

    pub fn put(&self, value: &Value) -> Result<ValueRef> {
        self.state.ensure_workflow_layout(&self.workflow_id)?;
        let raw = serde_json::to_vec(value)?;
        let hash = stable_hash_value(value)?;
        let reference = ValueRef {
            uri: format!("valuestore://{hash}"),
            schema_hash: hash.clone(),
            byte_len: raw.len(),
        };
        write_json_pretty(
            &self
                .state
                .workflow_dir(&self.workflow_id)?
                .join("values")
                .join(format!("{hash}.json")),
            value,
        )?;
        Ok(reference)
    }

    pub fn get(&self, reference: &ValueRef) -> Result<Value> {
        let hash = reference
            .uri
            .strip_prefix("valuestore://")
            .unwrap_or(&reference.schema_hash);
        read_json(
            &self
                .state
                .workflow_dir(&self.workflow_id)?
                .join("values")
                .join(format!("{hash}.json")),
        )
    }

    pub fn placeholder(reference: &ValueRef) -> Value {
        json!({
            "$value_ref": reference.uri,
            "schema_hash": reference.schema_hash,
            "byte_len": reference.byte_len,
        })
    }
}
