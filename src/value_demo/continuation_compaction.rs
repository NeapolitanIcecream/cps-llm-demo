use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;

use anyhow::{Context, Result, anyhow};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};
use sha2::{Digest, Sha256};

use crate::effects::{Continuation, EffectFrame, ReturnSlot, RuntimeFrame};

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
pub struct ValueRef {
    pub uri: String,
    pub schema_hash: String,
    pub byte_len: u64,
}

#[derive(Debug, Clone)]
pub struct ValueStore {
    run_id: String,
    root: PathBuf,
}

impl ValueStore {
    pub fn new(run_id: impl Into<String>, root: impl Into<PathBuf>) -> Result<Self> {
        let store = Self {
            run_id: run_id.into(),
            root: root.into(),
        };
        fs::create_dir_all(store.values_dir()).with_context(|| {
            format!(
                "failed to create value store values directory {}",
                store.values_dir().display()
            )
        })?;
        Ok(store)
    }

    pub fn put(&self, value: &Value) -> Result<ValueRef> {
        let bytes = serde_json::to_vec(value)?;
        let hash = sha256_hex(&bytes);
        let file_name = format!("v{hash}.json");
        let path = self.values_dir().join(&file_name);
        if !path.exists() {
            fs::write(&path, &bytes)
                .with_context(|| format!("failed to write value ref {}", path.display()))?;
        }
        self.write_index_entry(&file_name)?;
        Ok(ValueRef {
            uri: format!("valuestore://{}/values/{file_name}", self.run_id),
            schema_hash: format!("sha256:{hash}"),
            byte_len: bytes.len() as u64,
        })
    }

    pub fn get(&self, value_ref: &ValueRef) -> Result<Value> {
        let file_name = value_ref
            .uri
            .rsplit('/')
            .next()
            .ok_or_else(|| anyhow!("invalid value ref URI {}", value_ref.uri))?;
        let path = self.values_dir().join(file_name);
        let raw = fs::read(&path)
            .with_context(|| format!("failed to read value ref {}", path.display()))?;
        Ok(serde_json::from_slice(&raw)?)
    }

    fn values_dir(&self) -> PathBuf {
        self.root.join("values")
    }

    fn write_index_entry(&self, file_name: &str) -> Result<()> {
        let index_path = self.root.join("index.json");
        let mut entries = if index_path.exists() {
            serde_json::from_slice::<Vec<String>>(&fs::read(&index_path)?)?
        } else {
            Vec::new()
        };
        if !entries.iter().any(|entry| entry == file_name) {
            entries.push(file_name.to_owned());
            fs::write(index_path, serde_json::to_vec_pretty(&entries)?)?;
        }
        Ok(())
    }
}

#[derive(Debug, Default, Clone)]
pub struct ContinuationStore {
    map: BTreeMap<String, Continuation>,
}

impl ContinuationStore {
    pub fn insert(&mut self, continuation: Continuation) {
        self.map
            .insert(continuation.continuation_id.clone(), continuation);
    }

    pub fn get(&self, continuation_id: &str) -> Option<Continuation> {
        self.map.get(continuation_id).cloned()
    }

    pub fn remove(&mut self, continuation_id: &str) -> Option<Continuation> {
        self.map.remove(continuation_id)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
pub struct ContinuationCompactionConfig {
    pub max_public_continuation_frame_bytes: usize,
    pub max_inline_value_bytes: usize,
}

impl Default for ContinuationCompactionConfig {
    fn default() -> Self {
        Self {
            max_public_continuation_frame_bytes: 16 * 1024,
            max_inline_value_bytes: 2 * 1024,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
pub struct ContinuationCompactionReport {
    pub continuation_id: String,
    pub full_bytes: u64,
    pub public_bytes: u64,
    pub stored_refs: u64,
    pub truncated: bool,
}

pub fn compact_effect_frame(
    frame: &EffectFrame,
    config: &ContinuationCompactionConfig,
    value_store: Option<&ValueStore>,
) -> Result<(EffectFrame, ContinuationCompactionReport)> {
    let full_bytes = serialized_len(frame)?;
    let mut stored_refs = 0;
    let mut compacted = frame.clone();
    compacted.continuation =
        compact_continuation(&frame.continuation, config, value_store, &mut stored_refs)?;
    let mut truncated = false;

    if serialized_len(&compacted)? > config.max_public_continuation_frame_bytes {
        compacted.continuation.stack = compacted
            .continuation
            .stack
            .into_iter()
            .map(truncate_frame_env)
            .collect();
        truncated = true;
    }

    let public_bytes = serialized_len(&compacted)?;
    Ok((
        compacted,
        ContinuationCompactionReport {
            continuation_id: frame.continuation.continuation_id.clone(),
            full_bytes: full_bytes as u64,
            public_bytes: public_bytes as u64,
            stored_refs,
            truncated,
        },
    ))
}

fn compact_continuation(
    continuation: &Continuation,
    config: &ContinuationCompactionConfig,
    value_store: Option<&ValueStore>,
    stored_refs: &mut u64,
) -> Result<Continuation> {
    let mut compacted = continuation.clone();
    compacted.stack = continuation
        .stack
        .iter()
        .map(|frame| compact_frame(frame, config, value_store, stored_refs))
        .collect::<Result<Vec<_>>>()?;
    Ok(compacted)
}

fn compact_frame(
    frame: &RuntimeFrame,
    config: &ContinuationCompactionConfig,
    value_store: Option<&ValueStore>,
    stored_refs: &mut u64,
) -> Result<RuntimeFrame> {
    let mut compacted = frame.clone();
    let mut env = Map::new();
    for (name, value) in &frame.env {
        env.insert(
            name.clone(),
            compact_value(
                value,
                config.max_inline_value_bytes,
                value_store,
                stored_refs,
            )?,
        );
    }
    compacted.env = env;
    compacted.return_to = match &frame.return_to {
        Some(ReturnSlot::MapElement {
            caller_function,
            caller_pc,
            out,
            map_index,
            item_var,
            function,
            items,
            results,
        }) => Some(ReturnSlot::MapElement {
            caller_function: caller_function.clone(),
            caller_pc: *caller_pc,
            out: out.clone(),
            map_index: *map_index,
            item_var: item_var.clone(),
            function: function.clone(),
            items: compact_values(
                items,
                config.max_inline_value_bytes,
                value_store,
                stored_refs,
            )?,
            results: compact_values(
                results,
                config.max_inline_value_bytes,
                value_store,
                stored_refs,
            )?,
        }),
        other => other.clone(),
    };
    Ok(compacted)
}

fn compact_values(
    values: &[Value],
    max_inline_value_bytes: usize,
    value_store: Option<&ValueStore>,
    stored_refs: &mut u64,
) -> Result<Vec<Value>> {
    values
        .iter()
        .map(|value| compact_value(value, max_inline_value_bytes, value_store, stored_refs))
        .collect()
}

fn compact_value(
    value: &Value,
    max_inline_value_bytes: usize,
    value_store: Option<&ValueStore>,
    stored_refs: &mut u64,
) -> Result<Value> {
    let byte_len = serialized_len(value)?;
    if byte_len <= max_inline_value_bytes {
        return Ok(value.clone());
    }
    if let Some(value_store) = value_store {
        *stored_refs += 1;
        return Ok(serde_json::to_value(value_store.put(value)?)?);
    }
    Ok(json!({
        "$summary": "large_value_omitted",
        "byte_len": byte_len,
    }))
}

fn truncate_frame_env(mut frame: RuntimeFrame) -> RuntimeFrame {
    let keys = frame.env.keys().cloned().collect::<Vec<_>>();
    let mut env = Map::new();
    env.insert(
        "$summary".to_owned(),
        json!({
            "kind": "env_omitted",
            "keys": keys,
        }),
    );
    frame.env = env;
    frame
}

fn serialized_len(value: &impl Serialize) -> Result<usize> {
    Ok(serde_json::to_vec(value)?.len())
}

fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}
