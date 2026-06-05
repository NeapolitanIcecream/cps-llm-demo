use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::store::model_call_store::ModelUsage;
use crate::store::state_dir::{now_string, stable_hash_bytes, stable_hash_value};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelCacheMode {
    ReadWrite,
    Disabled,
    Refresh,
    ReadOnly,
}

#[derive(Debug, Clone)]
pub struct ModelCache {
    dir: PathBuf,
    mode: ModelCacheMode,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ModelCacheEntry {
    pub key: String,
    pub model: String,
    pub prompt_hash: String,
    pub schema_hash: String,
    pub model_config_hash: String,
    pub parsed_output: Value,
    pub raw_response: Value,
    pub usage: Option<ModelUsage>,
    pub created_at: String,
}

impl ModelCache {
    pub fn new(dir: impl Into<PathBuf>, mode: ModelCacheMode) -> Self {
        Self {
            dir: dir.into(),
            mode,
        }
    }

    pub fn mode(&self) -> ModelCacheMode {
        self.mode
    }

    pub fn get(&self, key: &str) -> Result<Option<ModelCacheEntry>> {
        if matches!(
            self.mode,
            ModelCacheMode::Disabled | ModelCacheMode::Refresh
        ) {
            return Ok(None);
        }
        let path = self.entry_path(key);
        if !path.exists() {
            if self.mode == ModelCacheMode::ReadOnly {
                return Err(anyhow!("model cache miss for key {key} in read-only mode"));
            }
            return Ok(None);
        }
        let raw = std::fs::read_to_string(&path)
            .with_context(|| format!("failed to read {}", path.display()))?;
        let entry = serde_json::from_str(&raw)
            .with_context(|| format!("invalid model cache entry {}", path.display()))?;
        Ok(Some(entry))
    }

    pub fn put(&self, entry: &ModelCacheEntry) -> Result<()> {
        if !matches!(
            self.mode,
            ModelCacheMode::ReadWrite | ModelCacheMode::Refresh
        ) {
            return Ok(());
        }
        let path = self.entry_path(&entry.key);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("failed to create {}", parent.display()))?;
        }
        std::fs::write(&path, serde_json::to_vec_pretty(entry)?)
            .with_context(|| format!("failed to write {}", path.display()))
    }

    fn entry_path(&self, key: &str) -> PathBuf {
        self.dir.join(format!("{key}.json"))
    }
}

pub fn structured_cache_key(
    provider_base_url: &str,
    model: &str,
    instructions: &str,
    input_json: &Value,
    output_schema: &Value,
    request_options_that_affect_output: &Value,
) -> Result<String> {
    stable_hash_value(&json!({
        "provider_base_url": provider_base_url,
        "model": model,
        "instructions": instructions,
        "input_json": input_json,
        "output_schema": output_schema,
        "request_options_that_affect_output": request_options_that_affect_output,
    }))
}

pub fn prompt_hash(instructions: &str) -> String {
    stable_hash_bytes(instructions.as_bytes())
}

pub fn schema_hash(schema: &Value) -> Result<String> {
    stable_hash_value(schema)
}

pub fn model_config_hash(base_url: &str, model: &str) -> Result<String> {
    stable_hash_value(&json!({
        "base_url": base_url,
        "model": model,
        "structured_outputs": "text.format.json_schema.strict",
    }))
}

#[derive(Debug, Clone)]
pub struct ModelCacheEntryParts {
    pub key: String,
    pub model: String,
    pub prompt_hash: String,
    pub schema_hash: String,
    pub model_config_hash: String,
    pub parsed_output: Value,
    pub raw_response: Value,
    pub usage: Option<ModelUsage>,
}

pub fn cache_entry_from_response(parts: ModelCacheEntryParts) -> ModelCacheEntry {
    ModelCacheEntry {
        key: parts.key,
        model: parts.model,
        prompt_hash: parts.prompt_hash,
        schema_hash: parts.schema_hash,
        model_config_hash: parts.model_config_hash,
        parsed_output: parts.parsed_output,
        raw_response: parts.raw_response,
        usage: parts.usage,
        created_at: now_string(),
    }
}

pub fn cache_mode_from_flags(
    no_cache: bool,
    refresh_cache: bool,
    cache_readonly: bool,
) -> Result<ModelCacheMode> {
    let enabled = [no_cache, refresh_cache, cache_readonly]
        .into_iter()
        .filter(|value| *value)
        .count();
    if enabled > 1 {
        return Err(anyhow!(
            "--no-cache, --refresh-cache, and --cache-readonly are mutually exclusive"
        ));
    }
    Ok(if no_cache {
        ModelCacheMode::Disabled
    } else if refresh_cache {
        ModelCacheMode::Refresh
    } else if cache_readonly {
        ModelCacheMode::ReadOnly
    } else {
        ModelCacheMode::ReadWrite
    })
}

pub fn ensure_cache_dir(path: &Path) -> Result<()> {
    std::fs::create_dir_all(path).with_context(|| format!("failed to create {}", path.display()))
}
