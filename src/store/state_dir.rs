use std::fs;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, bail};
use serde::{Serialize, de::DeserializeOwned};
use serde_json::Value;

#[derive(Debug, Clone)]
pub struct StateDir {
    root: PathBuf,
}

impl StateDir {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn workflow_dir(&self, workflow_id: &str) -> Result<PathBuf> {
        validate_path_component("workflow_id", workflow_id)?;
        Ok(self.root.join("workflows").join(workflow_id))
    }

    pub fn ensure_workflow_layout(&self, workflow_id: &str) -> Result<()> {
        let workflow = self.workflow_dir(workflow_id)?;
        for path in [
            workflow.join("programs"),
            workflow.join("patches").join("proposed"),
            workflow.join("patches").join("validated"),
            workflow.join("patches").join("installed"),
            workflow.join("patches").join("rejected"),
            workflow.join("traces"),
            workflow.join("metrics"),
            workflow.join("profiles"),
            workflow.join("values"),
            workflow.join("continuations"),
        ] {
            fs::create_dir_all(&path)
                .with_context(|| format!("failed to create {}", path.display()))?;
        }
        Ok(())
    }
}

pub fn validate_path_component(kind: &str, value: &str) -> Result<()> {
    if value.is_empty() {
        bail!("{kind} must not be empty");
    }
    if value == "." || value == ".." {
        bail!("{kind} must be a safe filename component");
    }
    if !value
        .bytes()
        .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'-'))
    {
        bail!(
            "{kind} {value:?} contains unsafe filename characters; use ASCII letters, digits, '.', '_' or '-'"
        );
    }
    Ok(())
}

pub fn read_json<T: DeserializeOwned>(path: &Path) -> Result<T> {
    let raw =
        fs::read_to_string(path).with_context(|| format!("failed to read {}", path.display()))?;
    serde_json::from_str(&raw).with_context(|| format!("invalid JSON in {}", path.display()))
}

pub fn write_json_pretty<T: Serialize>(path: &Path, value: &T) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }
    let raw = serde_json::to_vec_pretty(value)?;
    fs::write(path, raw).with_context(|| format!("failed to write {}", path.display()))
}

pub fn read_text(path: &Path) -> Result<String> {
    fs::read_to_string(path).with_context(|| format!("failed to read {}", path.display()))
}

pub fn write_text(path: &Path, value: &str) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }
    fs::write(path, value).with_context(|| format!("failed to write {}", path.display()))
}

pub fn now_string() -> String {
    let seconds = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or_default();
    format!("unix:{seconds}")
}

pub fn stable_hash_bytes(bytes: &[u8]) -> String {
    let mut hasher = StableHasher::default();
    bytes.hash(&mut hasher);
    format!("{:016x}", hasher.finish())
}

pub fn stable_hash_value(value: &Value) -> Result<String> {
    let raw = serde_json::to_vec(value)?;
    Ok(stable_hash_bytes(&raw))
}

#[derive(Default)]
struct StableHasher(u64);

impl Hasher for StableHasher {
    fn finish(&self) -> u64 {
        self.0
    }

    fn write(&mut self, bytes: &[u8]) {
        let mut hash = if self.0 == 0 {
            0xcbf29ce484222325
        } else {
            self.0
        };
        for byte in bytes {
            hash ^= u64::from(*byte);
            hash = hash.wrapping_mul(0x100000001b3);
        }
        self.0 = hash;
    }
}
