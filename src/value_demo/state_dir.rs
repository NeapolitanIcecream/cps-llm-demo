use std::fs;
use std::path::PathBuf;

use anyhow::{Context, Result};

#[derive(Debug, Clone)]
pub struct StateDir {
    pub root: PathBuf,
}

impl StateDir {
    pub fn open_or_create(root: impl Into<PathBuf>) -> Result<Self> {
        let state_dir = Self { root: root.into() };
        for dir in [
            state_dir.programs_dir(),
            state_dir.patches_dir().join("pending"),
            state_dir.patches_dir().join("validated"),
            state_dir.patches_dir().join("installed"),
            state_dir.patches_dir().join("rejected"),
            state_dir.traces_dir(),
            state_dir.metrics_dir(),
            state_dir.profiles_dir(),
        ] {
            fs::create_dir_all(&dir)
                .with_context(|| format!("failed to create state directory {}", dir.display()))?;
        }
        Ok(state_dir)
    }

    pub fn programs_dir(&self) -> PathBuf {
        self.root.join("programs")
    }

    pub fn patches_dir(&self) -> PathBuf {
        self.root.join("patches")
    }

    pub fn traces_dir(&self) -> PathBuf {
        self.root.join("traces")
    }

    pub fn metrics_dir(&self) -> PathBuf {
        self.root.join("metrics")
    }

    pub fn profiles_dir(&self) -> PathBuf {
        self.root.join("profiles")
    }

    pub fn value_store_dir(&self, run_id: &str) -> PathBuf {
        self.root.join("value_store").join(run_id)
    }
}
