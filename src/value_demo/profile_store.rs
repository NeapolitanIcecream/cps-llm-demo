use anyhow::Result;
use serde::{Deserialize, Serialize};

use crate::value_demo::state_dir::StateDir;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ProgramProfile {
    pub program_id: String,
    pub latest_version: Option<String>,
}

#[derive(Debug, Clone)]
pub struct ProfileStore {
    state_dir: StateDir,
}

impl ProfileStore {
    pub fn new(state_dir: StateDir) -> Self {
        Self { state_dir }
    }

    pub fn profile_path(&self, program_id: &str) -> std::path::PathBuf {
        self.state_dir
            .profiles_dir()
            .join(format!("{program_id}.profile.json"))
    }

    pub fn save(&self, profile: &ProgramProfile) -> Result<()> {
        std::fs::write(
            self.profile_path(&profile.program_id),
            serde_json::to_vec_pretty(profile)?,
        )?;
        Ok(())
    }
}
