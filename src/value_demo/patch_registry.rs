use std::fs;
use std::path::PathBuf;

use anyhow::{Context, Result, anyhow};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::program::ProgramPatch;
use crate::value_demo::patch_evaluator::PatchEvaluationReport;
use crate::value_demo::state_dir::StateDir;

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum PatchStatus {
    Pending,
    Validated,
    Installed,
    Rejected,
}

impl PatchStatus {
    fn dir_name(&self) -> &'static str {
        match self {
            Self::Pending => "pending",
            Self::Validated => "validated",
            Self::Installed => "installed",
            Self::Rejected => "rejected",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
pub struct PatchEnvelope {
    pub patch: ProgramPatch,
    pub status: PatchStatus,
    pub proposed_by_run_id: String,
    pub proposed_for_program_version: String,
    pub failure_fingerprint: Option<String>,
    pub evaluation: Option<PatchEvaluationReport>,
}

#[derive(Debug, Clone)]
pub struct PatchRegistry {
    state_dir: StateDir,
}

impl PatchRegistry {
    pub fn new(state_dir: StateDir) -> Self {
        Self { state_dir }
    }

    pub fn save_pending(&self, envelope: &PatchEnvelope) -> Result<()> {
        let mut pending = envelope.clone();
        pending.status = PatchStatus::Pending;
        self.write_envelope(&pending)
    }

    pub fn mark_validated(&self, patch_id: &str, report: PatchEvaluationReport) -> Result<()> {
        let mut envelope = self.load_any(patch_id)?;
        envelope.status = PatchStatus::Validated;
        envelope.evaluation = Some(report);
        self.remove_from_all_dirs(patch_id)?;
        self.write_envelope(&envelope)
    }

    pub fn mark_installed(&self, patch_id: &str, installed_version: &str) -> Result<()> {
        let mut envelope = self.load_any(patch_id)?;
        envelope.status = PatchStatus::Installed;
        if let Some(evaluation) = &mut envelope.evaluation {
            evaluation.installed_version = Some(installed_version.to_owned());
        }
        self.remove_from_all_dirs(patch_id)?;
        self.write_envelope(&envelope)
    }

    pub fn mark_rejected(&self, patch_id: &str, reason: &str) -> Result<()> {
        let mut envelope = self.load_any(patch_id)?;
        envelope.status = PatchStatus::Rejected;
        envelope.evaluation = Some(PatchEvaluationReport::rejected(patch_id, reason));
        self.remove_from_all_dirs(patch_id)?;
        self.write_envelope(&envelope)
    }

    pub fn pending_for_program(&self, program_id: &str) -> Result<Vec<PatchEnvelope>> {
        self.load_from_status(PatchStatus::Pending)
            .map(|envelopes| {
                envelopes
                    .into_iter()
                    .filter(|envelope| envelope.patch.target_program_id == program_id)
                    .collect()
            })
    }

    pub fn list(&self, status: Option<PatchStatus>) -> Result<Vec<PatchEnvelope>> {
        match status {
            Some(status) => self.load_from_status(status),
            None => {
                let mut envelopes = Vec::new();
                for status in [
                    PatchStatus::Pending,
                    PatchStatus::Validated,
                    PatchStatus::Installed,
                    PatchStatus::Rejected,
                ] {
                    envelopes.extend(self.load_from_status(status)?);
                }
                Ok(envelopes)
            }
        }
    }

    fn load_from_status(&self, status: PatchStatus) -> Result<Vec<PatchEnvelope>> {
        let dir = self.status_dir(&status);
        let mut envelopes = Vec::new();
        if !dir.exists() {
            return Ok(envelopes);
        }
        for entry in fs::read_dir(dir)? {
            let entry = entry?;
            if entry.file_type()?.is_file()
                && entry.path().extension().and_then(|ext| ext.to_str()) == Some("json")
            {
                let raw = fs::read(entry.path())?;
                envelopes.push(serde_json::from_slice(&raw)?);
            }
        }
        Ok(envelopes)
    }

    fn write_envelope(&self, envelope: &PatchEnvelope) -> Result<()> {
        let dir = self.status_dir(&envelope.status);
        fs::create_dir_all(&dir)?;
        fs::write(
            dir.join(format!("{}.patch.json", envelope.patch.patch_id)),
            serde_json::to_vec_pretty(envelope)?,
        )?;
        Ok(())
    }

    fn load_any(&self, patch_id: &str) -> Result<PatchEnvelope> {
        for status in [
            PatchStatus::Pending,
            PatchStatus::Validated,
            PatchStatus::Installed,
            PatchStatus::Rejected,
        ] {
            let path = self.patch_path(&status, patch_id);
            if path.exists() {
                let raw = fs::read(&path)
                    .with_context(|| format!("failed to read patch {}", path.display()))?;
                return Ok(serde_json::from_slice(&raw)?);
            }
        }
        Err(anyhow!("patch {patch_id} is not registered"))
    }

    fn remove_from_all_dirs(&self, patch_id: &str) -> Result<()> {
        for status in [
            PatchStatus::Pending,
            PatchStatus::Validated,
            PatchStatus::Installed,
            PatchStatus::Rejected,
        ] {
            let path = self.patch_path(&status, patch_id);
            if path.exists() {
                fs::remove_file(path)?;
            }
        }
        Ok(())
    }

    fn patch_path(&self, status: &PatchStatus, patch_id: &str) -> PathBuf {
        self.status_dir(status)
            .join(format!("{patch_id}.patch.json"))
    }

    fn status_dir(&self, status: &PatchStatus) -> PathBuf {
        self.state_dir.patches_dir().join(status.dir_name())
    }
}
