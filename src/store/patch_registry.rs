use anyhow::{Result, bail};
use serde::{Deserialize, Serialize};

use crate::program::ProgramPatch;
use crate::store::state_dir::{
    StateDir, now_string, read_json, validate_path_component, write_json_pretty,
};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum PatchStatus {
    Proposed,
    Validated,
    Evaluated,
    Installed,
    Rejected,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum PatchSource {
    RuntimeStrongThink,
    OptimizerStrongThink,
    Fixture,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PatchEvaluationMetrics {
    pub base_strong_think_calls: u64,
    pub patched_strong_think_calls: u64,
    pub base_fast_path_hits: u64,
    pub patched_fast_path_hits: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PatchMetadata {
    pub patch_id: String,
    pub workflow_id: String,
    pub target_program_version: String,
    pub status: PatchStatus,
    pub source: PatchSource,
    pub created_at: String,
    pub evaluated_at: Option<String>,
    pub installed_at: Option<String>,
    pub rationale: String,
    pub metrics_delta: Option<PatchEvaluationMetrics>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PatchRecord {
    pub patch: ProgramPatch,
    pub metadata: PatchMetadata,
}

#[derive(Debug, Clone)]
pub struct FilePatchRegistry {
    state: StateDir,
}

impl FilePatchRegistry {
    pub fn new(state: StateDir) -> Self {
        Self { state }
    }

    pub fn record_proposed(
        &self,
        workflow_id: &str,
        patch: ProgramPatch,
        metadata: PatchMetadata,
    ) -> Result<()> {
        self.write_record(workflow_id, PatchStatus::Proposed, patch, metadata)
    }

    pub fn mark_validated(
        &self,
        workflow_id: &str,
        patch: ProgramPatch,
        mut metadata: PatchMetadata,
    ) -> Result<()> {
        metadata.status = PatchStatus::Validated;
        self.write_record(workflow_id, PatchStatus::Validated, patch, metadata)
    }

    pub fn mark_installed(
        &self,
        workflow_id: &str,
        patch: ProgramPatch,
        mut metadata: PatchMetadata,
        installed_version: &str,
    ) -> Result<()> {
        metadata.status = PatchStatus::Installed;
        metadata.installed_at = Some(now_string());
        metadata.target_program_version = installed_version.to_owned();
        self.write_record(workflow_id, PatchStatus::Installed, patch, metadata)
    }

    pub fn mark_rejected(
        &self,
        workflow_id: &str,
        patch: ProgramPatch,
        mut metadata: PatchMetadata,
        reason: &str,
    ) -> Result<()> {
        metadata.status = PatchStatus::Rejected;
        metadata.rationale = reason.to_owned();
        self.write_record(workflow_id, PatchStatus::Rejected, patch, metadata)
    }

    pub fn list_proposed(&self, workflow_id: &str) -> Result<Vec<PatchRecord>> {
        self.list_status(workflow_id, "proposed")
    }

    fn write_record(
        &self,
        workflow_id: &str,
        status: PatchStatus,
        patch: ProgramPatch,
        metadata: PatchMetadata,
    ) -> Result<()> {
        let filename = patch_record_filename(&patch.patch_id)?;
        if metadata.patch_id != patch.patch_id {
            bail!(
                "patch metadata patch_id {:?} does not match patch patch_id {:?}",
                metadata.patch_id,
                patch.patch_id
            );
        }
        self.state.ensure_workflow_layout(workflow_id)?;
        let dir = match status {
            PatchStatus::Proposed => "proposed",
            PatchStatus::Validated | PatchStatus::Evaluated => "validated",
            PatchStatus::Installed => "installed",
            PatchStatus::Rejected => "rejected",
        };
        write_json_pretty(
            &self
                .state
                .workflow_dir(workflow_id)?
                .join("patches")
                .join(dir)
                .join(filename),
            &PatchRecord { patch, metadata },
        )
    }

    fn list_status(&self, workflow_id: &str, status_dir: &str) -> Result<Vec<PatchRecord>> {
        let dir = self
            .state
            .workflow_dir(workflow_id)?
            .join("patches")
            .join(status_dir);
        let mut records = Vec::new();
        if !dir.exists() {
            return Ok(records);
        }
        for entry in std::fs::read_dir(dir)? {
            let entry = entry?;
            if entry.file_type()?.is_file() {
                records.push(read_json(&entry.path())?);
            }
        }
        Ok(records)
    }
}

fn patch_record_filename(patch_id: &str) -> Result<String> {
    validate_path_component("patch_id", patch_id)?;
    Ok(format!("{patch_id}.json"))
}

pub fn fixture_patch_metadata(
    workflow_id: &str,
    patch_id: &str,
    target_program_version: &str,
    rationale: &str,
) -> PatchMetadata {
    PatchMetadata {
        patch_id: patch_id.to_owned(),
        workflow_id: workflow_id.to_owned(),
        target_program_version: target_program_version.to_owned(),
        status: PatchStatus::Proposed,
        source: PatchSource::Fixture,
        created_at: now_string(),
        evaluated_at: None,
        installed_at: None,
        rationale: rationale.to_owned(),
        metrics_delta: None,
    }
}
