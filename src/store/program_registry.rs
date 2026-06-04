use anyhow::{Result, anyhow};
use serde::{Deserialize, Serialize};

use crate::program::Program;
use crate::store::state_dir::{
    StateDir, now_string, read_json, read_text, stable_hash_bytes, write_json_pretty, write_text,
};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct WorkflowMetadata {
    pub workflow_id: String,
    pub created_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ProgramMetadata {
    pub workflow_id: String,
    pub program_id: String,
    pub version: String,
    pub created_at: String,
    pub source: ProgramSource,
    pub parent_version: Option<String>,
    pub patch_id: Option<String>,
    pub task_hash: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ProgramSource {
    StrongCompile,
    Fixture,
    PatchInstall,
}

#[derive(Debug, Clone)]
pub struct FileProgramRegistry {
    state: StateDir,
}

impl FileProgramRegistry {
    pub fn new(state: StateDir) -> Self {
        Self { state }
    }

    pub fn init_workflow(
        &self,
        workflow_id: &str,
        mut program: Program,
        mut meta: ProgramMetadata,
    ) -> Result<()> {
        let workflow_dir = self.state.workflow_dir(workflow_id)?;
        if workflow_dir.exists() {
            return Err(anyhow!(
                "workflow {workflow_id:?} already exists; init-workflow will not overwrite existing workflow state"
            ));
        }
        self.state.ensure_workflow_layout(workflow_id)?;
        program.version = "v0001".to_owned();
        meta.workflow_id = workflow_id.to_owned();
        meta.program_id = program.program_id.clone();
        meta.version = program.version.clone();
        meta.parent_version = None;
        meta.patch_id = None;
        if meta.created_at.is_empty() {
            meta.created_at = now_string();
        }

        write_json_pretty(
            &workflow_dir.join("workflow.json"),
            &WorkflowMetadata {
                workflow_id: workflow_id.to_owned(),
                created_at: meta.created_at.clone(),
            },
        )?;
        self.write_version(workflow_id, &program, &meta)?;
        write_text(&workflow_dir.join("programs").join("latest"), "v0001\n")?;
        Ok(())
    }

    pub fn load_latest(&self, workflow_id: &str) -> Result<Program> {
        let latest = self.latest_version(workflow_id)?;
        self.load_version(workflow_id, latest.trim())
    }

    pub fn latest_version(&self, workflow_id: &str) -> Result<String> {
        read_text(
            &self
                .state
                .workflow_dir(workflow_id)?
                .join("programs")
                .join("latest"),
        )
        .map(|value| value.trim().to_owned())
    }

    pub fn load_version(&self, workflow_id: &str, version: &str) -> Result<Program> {
        read_json(
            &self
                .state
                .workflow_dir(workflow_id)?
                .join("programs")
                .join(version)
                .join("program.json"),
        )
    }

    pub fn install_version(
        &self,
        workflow_id: &str,
        mut program: Program,
        mut meta: ProgramMetadata,
    ) -> Result<String> {
        self.state.ensure_workflow_layout(workflow_id)?;
        let version = self.next_version(workflow_id)?;
        program.version = version.clone();
        meta.workflow_id = workflow_id.to_owned();
        meta.program_id = program.program_id.clone();
        meta.version = version.clone();
        if meta.created_at.is_empty() {
            meta.created_at = now_string();
        }
        self.write_version(workflow_id, &program, &meta)?;
        write_text(
            &self
                .state
                .workflow_dir(workflow_id)?
                .join("programs")
                .join("latest"),
            &format!("{version}\n"),
        )?;
        Ok(version)
    }

    pub fn list_versions(&self, workflow_id: &str) -> Result<Vec<ProgramMetadata>> {
        let programs_dir = self.state.workflow_dir(workflow_id)?.join("programs");
        let mut versions = Vec::new();
        for entry in std::fs::read_dir(&programs_dir)? {
            let entry = entry?;
            if !entry.file_type()?.is_dir() {
                continue;
            }
            let metadata_path = entry.path().join("metadata.json");
            if metadata_path.exists() {
                versions.push(read_json(&metadata_path)?);
            }
        }
        versions.sort_by(|left: &ProgramMetadata, right| left.version.cmp(&right.version));
        Ok(versions)
    }

    fn write_version(
        &self,
        workflow_id: &str,
        program: &Program,
        meta: &ProgramMetadata,
    ) -> Result<()> {
        let version_dir = self
            .state
            .workflow_dir(workflow_id)?
            .join("programs")
            .join(&program.version);
        write_json_pretty(&version_dir.join("program.json"), program)?;
        write_json_pretty(&version_dir.join("metadata.json"), meta)?;
        Ok(())
    }

    fn next_version(&self, workflow_id: &str) -> Result<String> {
        let latest = self.latest_version(workflow_id)?;
        let number = latest
            .strip_prefix('v')
            .ok_or_else(|| anyhow!("latest program version {latest} does not start with v"))?
            .parse::<u32>()?;
        Ok(format!("v{:04}", number + 1))
    }
}

pub fn fixture_program_metadata(workflow_id: &str, program: &Program) -> ProgramMetadata {
    ProgramMetadata {
        workflow_id: workflow_id.to_owned(),
        program_id: program.program_id.clone(),
        version: program.version.clone(),
        created_at: now_string(),
        source: ProgramSource::Fixture,
        parent_version: None,
        patch_id: None,
        task_hash: stable_hash_bytes(program.program_id.as_bytes()),
    }
}
