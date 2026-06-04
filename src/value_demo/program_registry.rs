use std::fs;
use std::path::PathBuf;

use anyhow::{Context, Result, anyhow};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::program::Program;
use crate::value_demo::state_dir::StateDir;

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
pub struct ProgramRegistryMetadata {
    pub program_id: String,
    pub latest_version: String,
    pub versions: Vec<ProgramVersionMetadata>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
pub struct ProgramVersionMetadata {
    pub version: String,
    pub path: String,
    pub created_by: String,
    pub created_at: String,
    pub source_patch_id: Option<String>,
}

#[derive(Debug, Clone)]
pub struct ProgramRegistry {
    state_dir: StateDir,
}

impl ProgramRegistry {
    pub fn new(state_dir: StateDir) -> Self {
        Self { state_dir }
    }

    pub fn install_initial(&self, program: &Program, run_id: &str) -> Result<()> {
        self.install_version(program, "compile_run", run_id, None)
    }

    pub fn install_patched(
        &self,
        program: &Program,
        source_patch_id: &str,
        run_id: &str,
    ) -> Result<()> {
        self.install_version(program, "patch_install", run_id, Some(source_patch_id))
    }

    pub fn load_latest(&self, program_id: &str) -> Result<Program> {
        let version = self.latest_version(program_id)?;
        self.load_version(program_id, &version)
    }

    pub fn load_version(&self, program_id: &str, version: &str) -> Result<Program> {
        let metadata = self.load_metadata(program_id)?;
        let entry = metadata
            .versions
            .iter()
            .find(|entry| entry.version == version)
            .ok_or_else(|| anyhow!("program {program_id} version {version} is not registered"))?;
        let raw = fs::read(self.program_dir(program_id).join(&entry.path))
            .with_context(|| format!("failed to read program {program_id} version {version}"))?;
        Ok(serde_json::from_slice(&raw)?)
    }

    pub fn latest_version(&self, program_id: &str) -> Result<String> {
        Ok(self.load_metadata(program_id)?.latest_version)
    }

    pub fn list_programs(&self) -> Result<Vec<ProgramRegistryMetadata>> {
        let mut programs = Vec::new();
        for entry in fs::read_dir(self.state_dir.programs_dir())? {
            let entry = entry?;
            if entry.file_type()?.is_dir() {
                let program_id = entry.file_name().to_string_lossy().into_owned();
                programs.push(self.load_metadata(&program_id)?);
            }
        }
        Ok(programs)
    }

    fn install_version(
        &self,
        program: &Program,
        created_by: &str,
        run_id: &str,
        source_patch_id: Option<&str>,
    ) -> Result<()> {
        let program_dir = self.program_dir(&program.program_id);
        let versions_dir = program_dir.join("versions");
        fs::create_dir_all(&versions_dir)?;
        let file_name = format!("{}.program.json", program.version);
        fs::write(
            versions_dir.join(&file_name),
            serde_json::to_vec_pretty(program)?,
        )?;

        let mut metadata =
            self.load_metadata(&program.program_id)
                .unwrap_or_else(|_| ProgramRegistryMetadata {
                    program_id: program.program_id.clone(),
                    latest_version: program.version.clone(),
                    versions: Vec::new(),
                });
        metadata.latest_version = program.version.clone();
        let relative_path = format!("versions/{file_name}");
        if let Some(entry) = metadata
            .versions
            .iter_mut()
            .find(|entry| entry.version == program.version)
        {
            entry.path = relative_path;
            entry.created_by = created_by.to_owned();
            entry.created_at = run_id.to_owned();
            entry.source_patch_id = source_patch_id.map(str::to_owned);
        } else {
            metadata.versions.push(ProgramVersionMetadata {
                version: program.version.clone(),
                path: relative_path,
                created_by: created_by.to_owned(),
                created_at: run_id.to_owned(),
                source_patch_id: source_patch_id.map(str::to_owned),
            });
        }
        fs::write(
            self.registry_path(&program.program_id),
            serde_json::to_vec_pretty(&metadata)?,
        )?;
        Ok(())
    }

    fn load_metadata(&self, program_id: &str) -> Result<ProgramRegistryMetadata> {
        let raw = fs::read(self.registry_path(program_id))
            .with_context(|| format!("program {program_id} is not registered"))?;
        Ok(serde_json::from_slice(&raw)?)
    }

    fn program_dir(&self, program_id: &str) -> PathBuf {
        self.state_dir.programs_dir().join(program_id)
    }

    fn registry_path(&self, program_id: &str) -> PathBuf {
        self.program_dir(program_id).join("registry.json")
    }
}
