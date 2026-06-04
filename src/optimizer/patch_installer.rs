use anyhow::{Context, Result};

use crate::program::ProgramPatch;
use crate::store::patch_registry::{FilePatchRegistry, PatchMetadata, fixture_patch_metadata};
use crate::store::program_registry::{FileProgramRegistry, ProgramMetadata, ProgramSource};
use crate::store::state_dir::now_string;
use crate::validator::validate_patch;

pub fn install_fixture_patch(
    programs: &FileProgramRegistry,
    patches: &FilePatchRegistry,
    workflow_id: &str,
    patch: ProgramPatch,
) -> Result<String> {
    let base_version = programs.latest_version(workflow_id)?;
    let base = programs.load_version(workflow_id, &base_version)?;
    let patched = match validate_patch(&base, &patch) {
        Ok(patched) => patched,
        Err(err) => {
            let metadata = fixture_patch_metadata(
                workflow_id,
                &patch.patch_id,
                &base_version,
                &format!("patch rejected: {err}"),
            );
            patches.mark_rejected(workflow_id, patch, metadata, &err.to_string())?;
            return Err(err).context("patch validation failed");
        }
    };
    let metadata = fixture_patch_metadata(
        workflow_id,
        &patch.patch_id,
        &base_version,
        &patch.rationale,
    );
    patches.record_proposed(workflow_id, patch.clone(), metadata.clone())?;
    let installed_version = programs.install_version(
        workflow_id,
        patched,
        ProgramMetadata {
            workflow_id: workflow_id.to_owned(),
            program_id: base.program_id,
            version: String::new(),
            created_at: now_string(),
            source: ProgramSource::PatchInstall,
            parent_version: Some(base_version),
            patch_id: Some(patch.patch_id.clone()),
            task_hash: metadata.patch_id.clone(),
        },
    )?;
    patches.mark_installed(
        workflow_id,
        patch,
        PatchMetadata {
            target_program_version: installed_version.clone(),
            ..metadata
        },
        &installed_version,
    )?;
    Ok(installed_version)
}
