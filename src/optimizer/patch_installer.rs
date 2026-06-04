use std::sync::Arc;

use anyhow::{Context, Result};

use crate::models::EffectHandler;
use crate::optimizer::patch_evaluator::{
    evaluate_candidate, evaluate_program_on_events, patch_has_fast_path,
};
use crate::program::ProgramPatch;
use crate::store::patch_registry::{
    FilePatchRegistry, PatchEvaluationMetrics, PatchMetadata, fixture_patch_metadata,
};
use crate::store::program_registry::{FileProgramRegistry, ProgramMetadata, ProgramSource};
use crate::store::state_dir::now_string;
use crate::store::trace_store::FileTraceStore;
use crate::validator::{validate_patch, validate_patch_id};

pub async fn install_fixture_patch(
    programs: &FileProgramRegistry,
    patches: &FilePatchRegistry,
    traces: &FileTraceStore,
    workflow_id: &str,
    patch: ProgramPatch,
    weak: Arc<dyn EffectHandler>,
    strong: Arc<dyn EffectHandler>,
) -> Result<String> {
    let base_version = programs.latest_version(workflow_id)?;
    let base = programs.load_version(workflow_id, &base_version)?;
    validate_patch_id(&patch.patch_id).context("patch validation failed")?;
    let metadata = fixture_patch_metadata(
        workflow_id,
        &patch.patch_id,
        &base_version,
        &patch.rationale,
    );
    patches.record_proposed(workflow_id, patch.clone(), metadata.clone())?;

    let patched = match validate_patch(&base, &patch) {
        Ok(patched) => patched,
        Err(err) => {
            patches.mark_rejected(workflow_id, patch, metadata, &err.to_string())?;
            return Err(err).context("patch validation failed");
        }
    };
    patches.mark_validated(workflow_id, patch.clone(), metadata.clone())?;

    let sample_events = traces.list_stored_events(workflow_id, 32)?;
    if sample_events.is_empty() {
        let reason = "patch evaluation requires at least one stored stream event";
        patches.mark_rejected(workflow_id, patch, metadata, reason)?;
        anyhow::bail!(reason);
    }
    let base_evaluation = evaluate_program_on_events(
        workflow_id,
        base.clone(),
        &sample_events,
        Arc::clone(&weak),
        Arc::clone(&strong),
    )
    .await?;
    let patched_evaluation = evaluate_program_on_events(
        workflow_id,
        patched.clone(),
        &sample_events,
        Arc::clone(&weak),
        Arc::clone(&strong),
    )
    .await?;
    let evaluation = evaluate_candidate(
        &base_evaluation,
        &patched_evaluation,
        patch_has_fast_path(&patch),
        16_384,
    );
    if !evaluation.accepted {
        patches.mark_rejected(workflow_id, patch, metadata, &evaluation.reason)?;
        anyhow::bail!("patch rejected during evaluation: {}", evaluation.reason);
    }

    let mut installed_metadata = PatchMetadata {
        metrics_delta: Some(PatchEvaluationMetrics {
            base_strong_think_calls: base_evaluation.metrics.strong_think_calls,
            patched_strong_think_calls: patched_evaluation.metrics.strong_think_calls,
            base_fast_path_hits: base_evaluation.metrics.fast_path_hits,
            patched_fast_path_hits: patched_evaluation.metrics.fast_path_hits,
        }),
        ..metadata
    };
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
            task_hash: patch.patch_id.clone(),
        },
    )?;
    patches.mark_installed(
        workflow_id,
        patch,
        {
            installed_metadata.target_program_version = installed_version.clone();
            installed_metadata
        },
        &installed_version,
    )?;
    Ok(installed_version)
}
