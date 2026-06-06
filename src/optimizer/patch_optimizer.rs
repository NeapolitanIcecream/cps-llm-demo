use std::sync::Arc;

use anyhow::{Result, anyhow};
use serde_json::json;

use crate::effects::{HandlerBudget, HandlerDecision, HandlerRequest, RuntimeBudget};
use crate::local_tools::builtin_local_tool_names;
use crate::models::EffectHandler;
use crate::optimizer::patch_installer::install_fixture_patch;
use crate::optimizer::patch_request::{CompactEffectExample, PatchRequest, PatchTarget};
use crate::program::{EffectCall, ModelStrength, ModelTaskSpec};
use crate::store::patch_registry::FilePatchRegistry;
use crate::store::profile_store::{FailureFingerprintStats, FileProfileStore};
use crate::store::program_registry::FileProgramRegistry;
use crate::store::trace_store::FileTraceStore;

pub struct OptimizerContext<'a> {
    pub programs: &'a FileProgramRegistry,
    pub patches: &'a FilePatchRegistry,
    pub traces: &'a FileTraceStore,
    pub profiles: &'a FileProfileStore,
    pub weak: Arc<dyn EffectHandler>,
    pub strong: Arc<dyn EffectHandler>,
}

pub async fn optimize_from_profile(
    context: OptimizerContext<'_>,
    workflow_id: &str,
    max_patches: usize,
) -> Result<String> {
    let latest = context.programs.load_latest(workflow_id)?;
    let profile = context.profiles.load(workflow_id)?;
    let latest_program_id = latest.program_id.as_str();
    let latest_program_version = latest.version.as_str();
    let mut failures = profile
        .failure_fingerprints
        .values()
        .filter(|stats| {
            stats.fingerprint.program_id == latest_program_id
                && stats.fingerprint.program_version == latest_program_version
        })
        .cloned()
        .collect::<Vec<_>>();
    failures.sort_by(|left, right| {
        right.count.cmp(&left.count).then_with(|| {
            left.fingerprint
                .fingerprint_id
                .cmp(&right.fingerprint.fingerprint_id)
        })
    });
    if failures.is_empty() {
        return Err(anyhow!(
            "optimizer found no failure fingerprints for program {} version {}",
            latest.program_id,
            latest.version
        ));
    }

    let mut last_error = None;
    for failure in failures.into_iter().take(max_patches) {
        let request = build_patch_request(workflow_id, latest.clone(), &failure);
        let decision = context
            .strong
            .handle(HandlerRequest {
                run_id: None,
                budget_scope_id: None,
                workflow_id: Some(workflow_id.to_owned()),
                phase: Some("optimize".to_owned()),
                effect: EffectCall::ModelTask {
                    strength: ModelStrength::Strong,
                    task: ModelTaskSpec {
                        name: "optimize_program_patch".to_owned(),
                        instructions: STRONG_OPTIMIZER_INSTRUCTIONS.to_owned(),
                    },
                },
                input: serde_json::to_value(request)?,
                expected_schema: json!({ "type": "null" }),
                continuation_summary: None,
                effect_frame: None,
                observations: Vec::new(),
                budget: HandlerBudget {
                    effect_depth: 0,
                    effects_remaining: RuntimeBudget::default().max_effects,
                    handler_reentries_remaining: RuntimeBudget::default().max_handler_reentries,
                },
            })
            .await?;

        match decision {
            HandlerDecision::ReturnProgramPatch { patch, .. } => {
                match install_fixture_patch(
                    context.programs,
                    context.patches,
                    context.traces,
                    workflow_id,
                    patch,
                    Arc::clone(&context.weak),
                    Arc::clone(&context.strong),
                )
                .await
                {
                    Ok(version) => return Ok(version),
                    Err(err) => last_error = Some(err),
                }
            }
            HandlerDecision::Abort { reason } => {
                last_error = Some(anyhow!("strong optimizer aborted: {reason}"));
            }
            other => {
                last_error = Some(anyhow!(
                    "strong optimizer returned {}, expected return_program_patch",
                    other.decision_name()
                ));
            }
        }
    }

    Err(last_error.unwrap_or_else(|| anyhow!("optimizer did not install a patch")))
}

fn build_patch_request(
    workflow_id: &str,
    base_program: crate::program::Program,
    failure: &FailureFingerprintStats,
) -> PatchRequest {
    PatchRequest {
        workflow_id: workflow_id.to_owned(),
        base_program,
        failure_fingerprint: failure.fingerprint.clone(),
        compact_examples: failure
            .sample_event_refs
            .iter()
            .map(|event_ref| CompactEffectExample {
                event_ref: Some(event_ref.clone()),
                effect_frame_ref: failure.sample_continuation_refs.first().cloned(),
                shape: json!({
                    "fingerprint_id": failure.fingerprint.fingerprint_id,
                    "count": failure.count,
                }),
            })
            .collect(),
        allowed_patch_ops: vec![
            "add_function".to_owned(),
            "insert_instruction".to_owned(),
            "replace_instruction".to_owned(),
            "update_acceptance_policy".to_owned(),
        ],
        allowed_local_tools: builtin_local_tool_names()
            .iter()
            .map(ToString::to_string)
            .collect(),
        target: PatchTarget::AddFastPath,
    }
}

const STRONG_OPTIMIZER_INSTRUCTIONS: &str = r#"You are optimizing a typed Program IR using profile data.
Do not return natural language.
Return only return_program_patch or abort.
Patch must be domain-specific only through Program IR data, not Rust code.
Prefer:
- fast_path_apply for repeated deterministic cases;
- validator_apply for repeated schema or quality failures;
- weak/local probes before StrongThink when they reduce expensive calls.
Do not assume core runtime has business-specific branches."#;
