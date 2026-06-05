use crate::program::{
    EffectCall, GeneralizationScope, Instr, ModelStrength, PatchGeneralizationMetadata, PatchKind,
    PatchOp, ProgramPatch,
};

pub const SEMANTIC_FAST_PATH_TASK: &str = "semantic_fast_path_match";

pub fn semantic_fast_path_patch_uses_weak_matcher_not_exact_text(patch: &ProgramPatch) -> bool {
    let Some(metadata) = patch.generalization.as_ref() else {
        return false;
    };
    metadata.uses_weak_semantic_matcher
        && metadata.generalization_scope == GeneralizationScope::SemanticClass
        && contains_weak_semantic_matcher(patch)
}

pub fn generalized_semantic_metadata(
    positive_clusters: Vec<String>,
    negative_clusters: Vec<String>,
) -> PatchGeneralizationMetadata {
    PatchGeneralizationMetadata {
        patch_kind: PatchKind::WeakSemanticFastPath,
        generalization_scope: GeneralizationScope::SemanticClass,
        uses_weak_semantic_matcher: true,
        uses_deterministic_fast_path: false,
        uses_validator: true,
        declared_positive_clusters: positive_clusters,
        declared_negative_clusters: negative_clusters,
    }
}

fn contains_weak_semantic_matcher(patch: &ProgramPatch) -> bool {
    patch.operations.iter().any(|operation| match operation {
        PatchOp::ReplaceInstruction { instr, .. } | PatchOp::InsertInstruction { instr, .. } => {
            instruction_is_weak_semantic_matcher(instr)
        }
        PatchOp::AddFunction { function, .. } => function
            .body
            .iter()
            .any(instruction_is_weak_semantic_matcher),
        PatchOp::AddEffectPermission { .. } => false,
        PatchOp::UpdateAcceptancePolicy { .. } => false,
    })
}

fn instruction_is_weak_semantic_matcher(instr: &Instr) -> bool {
    matches!(
        instr,
        Instr::Perform {
            effect: EffectCall::ModelTask {
                strength: ModelStrength::Weak,
                task,
            },
            ..
        } if task.name == SEMANTIC_FAST_PATH_TASK
    )
}
