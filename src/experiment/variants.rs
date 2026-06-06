use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub enum ExperimentVariant {
    StrongDirect,
    WeakOnly,
    CpsUnoptimized,
    CpsExactMemo,
    CpsGeneralizedPatch,
    CpsGeneralizedPatchNoSemanticWeak,
}

impl ExperimentVariant {
    pub fn canonical() -> Vec<Self> {
        vec![
            Self::StrongDirect,
            Self::WeakOnly,
            Self::CpsUnoptimized,
            Self::CpsExactMemo,
            Self::CpsGeneralizedPatch,
            Self::CpsGeneralizedPatchNoSemanticWeak,
        ]
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::StrongDirect => "strong_direct",
            Self::WeakOnly => "weak_only",
            Self::CpsUnoptimized => "cps_unoptimized",
            Self::CpsExactMemo => "cps_exact_memo",
            Self::CpsGeneralizedPatch => "cps_generalized_patch",
            Self::CpsGeneralizedPatchNoSemanticWeak => "cps_generalized_patch_no_semantic_weak",
        }
    }

    pub fn semantic_matcher_enabled(self) -> bool {
        !matches!(self, Self::CpsGeneralizedPatchNoSemanticWeak)
    }
}

pub fn variant_runner_runs_all_required_variants(variants: &[ExperimentVariant]) -> bool {
    ExperimentVariant::canonical()
        .into_iter()
        .all(|variant| variants.contains(&variant))
}
