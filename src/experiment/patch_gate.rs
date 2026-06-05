use anyhow::{Result, anyhow};
use serde::{Deserialize, Serialize};

use crate::program::{GeneralizationScope, PatchGeneralizationMetadata};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PatchGateConfig {
    pub max_quality_drop: f64,
    pub max_critical_miss_delta: f64,
    pub max_false_fast_path_rate: f64,
    pub min_fast_path_hit_rate_lift: f64,
    pub min_strong_think_rate_reduction: f64,
    pub max_continuation_frame_p95_bytes: u64,
    pub require_non_exact_generalization: bool,
}

impl Default for PatchGateConfig {
    fn default() -> Self {
        Self {
            max_quality_drop: 0.02,
            max_critical_miss_delta: 0.005,
            max_false_fast_path_rate: 0.01,
            min_fast_path_hit_rate_lift: 0.10,
            min_strong_think_rate_reduction: 0.20,
            max_continuation_frame_p95_bytes: 16_384,
            require_non_exact_generalization: true,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PatchGateInput {
    pub baseline_quality: f64,
    pub candidate_quality: f64,
    pub baseline_critical_miss_rate: f64,
    pub candidate_critical_miss_rate: f64,
    pub false_fast_path_rate: f64,
    pub fast_path_hit_rate_lift: f64,
    pub strong_think_rate_reduction: f64,
    pub continuation_frame_p95_bytes: u64,
    pub metadata: PatchGeneralizationMetadata,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PatchGateDecision {
    pub accepted: bool,
    pub reasons: Vec<String>,
}

pub fn evaluate_patch_gate(config: &PatchGateConfig, input: &PatchGateInput) -> PatchGateDecision {
    let mut reasons = Vec::new();
    if input.baseline_quality - input.candidate_quality > config.max_quality_drop {
        reasons.push("quality regression exceeds gate".to_owned());
    }
    if input.candidate_critical_miss_rate - input.baseline_critical_miss_rate
        > config.max_critical_miss_delta
    {
        reasons.push("critical miss delta exceeds gate".to_owned());
    }
    if input.false_fast_path_rate > config.max_false_fast_path_rate {
        reasons.push("false fast path rate exceeds gate".to_owned());
    }
    if input.fast_path_hit_rate_lift < config.min_fast_path_hit_rate_lift {
        reasons.push("fast path hit-rate lift is too small".to_owned());
    }
    if input.strong_think_rate_reduction < config.min_strong_think_rate_reduction {
        reasons.push("strong Think reduction is too small".to_owned());
    }
    if input.continuation_frame_p95_bytes > config.max_continuation_frame_p95_bytes {
        reasons.push("continuation frame p95 exceeds gate".to_owned());
    }
    if config.require_non_exact_generalization
        && input.metadata.generalization_scope == GeneralizationScope::Exact
    {
        reasons.push("exact-only patch cannot satisfy generalized gate".to_owned());
    }

    PatchGateDecision {
        accepted: reasons.is_empty(),
        reasons,
    }
}

pub fn require_patch_gate(config: &PatchGateConfig, input: &PatchGateInput) -> Result<()> {
    let decision = evaluate_patch_gate(config, input);
    if decision.accepted {
        Ok(())
    } else {
        Err(anyhow!(
            "patch gate rejected: {}",
            decision.reasons.join("; ")
        ))
    }
}
