use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::effects::EffectReturnMode;
use crate::optimizer::failure_fingerprint::FailureFingerprint;
use crate::program::{EffectCall, Program};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CompactEffectExample {
    pub event_ref: Option<String>,
    pub effect_frame_ref: Option<String>,
    pub shape: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PatchRequest {
    pub workflow_id: String,
    pub base_program: Program,
    pub failure_fingerprint: FailureFingerprint,
    pub compact_examples: Vec<CompactEffectExample>,
    pub allowed_patch_ops: Vec<String>,
    pub allowed_local_tools: Vec<String>,
    pub target: PatchTarget,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum PatchTarget {
    AddFastPath,
    AddValidator,
    AddProbe,
    ReduceStrongThink,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ProbeSpec {
    pub probe_id: String,
    pub effect: EffectCall,
    pub input: Value,
    pub expected_schema: Value,
    pub mode: EffectReturnMode,
}
