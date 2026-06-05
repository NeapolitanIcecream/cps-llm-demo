use anyhow::Result;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::program::{EffectCall, Instr, Program, ProgramFragment, ProgramPatch};

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
pub struct RuntimeBudget {
    pub max_instructions: u64,
    pub max_effects: u64,
    pub max_effect_depth: u32,
    pub max_handler_reentries: u32,
    pub max_program_fragments: u32,
    pub max_patch_attempts: u32,
}

impl Default for RuntimeBudget {
    fn default() -> Self {
        Self {
            max_instructions: 10_000,
            max_effects: 50,
            max_effect_depth: 4,
            max_handler_reentries: 3,
            max_program_fragments: 2,
            max_patch_attempts: 1,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
pub struct HandlerBudget {
    pub effect_depth: u32,
    pub effects_remaining: u64,
    pub handler_reentries_remaining: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
pub struct RuntimeFrame {
    pub function: String,
    pub pc: usize,
    pub env: Map<String, Value>,
    pub return_to: Option<ReturnSlot>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ReturnSlot {
    Call {
        caller_function: String,
        caller_pc: usize,
        var: String,
        #[serde(default)]
        expected_schema: Option<Value>,
    },
    MapElement {
        caller_function: String,
        caller_pc: usize,
        out: String,
        map_index: usize,
        item_var: String,
        function: String,
        items: Vec<Value>,
        results: Vec<Value>,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
pub struct Continuation {
    pub continuation_id: String,
    pub boundary_id: String,
    pub program_id: String,
    pub stack: Vec<RuntimeFrame>,
    pub resume_var: Option<String>,
    pub resume_pc: usize,
    pub expected_schema: Value,
    pub fuel_remaining: u64,
    pub effect_depth: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
pub struct EffectFrame {
    pub effect_id: String,
    pub boundary_id: String,
    pub reason: String,
    pub failed_effect: Option<EffectCall>,
    pub failed_instruction: Option<Instr>,
    pub continuation: Continuation,
    pub observations: Vec<Observation>,
    pub allowed_decisions: Vec<AllowedDecision>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
pub struct EncodedEffectFrame {
    pub model_visible_frame: EffectFrame,
    pub original_continuation_ref: Option<String>,
    pub encoded_bytes: usize,
    pub original_bytes: usize,
}

pub trait EffectFrameEncoder: Send + Sync {
    fn encode(&self, frame: &EffectFrame) -> Result<EncodedEffectFrame>;
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
pub struct Observation {
    pub name: String,
    pub value: Value,
    pub source: ObservationSource,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ObservationSource {
    WeakModel,
    StrongModel,
    LocalTool,
    Runtime,
}

impl ObservationSource {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::WeakModel => "weak_model",
            Self::StrongModel => "strong_model",
            Self::LocalTool => "local_tool",
            Self::Runtime => "runtime",
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum AllowedDecision {
    ReturnValue,
    RequestEffect,
    ReturnProgramFragment,
    ReturnProgramPatch,
    Abort,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
#[serde(tag = "decision", rename_all = "snake_case")]
pub enum HandlerDecision {
    ReturnValue {
        value: Value,
        confidence: f32,
        rationale: String,
    },
    RequestEffect {
        effect: EffectCall,
        input: Value,
        expected_schema: Value,
        mode: EffectReturnMode,
        rationale: String,
    },
    ReturnProgram {
        program: Program,
        rationale: String,
    },
    ReturnProgramFragment {
        fragment: ProgramFragment,
        rationale: String,
    },
    ReturnProgramPatch {
        patch: ProgramPatch,
        rationale: String,
    },
    Abort {
        reason: String,
    },
}

impl HandlerDecision {
    pub fn decision_name(&self) -> &'static str {
        match self {
            Self::ReturnValue { .. } => "return_value",
            Self::RequestEffect { .. } => "request_effect",
            Self::ReturnProgram { .. } => "return_program",
            Self::ReturnProgramFragment { .. } => "return_program_fragment",
            Self::ReturnProgramPatch { .. } => "return_program_patch",
            Self::Abort { .. } => "abort",
        }
    }

    pub fn confidence(&self) -> Option<f32> {
        match self {
            Self::ReturnValue { confidence, .. } => Some(*confidence),
            Self::RequestEffect { .. }
            | Self::ReturnProgram { .. }
            | Self::ReturnProgramFragment { .. }
            | Self::ReturnProgramPatch { .. }
            | Self::Abort { .. } => None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
#[serde(tag = "mode", rename_all = "snake_case")]
pub enum EffectReturnMode {
    UseAsValue,
    ReenterHandler { observation_name: String },
}

impl EffectReturnMode {
    pub fn mode_name(&self) -> &'static str {
        match self {
            Self::UseAsValue => "use_as_value",
            Self::ReenterHandler { .. } => "reenter_handler",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
pub struct HandlerRequest {
    #[serde(skip)]
    #[schemars(skip)]
    pub run_id: Option<String>,
    #[serde(skip)]
    #[schemars(skip)]
    pub budget_scope_id: Option<String>,
    #[serde(skip)]
    #[schemars(skip)]
    pub workflow_id: Option<String>,
    #[serde(skip)]
    #[schemars(skip)]
    pub phase: Option<String>,
    pub effect: EffectCall,
    pub input: Value,
    pub expected_schema: Value,
    pub continuation_summary: Option<ContinuationSummary>,
    pub effect_frame: Option<EffectFrame>,
    pub observations: Vec<Observation>,
    pub budget: HandlerBudget,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
pub struct ContinuationSummary {
    pub boundary_id: String,
    pub program_id: String,
    pub continuation_id: String,
    pub current_function: Option<String>,
    pub resume_pc: usize,
    pub resume_var: Option<String>,
    pub stack_depth: usize,
    pub expected_schema: Value,
}
