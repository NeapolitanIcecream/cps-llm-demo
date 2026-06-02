use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::program::{Instr, JsonExpr, WeakTaskSpec};

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
pub struct Continuation {
    pub program_id: String,
    pub pc: usize,
    pub resume_var: Option<String>,
    pub env: Map<String, Value>,
    pub expected_schema: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
pub struct EffectFrame {
    pub effect_id: String,
    pub reason: String,
    pub failed_instruction: Option<Instr>,
    pub continuation: Continuation,
    pub observations: Vec<Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
#[serde(tag = "decision", rename_all = "snake_case")]
pub enum ThinkDecision {
    ResumeWithValue {
        value: Value,
        confidence: f32,
        rationale: String,
    },
    RequestWeakProbe {
        out: String,
        task: WeakTaskSpec,
        input: JsonExpr,
        output_schema: Value,
        min_confidence: f32,
        rationale: String,
    },
    Abort {
        reason: String,
    },
}

impl ThinkDecision {
    pub fn decision_name(&self) -> &'static str {
        match self {
            Self::ResumeWithValue { .. } => "resume_with_value",
            Self::RequestWeakProbe { .. } => "request_weak_probe",
            Self::Abort { .. } => "abort",
        }
    }
}
