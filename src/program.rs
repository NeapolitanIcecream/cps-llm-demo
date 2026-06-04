use std::collections::BTreeMap;

use schemars::JsonSchema;
use serde::{Deserialize, Deserializer, Serialize};
use serde_json::Value;

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
pub struct Program {
    pub program_id: String,
    pub version: String,
    pub entry: String,
    pub input_schema: Value,
    pub output_schema: Value,
    pub functions: BTreeMap<String, FunctionDef>,
    pub allowed_effects: Vec<EffectPermission>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
pub struct FunctionDef {
    pub params: Vec<String>,
    pub output_schema: Value,
    pub body: Vec<Instr>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum Instr {
    Let {
        var: String,
        expr: JsonExpr,
    },
    Project {
        out: String,
        from: JsonExpr,
        path: Vec<String>,
    },
    Perform {
        out: String,
        effect: EffectCall,
        input: JsonExpr,
        expected_schema: Value,
        acceptance: AcceptancePolicy,
    },
    Guard {
        condition: GuardExpr,
        on_fail: GuardFail,
    },
    Branch {
        condition: GuardExpr,
        then_pc: usize,
        else_pc: usize,
    },
    Jump {
        pc: usize,
    },
    Call {
        out: String,
        function: String,
        args: Vec<JsonExpr>,
    },
    Map {
        out: String,
        items: JsonExpr,
        item_var: String,
        function: String,
    },
    CallDynamic {
        out: String,
        fragment: JsonExpr,
        args: Vec<JsonExpr>,
    },
    Return {
        value: JsonExpr,
    },
}

impl Instr {
    pub fn op_name(&self) -> &'static str {
        match self {
            Self::Let { .. } => "let",
            Self::Project { .. } => "project",
            Self::Perform { .. } => "perform",
            Self::Guard { .. } => "guard",
            Self::Branch { .. } => "branch",
            Self::Jump { .. } => "jump",
            Self::Call { .. } => "call",
            Self::Map { .. } => "map",
            Self::CallDynamic { .. } => "call_dynamic",
            Self::Return { .. } => "return",
        }
    }

    pub fn output_var(&self) -> Option<&str> {
        match self {
            Self::Let { var, .. } => Some(var),
            Self::Project { out, .. }
            | Self::Perform { out, .. }
            | Self::Call { out, .. }
            | Self::Map { out, .. }
            | Self::CallDynamic { out, .. } => Some(out),
            Self::Guard { .. } | Self::Branch { .. } | Self::Jump { .. } | Self::Return { .. } => {
                None
            }
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum JsonExpr {
    Literal { value: Value },
    Var { name: String },
    Object { fields: BTreeMap<String, JsonExpr> },
    Array { items: Vec<JsonExpr> },
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum GuardExpr {
    VarExists {
        name: String,
    },
    JsonSchemaValid {
        var: String,
        schema: Value,
    },
    FieldEquals {
        var: String,
        path: Vec<String>,
        value: Value,
    },
    FieldIsTruthy {
        var: String,
        path: Vec<String>,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum GuardFail {
    Think { reason: String },
    Abort { reason: String },
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum EffectCall {
    ModelTask {
        strength: ModelStrength,
        task: ModelTaskSpec,
    },
    Think {
        reason: String,
    },
    CompileProgram {
        strength: ModelStrength,
        task_spec: String,
        input_schema: Value,
        output_schema: Value,
    },
    LocalTool {
        tool_name: String,
        args_schema: Value,
    },
}

impl EffectCall {
    pub fn kind_name(&self) -> &'static str {
        match self {
            Self::ModelTask { .. } => "model_task",
            Self::Think { .. } => "think",
            Self::CompileProgram { .. } => "compile_program",
            Self::LocalTool { .. } => "local_tool",
        }
    }

    pub fn handler_name(&self) -> &'static str {
        match self {
            Self::ModelTask {
                strength: ModelStrength::Weak,
                ..
            } => "weak_model",
            Self::ModelTask {
                strength: ModelStrength::Strong,
                ..
            }
            | Self::Think { .. }
            | Self::CompileProgram {
                strength: ModelStrength::Strong,
                ..
            } => "strong_model",
            Self::CompileProgram {
                strength: ModelStrength::Weak,
                ..
            } => "weak_model",
            Self::LocalTool { .. } => "local_tool",
        }
    }

    pub fn model_task_name(&self) -> Option<&str> {
        match self {
            Self::ModelTask { task, .. } => Some(task.name.as_str()),
            _ => None,
        }
    }

    pub fn strength(&self) -> Option<ModelStrength> {
        match self {
            Self::ModelTask { strength, .. } | Self::CompileProgram { strength, .. } => {
                Some(*strength)
            }
            Self::Think { .. } | Self::LocalTool { .. } => None,
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ModelStrength {
    Weak,
    Strong,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
pub struct ModelTaskSpec {
    pub name: String,
    pub instructions: String,
}

pub type WeakTaskSpec = ModelTaskSpec;

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
pub struct AcceptancePolicy {
    #[serde(default, deserialize_with = "deserialize_optional_probability")]
    #[schemars(range(min = 0.0, max = 1.0))]
    pub min_confidence: Option<f32>,
    pub require_schema_valid: bool,
    pub on_failure: FailureHandler,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum FailureHandler {
    CaptureToThink { reason: String },
    Abort { reason: String },
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum EffectPermission {
    ModelTask { strength: ModelStrength },
    Think,
    CompileProgram { strength: ModelStrength },
    LocalTool { tool_name: String },
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
pub struct ProgramFragment {
    pub functions: BTreeMap<String, FunctionDef>,
    pub entry: String,
    pub input_schema: Value,
    pub output_schema: Value,
    pub allowed_effects: Vec<EffectPermission>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
pub struct ProgramPatch {
    pub target_program_id: String,
    pub patch_id: String,
    pub operations: Vec<PatchOp>,
    pub rationale: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum PatchOp {
    ReplaceInstruction {
        function: String,
        pc: usize,
        instr: Instr,
    },
    InsertInstruction {
        function: String,
        pc: usize,
        instr: Instr,
    },
    AddFunction {
        name: String,
        function: FunctionDef,
    },
    UpdateAcceptancePolicy {
        function: String,
        pc: usize,
        acceptance: AcceptancePolicy,
    },
}

fn deserialize_optional_probability<'de, D>(deserializer: D) -> Result<Option<f32>, D::Error>
where
    D: Deserializer<'de>,
{
    let value = Option::<f32>::deserialize(deserializer)?;
    match value {
        Some(value) if value.is_finite() && (0.0..=1.0).contains(&value) => Ok(Some(value)),
        Some(_) => Err(serde::de::Error::custom(
            "must be a finite probability between 0.0 and 1.0",
        )),
        None => Ok(None),
    }
}
