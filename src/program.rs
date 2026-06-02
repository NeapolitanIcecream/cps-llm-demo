use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
pub struct Program {
    pub program_id: String,
    pub input_schema: Value,
    pub output_schema: Value,
    pub instructions: Vec<Instr>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum Instr {
    Set {
        var: String,
        value: Value,
    },
    WeakCall {
        out: String,
        task: WeakTaskSpec,
        input: JsonExpr,
        output_schema: Value,
        min_confidence: f32,
    },
    Project {
        out: String,
        from: JsonExpr,
        path: Vec<String>,
    },
    Guard {
        condition: GuardExpr,
        on_fail: GuardFail,
    },
    Finish {
        value: JsonExpr,
    },
}

impl Instr {
    pub fn op_name(&self) -> &'static str {
        match self {
            Self::Set { .. } => "set",
            Self::WeakCall { .. } => "weak_call",
            Self::Project { .. } => "project",
            Self::Guard { .. } => "guard",
            Self::Finish { .. } => "finish",
        }
    }

    pub fn output_var(&self) -> Option<&str> {
        match self {
            Self::Set { var, .. } | Self::Project { out: var, .. } => Some(var),
            Self::WeakCall { out, .. } => Some(out),
            Self::Guard { .. } | Self::Finish { .. } => None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum JsonExpr {
    Literal { value: Value },
    Var { name: String },
    Object { fields: Vec<JsonObjectField> },
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
pub struct JsonObjectField {
    pub name: String,
    pub value: JsonExpr,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum GuardExpr {
    VarExists { name: String },
    JsonSchemaValid { var: String, schema: Value },
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum GuardFail {
    Think { reason: String },
    Abort { reason: String },
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
pub struct WeakTaskSpec {
    pub name: String,
    pub instructions: String,
}
