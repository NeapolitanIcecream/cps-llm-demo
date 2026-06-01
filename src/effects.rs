use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::domain::{MessageEvent, ResolvedIntent, WeakIntentGuess};

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
#[serde(tag = "tag", content = "data", rename_all = "snake_case")]
pub enum Continuation {
    AfterClassifyMessage {
        event: MessageEvent,
        weak_guess: Option<WeakIntentGuess>,
        weak_error: Option<String>,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ExpectedType {
    ResolvedIntent,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
pub struct ThinkFrame {
    pub reason: String,
    pub expected: ExpectedType,
    pub event: MessageEvent,
    pub weak_guess: Option<WeakIntentGuess>,
    pub weak_error: Option<String>,
    pub allowed_decisions: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
pub struct EffectFrame {
    pub effect_id: String,
    pub effect: EffectKind,
    pub continuation: Continuation,
    pub frame: ThinkFrame,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum EffectKind {
    Think,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
#[serde(tag = "decision", content = "data", rename_all = "snake_case")]
pub enum ThinkDecision {
    Value(ResolvedIntent),
    Abort { reason: String },
}
