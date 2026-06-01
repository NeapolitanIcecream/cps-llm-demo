use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
pub struct MessageEvent {
    pub event_id: String,
    pub text: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum IntentKind {
    Ignore,
    CreateTask,
    CreateCalendarEvent,
    DraftReply,
    NeedStrongThink,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
pub struct WeakIntentGuess {
    pub kind: IntentKind,
    pub title: Option<String>,
    pub datetime_hint: Option<String>,
    pub confidence: f32,
    pub rationale: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
pub struct ResolvedIntent {
    pub kind: IntentKind,
    pub title: String,
    pub datetime_hint: Option<String>,
    pub confidence: f32,
    pub source: DecisionSource,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum DecisionSource {
    DeterministicCode,
    WeakModel,
    StrongThink,
    UserFallback,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
pub struct ActionDraft {
    pub event_id: String,
    pub kind: IntentKind,
    pub title: String,
    pub datetime_hint: Option<String>,
    pub source: DecisionSource,
}
