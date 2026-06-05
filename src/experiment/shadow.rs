use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::experiment::prediction::PredictionRecord;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ShadowAuditRecord {
    pub event_id: String,
    pub fast_path_output: Value,
    pub shadow_output: Value,
    pub disagreed: bool,
    pub critical: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ShadowMetrics {
    pub fast_path_hits: u64,
    pub shadow_checked: u64,
    pub shadow_disagreements: u64,
    pub shadow_disagreement_rate: f64,
    pub critical_shadow_disagreements: u64,
}

pub fn shadow_execution_does_not_change_output(
    prediction: &PredictionRecord,
    shadow_output: Value,
    critical: bool,
) -> (PredictionRecord, ShadowAuditRecord) {
    let disagreed = outputs_semantically_disagree(&prediction.output, &shadow_output);
    (
        prediction.clone(),
        ShadowAuditRecord {
            event_id: prediction.event_id.clone(),
            fast_path_output: prediction.output.clone(),
            shadow_output,
            disagreed,
            critical,
        },
    )
}

pub fn summarize_shadow_audit(
    predictions: &[PredictionRecord],
    records: &[ShadowAuditRecord],
) -> ShadowMetrics {
    let fast_path_hits = predictions
        .iter()
        .filter(|prediction| prediction.fast_path.hit)
        .count() as u64;
    let shadow_checked = records.len() as u64;
    let shadow_disagreements = records.iter().filter(|record| record.disagreed).count() as u64;
    let critical_shadow_disagreements = records
        .iter()
        .filter(|record| record.disagreed && record.critical)
        .count() as u64;
    ShadowMetrics {
        fast_path_hits,
        shadow_checked,
        shadow_disagreements,
        shadow_disagreement_rate: if shadow_checked == 0 {
            0.0
        } else {
            shadow_disagreements as f64 / shadow_checked as f64
        },
        critical_shadow_disagreements,
    }
}

fn outputs_semantically_disagree(left: &Value, right: &Value) -> bool {
    let left_kind = normalized_action_kind(left);
    let right_kind = normalized_action_kind(right);
    if left_kind != right_kind {
        return true;
    }
    if left_kind.as_deref() == Some("no_action") {
        return false;
    }
    normalize_optional_text(
        left.get("datetime_hint")
            .or_else(|| left.get("datetime_canonical")),
    ) != normalize_optional_text(
        right
            .get("datetime_hint")
            .or_else(|| right.get("datetime_canonical")),
    )
}

fn normalized_action_kind(value: &Value) -> Option<String> {
    value
        .get("kind")
        .and_then(Value::as_str)
        .map(|kind| match kind {
            "ignore" | "none" => "no_action".to_owned(),
            other => other.to_owned(),
        })
}

fn normalize_optional_text(value: Option<&Value>) -> Option<String> {
    value
        .and_then(Value::as_str)
        .map(|value| value.split_whitespace().collect::<String>().to_lowercase())
}
