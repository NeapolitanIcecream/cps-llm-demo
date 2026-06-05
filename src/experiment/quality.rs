use std::collections::BTreeMap;
use std::path::Path;

use anyhow::{Context, Result, anyhow};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::experiment::prediction::PredictionRecord;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct GoldLabel {
    pub event_id: String,
    pub semantic_cluster: String,
    pub is_actionable: bool,
    pub kind: String,
    pub title_canonical: Option<String>,
    pub datetime_canonical: Option<String>,
    pub criticality: String,
    #[serde(default)]
    pub hard_negative: bool,
    #[serde(default)]
    pub generated_from_predictions: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct QualityMetrics {
    pub events_total: u64,
    pub schema_validity: f64,
    pub intent_accuracy: f64,
    pub actionable_precision: f64,
    pub actionable_recall: f64,
    pub datetime_accuracy: f64,
    pub critical_miss_rate: f64,
    pub hard_negative_false_action_rate: f64,
}

pub fn evaluate_quality(
    predictions: &[PredictionRecord],
    gold_labels: &[GoldLabel],
) -> Result<QualityMetrics> {
    if gold_labels
        .iter()
        .any(|label| label.generated_from_predictions)
    {
        return Err(anyhow!(
            "gold labels must be independent; generated_from_predictions=true is not allowed"
        ));
    }
    let gold_by_id = gold_labels
        .iter()
        .map(|label| (label.event_id.as_str(), label))
        .collect::<BTreeMap<_, _>>();

    let mut schema_valid = 0_u64;
    let mut intent_correct = 0_u64;
    let mut predicted_actionable = 0_u64;
    let mut true_actionable = 0_u64;
    let mut true_positive_actionable = 0_u64;
    let mut datetime_total = 0_u64;
    let mut datetime_correct = 0_u64;
    let mut critical_total = 0_u64;
    let mut critical_misses = 0_u64;
    let mut hard_negative_total = 0_u64;
    let mut hard_negative_false_actions = 0_u64;

    for prediction in predictions {
        let gold = gold_by_id
            .get(prediction.event_id.as_str())
            .ok_or_else(|| anyhow!("missing gold label for {}", prediction.event_id))?;
        if prediction.schema_valid {
            schema_valid += 1;
        }
        let predicted_kind_raw = prediction
            .output
            .get("kind")
            .and_then(Value::as_str)
            .unwrap_or("none");
        let predicted_kind = normalize_kind(predicted_kind_raw);
        let gold_kind = normalize_kind(&gold.kind);
        let prediction_actionable = is_actionable_kind(&predicted_kind);
        if prediction_actionable {
            predicted_actionable += 1;
        }
        if gold.is_actionable {
            true_actionable += 1;
        }
        if prediction_actionable && gold.is_actionable {
            true_positive_actionable += 1;
        }
        if predicted_kind == gold_kind {
            intent_correct += 1;
        }
        if gold.datetime_canonical.is_some() {
            datetime_total += 1;
            let predicted_datetime = prediction
                .output
                .get("datetime_hint")
                .or_else(|| prediction.output.get("datetime_canonical"))
                .and_then(Value::as_str)
                .map(normalize_text);
            if predicted_datetime == gold.datetime_canonical.as_deref().map(normalize_text) {
                datetime_correct += 1;
            }
        }
        if gold.criticality == "critical" && gold.is_actionable {
            critical_total += 1;
            if !prediction_actionable || predicted_kind != gold_kind {
                critical_misses += 1;
            }
        }
        if gold.hard_negative {
            hard_negative_total += 1;
            if prediction_actionable {
                hard_negative_false_actions += 1;
            }
        }
    }

    let total = predictions.len() as u64;
    Ok(QualityMetrics {
        events_total: total,
        schema_validity: rate(schema_valid, total),
        intent_accuracy: rate(intent_correct, total),
        actionable_precision: rate(true_positive_actionable, predicted_actionable),
        actionable_recall: rate(true_positive_actionable, true_actionable),
        datetime_accuracy: rate(datetime_correct, datetime_total),
        critical_miss_rate: rate(critical_misses, critical_total),
        hard_negative_false_action_rate: rate(hard_negative_false_actions, hard_negative_total),
    })
}

pub fn evaluate_quality_files(
    predictions: &Path,
    gold: &Path,
    out: &Path,
) -> Result<QualityMetrics> {
    let predictions = crate::experiment::prediction::PredictionStore::read(predictions)?;
    let gold_labels = read_gold_labels(gold)?;
    let metrics = evaluate_quality(&predictions, &gold_labels)?;
    if let Some(parent) = out.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }
    std::fs::write(out, serde_json::to_vec_pretty(&metrics)?)
        .with_context(|| format!("failed to write {}", out.display()))?;
    Ok(metrics)
}

pub fn read_gold_labels(path: &Path) -> Result<Vec<GoldLabel>> {
    read_jsonl(path)
}

pub fn read_jsonl<T: for<'de> Deserialize<'de>>(path: &Path) -> Result<Vec<T>> {
    let raw = std::fs::read_to_string(path)
        .with_context(|| format!("failed to read {}", path.display()))?;
    raw.lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| serde_json::from_str(line).context("invalid quality JSONL record"))
        .collect()
}

fn rate(numerator: u64, denominator: u64) -> f64 {
    if denominator == 0 {
        0.0
    } else {
        numerator as f64 / denominator as f64
    }
}

fn normalize_text(value: &str) -> String {
    value.split_whitespace().collect::<String>().to_lowercase()
}

fn normalize_kind(value: &str) -> String {
    match value {
        "ignore" | "none" => "no_action".to_owned(),
        other => other.to_owned(),
    }
}

fn is_actionable_kind(kind: &str) -> bool {
    kind != "no_action"
}
