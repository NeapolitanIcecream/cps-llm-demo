use std::collections::BTreeMap;
use std::path::Path;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::experiment::prediction::PredictionRecord;
use crate::experiment::split::normalized_text;
use crate::program::{GeneralizationScope, PatchGeneralizationMetadata, PatchKind, ProgramPatch};
use crate::store::state_dir::stable_hash_bytes;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ExactMemoEntry {
    pub normalized_text_hash: String,
    pub output: Value,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct ExactMemoTable {
    pub entries: BTreeMap<String, ExactMemoEntry>,
}

impl ExactMemoTable {
    pub fn from_predictions(events: &[Value], predictions: &[PredictionRecord]) -> Self {
        let output_by_id = predictions
            .iter()
            .map(|prediction| (prediction.event_id.as_str(), prediction.output.clone()))
            .collect::<BTreeMap<_, _>>();
        let mut entries = BTreeMap::new();
        for event in events {
            let Some(event_id) = event.get("event_id").and_then(Value::as_str) else {
                continue;
            };
            let Some(output) = output_by_id.get(event_id).cloned() else {
                continue;
            };
            let hash = normalized_text_hash(event);
            entries.insert(
                hash.clone(),
                ExactMemoEntry {
                    normalized_text_hash: hash,
                    output,
                },
            );
        }
        Self { entries }
    }

    pub fn lookup(&self, event: &Value) -> Option<&Value> {
        self.entries
            .get(&normalized_text_hash(event))
            .map(|entry| &entry.output)
    }
}

pub fn normalized_text_hash(event: &Value) -> String {
    stable_hash_bytes(normalized_text(event).as_bytes())
}

pub fn exact_memo_patch(program_id: &str, patch_id: &str) -> ProgramPatch {
    ProgramPatch {
        target_program_id: program_id.to_owned(),
        patch_id: patch_id.to_owned(),
        operations: Vec::new(),
        rationale: "exact memo ablation patch metadata".to_owned(),
        generalization: Some(PatchGeneralizationMetadata {
            patch_kind: PatchKind::ExactMemo,
            generalization_scope: GeneralizationScope::Exact,
            uses_weak_semantic_matcher: false,
            uses_deterministic_fast_path: true,
            uses_validator: false,
            declared_positive_clusters: Vec::new(),
            declared_negative_clusters: Vec::new(),
        }),
    }
}

pub fn write_exact_memo_patch(path: &Path, program_id: &str) -> Result<ProgramPatch> {
    let patch = exact_memo_patch(program_id, "exact_memo_v1");
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }
    std::fs::write(path, serde_json::to_vec_pretty(&patch)?)
        .with_context(|| format!("failed to write {}", path.display()))?;
    Ok(patch)
}
