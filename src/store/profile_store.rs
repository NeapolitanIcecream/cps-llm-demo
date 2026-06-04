use std::collections::BTreeMap;

use anyhow::Result;
use serde::{Deserialize, Serialize};

use crate::optimizer::failure_fingerprint::{FailureFingerprint, fingerprint_from_capture_event};
use crate::store::state_dir::{StateDir, read_json, write_json_pretty};
use crate::trace::TraceEvent;

#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct ProfileStoreData {
    pub failure_fingerprints: BTreeMap<String, FailureFingerprintStats>,
    pub fast_path_stats: BTreeMap<String, FastPathRuleStats>,
    pub effect_stats: BTreeMap<String, EffectStats>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FailureFingerprintStats {
    pub fingerprint: FailureFingerprint,
    pub count: u64,
    pub sample_continuation_refs: Vec<String>,
    pub sample_event_refs: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FastPathRuleStats {
    pub rule_id: String,
    pub hits: u64,
    pub misses: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct EffectStats {
    pub effect_key: String,
    pub calls: u64,
    pub captures: u64,
    pub accepted: u64,
}

#[derive(Debug, Clone)]
pub struct FileProfileStore {
    state: StateDir,
}

impl FileProfileStore {
    pub fn new(state: StateDir) -> Self {
        Self { state }
    }

    pub fn load(&self, workflow_id: &str) -> Result<ProfileStoreData> {
        let path = self
            .state
            .workflow_dir(workflow_id)
            .join("profiles")
            .join("profile.json");
        if !path.exists() {
            return Ok(ProfileStoreData::default());
        }
        read_json(&path)
    }

    pub fn update_from_trace(&self, workflow_id: &str, events: &[TraceEvent]) -> Result<()> {
        self.state.ensure_workflow_layout(workflow_id)?;
        let mut profile = self.load(workflow_id)?;

        for event in events {
            match event.event.as_str() {
                "capture_continuation" => {
                    let fingerprint = fingerprint_from_capture_event(event);
                    let stats = profile
                        .failure_fingerprints
                        .entry(fingerprint.fingerprint_id.clone())
                        .or_insert_with(|| FailureFingerprintStats {
                            fingerprint,
                            count: 0,
                            sample_continuation_refs: Vec::new(),
                            sample_event_refs: Vec::new(),
                        });
                    stats.count += 1;
                    if let Some(continuation_id) = event
                        .detail
                        .get("continuation_id")
                        .and_then(serde_json::Value::as_str)
                    {
                        push_sample(&mut stats.sample_continuation_refs, continuation_id);
                    }
                    push_sample(&mut stats.sample_event_refs, &event.event_id);
                }
                "fast_path_hit" | "fast_path_miss" => {
                    let rule_id = event
                        .detail
                        .get("rule_id")
                        .and_then(serde_json::Value::as_str)
                        .unwrap_or("none")
                        .to_owned();
                    let stats = profile.fast_path_stats.entry(rule_id.clone()).or_insert(
                        FastPathRuleStats {
                            rule_id,
                            hits: 0,
                            misses: 0,
                        },
                    );
                    if event.event == "fast_path_hit" {
                        stats.hits += 1;
                    } else {
                        stats.misses += 1;
                    }
                }
                "perform_effect" => {
                    let effect = event
                        .detail
                        .get("effect")
                        .and_then(serde_json::Value::as_str)
                        .unwrap_or("unknown")
                        .to_owned();
                    let stats = profile
                        .effect_stats
                        .entry(effect.clone())
                        .or_insert(EffectStats {
                            effect_key: effect,
                            calls: 0,
                            captures: 0,
                            accepted: 0,
                        });
                    stats.calls += 1;
                }
                _ => {}
            }
        }

        write_json_pretty(
            &self
                .state
                .workflow_dir(workflow_id)
                .join("profiles")
                .join("profile.json"),
            &profile,
        )
    }
}

fn push_sample(samples: &mut Vec<String>, value: &str) {
    if samples.len() < 8 && !samples.iter().any(|sample| sample == value) {
        samples.push(value.to_owned());
    }
}
