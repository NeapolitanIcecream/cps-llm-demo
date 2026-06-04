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
    #[serde(default)]
    pub successful_probes: Vec<ProbeProfileStats>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ProbeProfileStats {
    pub probe_id: String,
    pub effect_kind: String,
    pub handler: String,
    pub success_count: u64,
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
        if path.exists() {
            return read_json(&path);
        }
        self.load_split_files(workflow_id)
    }

    pub fn update_from_trace(&self, workflow_id: &str, events: &[TraceEvent]) -> Result<()> {
        self.state.ensure_workflow_layout(workflow_id)?;
        let mut profile = self.load(workflow_id)?;
        let mut active_capture_fingerprint_id = None::<String>;
        let mut pending_probe = None::<(String, ProbeProfileStats)>;

        for event in events {
            match event.event.as_str() {
                "capture_continuation" => {
                    if let Some(effect_key) = captured_perform_effect_key(event) {
                        if let Some(stats) = profile.effect_stats.get_mut(effect_key) {
                            stats.captures += 1;
                        }
                    }
                    let fingerprint = fingerprint_from_capture_event(event);
                    active_capture_fingerprint_id = Some(fingerprint.fingerprint_id.clone());
                    pending_probe = None;
                    let stats = profile
                        .failure_fingerprints
                        .entry(fingerprint.fingerprint_id.clone())
                        .or_insert_with(|| FailureFingerprintStats {
                            fingerprint,
                            count: 0,
                            sample_continuation_refs: Vec::new(),
                            sample_event_refs: Vec::new(),
                            successful_probes: Vec::new(),
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
                            effect_key: effect.clone(),
                            calls: 0,
                            captures: 0,
                            accepted: 0,
                        });
                    stats.calls += 1;
                }
                "effect_accepted" => {
                    if let Some(effect_key) = event
                        .detail
                        .get("effect")
                        .and_then(serde_json::Value::as_str)
                    {
                        if let Some(stats) = profile.effect_stats.get_mut(effect_key) {
                            stats.accepted += 1;
                        }
                    }
                    if event
                        .detail
                        .get("captured")
                        .and_then(serde_json::Value::as_bool)
                        .unwrap_or(false)
                    {
                        active_capture_fingerprint_id = None;
                        pending_probe = None;
                    }
                }
                "request_nested_effect" => {
                    let Some(fingerprint_id) = active_capture_fingerprint_id.as_ref() else {
                        pending_probe = None;
                        continue;
                    };
                    pending_probe = Some((
                        fingerprint_id.clone(),
                        ProbeProfileStats {
                            probe_id: format!(
                                "{}:{}",
                                event
                                    .detail
                                    .get("to_handler")
                                    .and_then(serde_json::Value::as_str)
                                    .unwrap_or("unknown"),
                                event
                                    .detail
                                    .get("to_effect")
                                    .and_then(serde_json::Value::as_str)
                                    .unwrap_or("unknown")
                            ),
                            effect_kind: event
                                .detail
                                .get("to_effect")
                                .and_then(serde_json::Value::as_str)
                                .unwrap_or("unknown")
                                .to_owned(),
                            handler: event
                                .detail
                                .get("to_handler")
                                .and_then(serde_json::Value::as_str)
                                .unwrap_or("unknown")
                                .to_owned(),
                            success_count: 1,
                        },
                    ));
                }
                "nested_effect_result" => {
                    let schema_valid = event
                        .detail
                        .get("schema_valid")
                        .and_then(serde_json::Value::as_bool)
                        .unwrap_or(false);
                    if !schema_valid {
                        pending_probe = None;
                        continue;
                    }
                    let Some((fingerprint_id, probe)) = pending_probe.take() else {
                        continue;
                    };
                    if let Some(stats) = profile.failure_fingerprints.get_mut(&fingerprint_id) {
                        if let Some(existing) = stats
                            .successful_probes
                            .iter_mut()
                            .find(|existing| existing.probe_id == probe.probe_id)
                        {
                            existing.success_count += 1;
                        } else {
                            stats.successful_probes.push(probe);
                        }
                    }
                }
                "resume_continuation" | "program_aborted" => {
                    active_capture_fingerprint_id = None;
                    pending_probe = None;
                }
                _ => {}
            }
        }

        self.write_profile_files(workflow_id, &profile)
    }

    fn load_split_files(&self, workflow_id: &str) -> Result<ProfileStoreData> {
        let profiles_dir = self.state.workflow_dir(workflow_id).join("profiles");
        let mut data = ProfileStoreData::default();
        let failures = profiles_dir.join("failure_fingerprints.json");
        if failures.exists() {
            data.failure_fingerprints = read_json(&failures)?;
        }
        let fast_paths = profiles_dir.join("fast_path_stats.json");
        if fast_paths.exists() {
            data.fast_path_stats = read_json(&fast_paths)?;
        }
        let effects = profiles_dir.join("effect_stats.json");
        if effects.exists() {
            data.effect_stats = read_json(&effects)?;
        }
        Ok(data)
    }

    fn write_profile_files(&self, workflow_id: &str, profile: &ProfileStoreData) -> Result<()> {
        let profiles_dir = self.state.workflow_dir(workflow_id).join("profiles");
        write_json_pretty(&profiles_dir.join("profile.json"), profile)?;
        write_json_pretty(
            &profiles_dir.join("failure_fingerprints.json"),
            &profile.failure_fingerprints,
        )?;
        write_json_pretty(
            &profiles_dir.join("fast_path_stats.json"),
            &profile.fast_path_stats,
        )?;
        write_json_pretty(
            &profiles_dir.join("effect_stats.json"),
            &profile.effect_stats,
        )
    }
}

fn push_sample(samples: &mut Vec<String>, value: &str) {
    if samples.len() < 8 && !samples.iter().any(|sample| sample == value) {
        samples.push(value.to_owned());
    }
}

fn captured_perform_effect_key(event: &TraceEvent) -> Option<&str> {
    match event
        .detail
        .get("failed_instruction_op")
        .and_then(serde_json::Value::as_str)
    {
        Some("perform") => event
            .detail
            .get("failed_effect_kind")
            .and_then(serde_json::Value::as_str),
        _ => None,
    }
}
