use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use anyhow::{Context, Result, anyhow};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::experiment::quality::GoldLabel;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SplitStrategy {
    TimeCluster,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SplitCounts {
    pub profile_train: usize,
    pub patch_validation: usize,
    pub heldout_test: usize,
    pub adversarial_test: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LeakageReport {
    pub events_total: usize,
    pub exact_duplicate_cross_split: usize,
    pub near_duplicate_cross_split_rate: f64,
    pub semantic_clusters_total: usize,
    pub heldout_unseen_cluster_rate: f64,
    pub adversarial_cases_total: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SplitResult {
    pub leakage_report: LeakageReport,
}

pub fn split_events_files(
    events: &Path,
    gold: &Path,
    out: &Path,
    _strategy: SplitStrategy,
    counts: SplitCounts,
) -> Result<SplitResult> {
    let events = read_jsonl::<Value>(events)?;
    let labels = read_jsonl::<GoldLabel>(gold)?;
    let split = split_events(&events, &labels, &counts)?;
    std::fs::create_dir_all(out).with_context(|| format!("failed to create {}", out.display()))?;
    for (name, records) in [
        ("profile_train", &split.profile_train),
        ("patch_validation", &split.patch_validation),
        ("heldout_test", &split.heldout_test),
        ("adversarial_test", &split.adversarial_test),
    ] {
        write_jsonl(&out.join(format!("{name}.events.jsonl")), records)?;
        let ids = records.iter().filter_map(event_id).collect::<BTreeSet<_>>();
        let gold_records = labels
            .iter()
            .filter(|label| ids.contains(label.event_id.as_str()))
            .cloned()
            .collect::<Vec<_>>();
        write_jsonl(&out.join(format!("{name}.gold.jsonl")), &gold_records)?;
    }
    let report = leakage_report(&split, &labels);
    std::fs::write(
        out.join("leakage_report.json"),
        serde_json::to_vec_pretty(&report)?,
    )?;
    Ok(SplitResult {
        leakage_report: report,
    })
}

#[derive(Debug, Clone, PartialEq)]
pub struct EventSplits {
    pub profile_train: Vec<Value>,
    pub patch_validation: Vec<Value>,
    pub heldout_test: Vec<Value>,
    pub adversarial_test: Vec<Value>,
}

pub fn split_events(
    events: &[Value],
    labels: &[GoldLabel],
    counts: &SplitCounts,
) -> Result<EventSplits> {
    let labels_by_id = labels
        .iter()
        .map(|label| (label.event_id.as_str(), label))
        .collect::<BTreeMap<_, _>>();
    let mut adversarial = Vec::new();
    let mut ordinary = Vec::new();

    for event in events {
        let id = event_id(event).ok_or_else(|| anyhow!("event is missing event_id"))?;
        let label = labels_by_id
            .get(id)
            .ok_or_else(|| anyhow!("missing gold label for {id}"))?;
        if label.hard_negative {
            adversarial.push(event.clone());
        } else {
            ordinary.push(event.clone());
        }
    }

    ordinary.sort_by_key(sort_key);
    adversarial.sort_by_key(sort_key);
    let adversarial_test = take_exact(&mut adversarial, counts.adversarial_test);
    ordinary.extend(adversarial);
    ordinary.sort_by_key(sort_key);

    let groups = exact_duplicate_groups(ordinary);
    let mut profile_train = Vec::new();
    let mut patch_validation = Vec::new();
    let mut heldout_test = Vec::new();
    for group in groups {
        if profile_train.len() + group.len() <= counts.profile_train {
            profile_train.extend(group);
        } else if patch_validation.len() + group.len() <= counts.patch_validation {
            patch_validation.extend(group);
        } else {
            heldout_test.extend(group);
        }
    }
    heldout_test.truncate(counts.heldout_test);

    Ok(EventSplits {
        profile_train,
        patch_validation,
        heldout_test,
        adversarial_test,
    })
}

pub fn leakage_report(split: &EventSplits, labels: &[GoldLabel]) -> LeakageReport {
    let labeled_clusters = labels
        .iter()
        .map(|label| label.semantic_cluster.clone())
        .collect::<BTreeSet<_>>();
    let split_sets = [
        ("profile_train", &split.profile_train),
        ("patch_validation", &split.patch_validation),
        ("heldout_test", &split.heldout_test),
        ("adversarial_test", &split.adversarial_test),
    ];
    let mut exact_locations: BTreeMap<String, BTreeSet<&str>> = BTreeMap::new();
    let mut texts = Vec::new();
    for (name, events) in split_sets {
        for event in events {
            let normalized = normalized_text(event);
            exact_locations
                .entry(normalized.clone())
                .or_default()
                .insert(name);
            texts.push((name, normalized));
        }
    }
    let exact_duplicate_cross_split = exact_locations
        .values()
        .filter(|locations| locations.len() > 1)
        .count();

    let mut cross_pairs = 0_u64;
    let mut near_cross_pairs = 0_u64;
    for left in 0..texts.len() {
        for right in left + 1..texts.len() {
            if texts[left].0 == texts[right].0 {
                continue;
            }
            cross_pairs += 1;
            if char_ngram_jaccard(&texts[left].1, &texts[right].1, 3) >= 0.70 {
                near_cross_pairs += 1;
            }
        }
    }

    let label_by_id = labels
        .iter()
        .map(|label| (label.event_id.as_str(), label))
        .collect::<BTreeMap<_, _>>();
    let train_clusters = split
        .profile_train
        .iter()
        .filter_map(|event| event_id(event).and_then(|id| label_by_id.get(id)))
        .map(|label| label.semantic_cluster.as_str())
        .collect::<BTreeSet<_>>();
    let heldout_total = split.heldout_test.len() as u64;
    let heldout_unseen = split
        .heldout_test
        .iter()
        .filter_map(|event| event_id(event).and_then(|id| label_by_id.get(id)))
        .filter(|label| !train_clusters.contains(label.semantic_cluster.as_str()))
        .count() as u64;

    LeakageReport {
        events_total: split.profile_train.len()
            + split.patch_validation.len()
            + split.heldout_test.len()
            + split.adversarial_test.len(),
        exact_duplicate_cross_split,
        near_duplicate_cross_split_rate: if cross_pairs == 0 {
            0.0
        } else {
            near_cross_pairs as f64 / cross_pairs as f64
        },
        semantic_clusters_total: labeled_clusters.len(),
        heldout_unseen_cluster_rate: if heldout_total == 0 {
            0.0
        } else {
            heldout_unseen as f64 / heldout_total as f64
        },
        adversarial_cases_total: split.adversarial_test.len(),
    }
}

pub fn normalized_text(event: &Value) -> String {
    event
        .get("text")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .chars()
        .filter(|ch| !ch.is_whitespace() && !ch.is_ascii_punctuation())
        .flat_map(char::to_lowercase)
        .collect()
}

pub fn char_ngram_jaccard(left: &str, right: &str, n: usize) -> f64 {
    let left = char_ngrams(left, n);
    let right = char_ngrams(right, n);
    if left.is_empty() && right.is_empty() {
        return 1.0;
    }
    let intersection = left.intersection(&right).count();
    let union = left.union(&right).count();
    intersection as f64 / union.max(1) as f64
}

fn char_ngrams(value: &str, n: usize) -> BTreeSet<String> {
    let chars = value.chars().collect::<Vec<_>>();
    if chars.len() <= n {
        return [value.to_owned()].into_iter().collect();
    }
    chars
        .windows(n)
        .map(|window| window.iter().collect::<String>())
        .collect()
}

fn exact_duplicate_groups(events: Vec<Value>) -> Vec<Vec<Value>> {
    let mut group_index_by_text: BTreeMap<String, usize> = BTreeMap::new();
    let mut groups: Vec<Vec<Value>> = Vec::new();
    for event in events {
        let normalized = normalized_text(&event);
        if let Some(index) = group_index_by_text.get(&normalized).copied() {
            groups[index].push(event);
        } else {
            group_index_by_text.insert(normalized, groups.len());
            groups.push(vec![event]);
        }
    }
    groups
}

fn take_exact(values: &mut Vec<Value>, count: usize) -> Vec<Value> {
    values.drain(0..values.len().min(count)).collect()
}

fn sort_key(event: &Value) -> String {
    format!(
        "{}:{}",
        event.get("timestamp").and_then(Value::as_str).unwrap_or(""),
        event.get("event_id").and_then(Value::as_str).unwrap_or("")
    )
}

fn event_id(event: &Value) -> Option<&str> {
    event.get("event_id").and_then(Value::as_str)
}

fn read_jsonl<T: for<'de> Deserialize<'de>>(path: &Path) -> Result<Vec<T>> {
    let raw = std::fs::read_to_string(path)
        .with_context(|| format!("failed to read {}", path.display()))?;
    raw.lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| serde_json::from_str(line).context("invalid split JSONL record"))
        .collect()
}

fn write_jsonl<T: Serialize>(path: &Path, values: &[T]) -> Result<()> {
    let mut raw = Vec::new();
    for value in values {
        raw.extend(serde_json::to_vec(value)?);
        raw.push(b'\n');
    }
    std::fs::write(path, raw).with_context(|| format!("failed to write {}", path.display()))
}
