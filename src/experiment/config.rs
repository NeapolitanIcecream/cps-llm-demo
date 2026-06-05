use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use crate::store::budget_store::BudgetConfig;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ExperimentConfig {
    pub experiment_id: String,
    pub workflow_id: String,
    pub state_dir: Option<PathBuf>,
    #[serde(default)]
    pub workflow: Option<ExperimentWorkflow>,
    pub models: ExperimentModels,
    pub budget: ExperimentBudget,
    pub cache: ExperimentCache,
    pub schemas: ExperimentSchemas,
    pub data: ExperimentData,
    #[serde(default)]
    pub split_counts: Option<ExperimentSplitCounts>,
    pub phases: Vec<String>,
    #[serde(default)]
    pub shadow: Option<ShadowConfig>,
    #[serde(default)]
    pub patch_gate: Option<PatchGateConfigYaml>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ExperimentWorkflow {
    pub program: PathBuf,
    pub task: PathBuf,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ExperimentModels {
    pub weak_model: String,
    pub strong_model: String,
    pub base_url_env: String,
    pub api_key_env: String,
    pub use_responses_api: bool,
    pub structured_outputs: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ExperimentBudget {
    pub price_catalog: PathBuf,
    pub soft_cap_usd: f64,
    pub hard_cap_usd: f64,
    #[serde(default = "default_projection_multiplier")]
    pub projection_multiplier: f64,
}

impl ExperimentBudget {
    pub fn budget_config(&self) -> BudgetConfig {
        BudgetConfig {
            soft_cap_usd: self.soft_cap_usd,
            hard_cap_usd: self.hard_cap_usd,
            projection_multiplier: self.projection_multiplier,
            ..BudgetConfig::default()
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ExperimentCache {
    pub dir: PathBuf,
    pub mode: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ExperimentSchemas {
    pub event_schema: PathBuf,
    pub output_schema: PathBuf,
    pub gold_schema: PathBuf,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ExperimentData {
    pub all_events: PathBuf,
    pub gold_labels: PathBuf,
    pub splits_dir: PathBuf,
    #[serde(default)]
    pub optimizer_hard_negative_evidence: Option<PathBuf>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ExperimentSplitCounts {
    pub profile_train: usize,
    pub patch_validation: usize,
    pub heldout_test: usize,
    pub adversarial_test: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ShadowConfig {
    pub mode: String,
    pub sample_rate: f64,
    pub only_when_fast_path_hit: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PatchGateConfigYaml {
    pub max_quality_drop: f64,
    pub max_critical_miss_delta: f64,
    pub max_false_fast_path_rate: f64,
    pub min_fast_path_hit_rate_lift: f64,
    pub min_strong_think_rate_reduction: f64,
    pub max_continuation_frame_p95_bytes: u64,
    pub require_non_exact_generalization: bool,
}

pub fn load_experiment_config(path: &Path) -> Result<ExperimentConfig> {
    let raw = std::fs::read_to_string(path)
        .with_context(|| format!("failed to read experiment config {}", path.display()))?;
    serde_yaml::from_str(&raw)
        .with_context(|| format!("invalid experiment config YAML in {}", path.display()))
}

pub fn write_locked_config(config: &ExperimentConfig, path: &Path) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }
    let raw = serde_yaml::to_string(config)?;
    std::fs::write(path, raw).with_context(|| format!("failed to write {}", path.display()))
}

fn default_projection_multiplier() -> f64 {
    1.5
}
