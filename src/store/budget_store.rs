use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result, anyhow};
use serde::{Deserialize, Serialize};

use crate::pricing::price_catalog::PriceCatalog;
use crate::store::model_call_store::{
    CacheStatus, ModelCallRecord, conservative_bytes_to_tokens, estimate_usage_from_bytes,
};
use crate::store::state_dir::{StateDir, now_string, read_json, write_json_pretty};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BudgetConfig {
    pub soft_cap_usd: f64,
    pub hard_cap_usd: f64,
    #[serde(default = "default_reserve_usd")]
    pub reserve_usd: f64,
    #[serde(default = "default_abort_when_projected_over_hard_cap")]
    pub abort_when_projected_over_hard_cap: bool,
    #[serde(default = "default_warn_when_over_soft_cap")]
    pub warn_when_over_soft_cap: bool,
    #[serde(default = "default_projection_multiplier")]
    pub projection_multiplier: f64,
}

impl Default for BudgetConfig {
    fn default() -> Self {
        Self {
            soft_cap_usd: 60.0,
            hard_cap_usd: 100.0,
            reserve_usd: 15.0,
            abort_when_projected_over_hard_cap: true,
            warn_when_over_soft_cap: true,
            projection_multiplier: 1.5,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BudgetSpendRecord {
    pub call_id: String,
    pub run_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub budget_scope_id: Option<String>,
    pub phase: Option<String>,
    pub model: String,
    pub cost_usd: f64,
    pub cache_status: CacheStatus,
    pub created_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BudgetReport {
    pub hard_cap_usd: f64,
    pub soft_cap_usd: f64,
    pub spent_usd: f64,
    pub remaining_usd: f64,
    pub calls_total: u64,
    pub cache_hit_rate: f64,
    pub by_model: BTreeMap<String, f64>,
    pub by_phase: BTreeMap<String, f64>,
}

#[derive(Debug, Clone)]
pub struct FileBudgetStore {
    state: StateDir,
}

impl FileBudgetStore {
    pub fn new(state: StateDir) -> Self {
        Self { state }
    }

    pub fn write_config(&self, config: &BudgetConfig) -> Result<()> {
        write_json_pretty(&self.config_path(), config)
    }

    pub fn read_config(&self) -> Result<BudgetConfig> {
        if self.config_path().exists() {
            read_json(&self.config_path())
        } else {
            Ok(BudgetConfig::default())
        }
    }

    pub fn append_spend(&self, record: &BudgetSpendRecord) -> Result<()> {
        append_jsonl(&self.spend_path(), record)
    }

    pub fn record_model_call(&self, record: &ModelCallRecord) -> Result<()> {
        self.append_spend(&BudgetSpendRecord {
            call_id: record.call_id.clone(),
            run_id: record.run_id.clone(),
            budget_scope_id: record.budget_scope_id.clone(),
            phase: record.phase.clone(),
            model: record.model.clone(),
            cost_usd: record.cost.total_usd,
            cache_status: record.cache_status,
            created_at: record.created_at.clone(),
        })
    }

    pub fn list_spend(&self) -> Result<Vec<BudgetSpendRecord>> {
        read_jsonl(&self.spend_path())
    }

    pub fn reset(&self) -> Result<()> {
        if self.spend_path().exists() {
            std::fs::remove_file(self.spend_path())
                .with_context(|| "failed to reset budget spend ledger")?;
        }
        Ok(())
    }

    pub fn report(&self, config: &BudgetConfig) -> Result<BudgetReport> {
        let spend = self.list_spend()?;
        Ok(build_budget_report(config, &spend))
    }

    pub fn report_for_scope(
        &self,
        config: &BudgetConfig,
        budget_scope_id: Option<&str>,
        run_id: Option<&str>,
    ) -> Result<BudgetReport> {
        let spend = scoped_spend(self.list_spend()?, budget_scope_id, run_id);
        Ok(build_budget_report(config, &spend))
    }

    fn config_path(&self) -> PathBuf {
        self.state
            .root()
            .join("budgets")
            .join("experiment_budget.json")
    }

    fn spend_path(&self) -> PathBuf {
        self.state.root().join("budgets").join("spend.jsonl")
    }
}

#[derive(Debug, Clone)]
pub struct BudgetGuard {
    pub config: BudgetConfig,
    pub catalog: PriceCatalog,
    pub store: FileBudgetStore,
    reservations: Arc<Mutex<BTreeMap<BudgetReservationKey, f64>>>,
}

impl BudgetGuard {
    pub fn new(config: BudgetConfig, catalog: PriceCatalog, store: FileBudgetStore) -> Self {
        Self {
            config,
            catalog,
            store,
            reservations: Arc::new(Mutex::new(BTreeMap::new())),
        }
    }

    pub fn ensure_call_allowed(
        &self,
        model: &str,
        input_bytes: u64,
        max_output_tokens: Option<u64>,
    ) -> Result<()> {
        self.ensure_call_allowed_for_scope(model, input_bytes, max_output_tokens, None, None)
    }

    pub fn ensure_call_allowed_for_scope(
        &self,
        model: &str,
        input_bytes: u64,
        max_output_tokens: Option<u64>,
        budget_scope_id: Option<&str>,
        run_id: Option<&str>,
    ) -> Result<()> {
        self.check_call_allowed_for_scope(
            model,
            input_bytes,
            max_output_tokens,
            budget_scope_id,
            run_id,
            false,
        )
        .map(|_| ())
    }

    pub fn reserve_call_for_scope(
        &self,
        model: &str,
        input_bytes: u64,
        max_output_tokens: Option<u64>,
        budget_scope_id: Option<&str>,
        run_id: Option<&str>,
    ) -> Result<BudgetReservation> {
        self.check_call_allowed_for_scope(
            model,
            input_bytes,
            max_output_tokens,
            budget_scope_id,
            run_id,
            true,
        )
    }

    fn check_call_allowed_for_scope(
        &self,
        model: &str,
        input_bytes: u64,
        max_output_tokens: Option<u64>,
        budget_scope_id: Option<&str>,
        run_id: Option<&str>,
        reserve: bool,
    ) -> Result<BudgetReservation> {
        let report = self
            .store
            .report_for_scope(&self.config, budget_scope_id, run_id)?;
        let Some(projected_cost) = self.projected_cost_usd(model, input_bytes, max_output_tokens)
        else {
            return Ok(BudgetReservation::none());
        };
        let key = BudgetReservationKey::new(budget_scope_id, run_id);
        let mut reservations = self
            .reservations
            .lock()
            .map_err(|_| anyhow!("budget reservation lock poisoned"))?;
        let in_flight_spend = reservations.get(&key).copied().unwrap_or(0.0);
        let projected_spend = report.spent_usd + in_flight_spend + projected_cost;
        if self.config.abort_when_projected_over_hard_cap
            && projected_spend > self.config.hard_cap_usd
        {
            return Err(anyhow!(
                "budget hard cap would be exceeded: spent ${:.4}, in-flight ${:.4}, projected call ${:.4}, hard cap ${:.4}",
                report.spent_usd,
                in_flight_spend,
                projected_cost,
                self.config.hard_cap_usd
            ));
        }
        if reserve {
            *reservations.entry(key.clone()).or_insert(0.0) += projected_cost;
            Ok(BudgetReservation {
                reservations: Some(Arc::clone(&self.reservations)),
                key,
                amount_usd: projected_cost,
            })
        } else {
            Ok(BudgetReservation::none())
        }
    }

    fn projected_cost_usd(
        &self,
        model: &str,
        input_bytes: u64,
        max_output_tokens: Option<u64>,
    ) -> Option<f64> {
        let projected_output_tokens = max_output_tokens.unwrap_or(4_096);
        let projected_usage = estimate_usage_from_bytes(input_bytes, projected_output_tokens * 2);
        match self.catalog.estimate_cost(model, &projected_usage, true) {
            Ok(cost) => Some(cost.total_usd * self.config.projection_multiplier),
            Err(_) => None,
        }
    }

    pub fn estimated_cost_for_request(&self, model: &str, input_bytes: u64) -> Result<f64> {
        let usage = crate::store::model_call_store::ModelUsage {
            input_tokens: conservative_bytes_to_tokens(input_bytes),
            cached_input_tokens: 0,
            output_tokens: 4_096,
            reasoning_tokens: 0,
            total_tokens: conservative_bytes_to_tokens(input_bytes) + 4_096,
        };
        Ok(self.catalog.estimate_cost(model, &usage, true)?.total_usd)
    }
}

#[derive(Debug)]
pub struct BudgetReservation {
    reservations: Option<Arc<Mutex<BTreeMap<BudgetReservationKey, f64>>>>,
    key: BudgetReservationKey,
    amount_usd: f64,
}

impl BudgetReservation {
    fn none() -> Self {
        Self {
            reservations: None,
            key: BudgetReservationKey::Global,
            amount_usd: 0.0,
        }
    }
}

impl Drop for BudgetReservation {
    fn drop(&mut self) {
        let Some(reservations) = self.reservations.take() else {
            return;
        };
        let Ok(mut reservations) = reservations.lock() else {
            return;
        };
        let Some(current) = reservations.get_mut(&self.key) else {
            return;
        };
        *current -= self.amount_usd;
        if *current <= f64::EPSILON {
            reservations.remove(&self.key);
        }
    }
}

#[derive(Debug, Clone, Eq, PartialEq, Ord, PartialOrd)]
enum BudgetReservationKey {
    Scope(String),
    Run(String),
    Global,
}

impl BudgetReservationKey {
    fn new(budget_scope_id: Option<&str>, run_id: Option<&str>) -> Self {
        if let Some(budget_scope_id) = budget_scope_id {
            Self::Scope(budget_scope_id.to_owned())
        } else if let Some(run_id) = run_id {
            Self::Run(run_id.to_owned())
        } else {
            Self::Global
        }
    }
}

pub fn build_budget_report(config: &BudgetConfig, spend: &[BudgetSpendRecord]) -> BudgetReport {
    let mut by_model = BTreeMap::new();
    let mut by_phase = BTreeMap::new();
    let mut spent_usd = 0.0;
    let mut cache_hits = 0_u64;
    for record in spend {
        spent_usd += record.cost_usd;
        *by_model.entry(record.model.clone()).or_insert(0.0) += record.cost_usd;
        *by_phase
            .entry(record.phase.clone().unwrap_or_else(|| "unknown".to_owned()))
            .or_insert(0.0) += record.cost_usd;
        if record.cache_status == CacheStatus::Hit {
            cache_hits += 1;
        }
    }
    let calls_total = spend.len() as u64;
    BudgetReport {
        hard_cap_usd: config.hard_cap_usd,
        soft_cap_usd: config.soft_cap_usd,
        spent_usd,
        remaining_usd: (config.hard_cap_usd - spent_usd).max(0.0),
        calls_total,
        cache_hit_rate: if calls_total == 0 {
            0.0
        } else {
            cache_hits as f64 / calls_total as f64
        },
        by_model,
        by_phase,
    }
}

fn scoped_spend(
    spend: Vec<BudgetSpendRecord>,
    budget_scope_id: Option<&str>,
    run_id: Option<&str>,
) -> Vec<BudgetSpendRecord> {
    if let Some(budget_scope_id) = budget_scope_id {
        return spend
            .into_iter()
            .filter(|record| {
                record.budget_scope_id.as_deref() == Some(budget_scope_id)
                    || (record.budget_scope_id.is_none()
                        && run_id.is_some()
                        && record.run_id.as_deref() == run_id)
            })
            .collect();
    }
    if let Some(run_id) = run_id {
        return spend
            .into_iter()
            .filter(|record| record.run_id.as_deref() == Some(run_id))
            .collect();
    }
    spend
}

fn append_jsonl<T: Serialize>(path: &PathBuf, value: &T) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }
    let mut raw = serde_json::to_vec(value)?;
    raw.push(b'\n');
    use std::io::Write;
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .with_context(|| format!("failed to open {}", path.display()))?;
    file.write_all(&raw)
        .with_context(|| format!("failed to append {}", path.display()))
}

fn read_jsonl<T: for<'de> Deserialize<'de>>(path: &PathBuf) -> Result<Vec<T>> {
    if !path.exists() {
        return Ok(Vec::new());
    }
    let raw = std::fs::read_to_string(path)
        .with_context(|| format!("failed to read {}", path.display()))?;
    raw.lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| serde_json::from_str(line).context("invalid JSONL record"))
        .collect()
}

fn default_reserve_usd() -> f64 {
    15.0
}

fn default_abort_when_projected_over_hard_cap() -> bool {
    true
}

fn default_warn_when_over_soft_cap() -> bool {
    true
}

fn default_projection_multiplier() -> f64 {
    1.5
}

pub fn budget_event_from_call(record: &ModelCallRecord) -> BudgetSpendRecord {
    BudgetSpendRecord {
        call_id: record.call_id.clone(),
        run_id: record.run_id.clone(),
        budget_scope_id: record.budget_scope_id.clone(),
        phase: record.phase.clone(),
        model: record.model.clone(),
        cost_usd: record.cost.total_usd,
        cache_status: record.cache_status,
        created_at: now_string(),
    }
}
