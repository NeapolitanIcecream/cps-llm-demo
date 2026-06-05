use std::collections::BTreeMap;
use std::path::Path;

use anyhow::{Context, Result, anyhow};
use serde::{Deserialize, Serialize};

use crate::store::model_call_store::{CostBreakdown, ModelUsage};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PriceCatalog {
    pub prices_per_1m_tokens: BTreeMap<String, ModelPrice>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ModelPrice {
    pub input: f64,
    pub cached_input: f64,
    pub output: f64,
}

impl PriceCatalog {
    pub fn default_openai() -> Self {
        let mut prices = BTreeMap::new();
        prices.insert(
            "gpt-5.5".to_owned(),
            ModelPrice {
                input: 5.00,
                cached_input: 0.50,
                output: 30.00,
            },
        );
        prices.insert(
            "gpt-5.4-mini".to_owned(),
            ModelPrice {
                input: 0.75,
                cached_input: 0.075,
                output: 4.50,
            },
        );
        Self {
            prices_per_1m_tokens: prices,
        }
    }

    pub fn load(path: &Path) -> Result<Self> {
        let raw = std::fs::read_to_string(path)
            .with_context(|| format!("failed to read price catalog {}", path.display()))?;
        serde_yaml::from_str(&raw)
            .with_context(|| format!("invalid price catalog YAML in {}", path.display()))
    }

    pub fn write_yaml(&self, path: &Path) -> Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("failed to create {}", parent.display()))?;
        }
        let raw = serde_yaml::to_string(self)?;
        std::fs::write(path, raw)
            .with_context(|| format!("failed to write price catalog {}", path.display()))
    }

    pub fn require_model(&self, model: &str) -> Result<()> {
        self.model_price(model).map(|_| ())
    }

    pub fn estimate_cost(
        &self,
        model: &str,
        usage: &ModelUsage,
        estimated: bool,
    ) -> Result<CostBreakdown> {
        let price = self.model_price(model)?;
        let billable_input = usage.input_tokens.saturating_sub(usage.cached_input_tokens);
        let input_usd = billable_input as f64 / 1_000_000.0 * price.input;
        let cached_input_usd = usage.cached_input_tokens as f64 / 1_000_000.0 * price.cached_input;
        let output_usd = usage.output_tokens as f64 / 1_000_000.0 * price.output;
        Ok(CostBreakdown {
            input_usd,
            cached_input_usd,
            output_usd,
            total_usd: input_usd + cached_input_usd + output_usd,
            estimated,
        })
    }

    fn model_price(&self, model: &str) -> Result<&ModelPrice> {
        self.prices_per_1m_tokens
            .get(model)
            .ok_or_else(|| anyhow!("model {model} is missing from price catalog"))
    }
}
