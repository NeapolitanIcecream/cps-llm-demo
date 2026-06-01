use anyhow::{Context, Result};
use secrecy::SecretString;
use url::Url;

use crate::error::DemoError;
use crate::responses_client::{ResponsesClient, ResponsesClientConfig};

pub const DEFAULT_BASE_URL: &str = "https://api.openai.com/v1";
pub const DEFAULT_WEAK_MODEL: &str = "gpt-5.4-mini";
pub const DEFAULT_STRONG_MODEL: &str = "gpt-5.5";
pub const DEFAULT_THRESHOLD: f32 = 0.75;

#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub base_url: Url,
    pub api_key: SecretString,
    pub weak_model: String,
    pub strong_model: String,
    pub threshold: f32,
}

impl ModelConfig {
    pub fn new(
        base_url: String,
        api_key: Option<String>,
        weak_model: String,
        strong_model: String,
        threshold: f32,
    ) -> Result<Self> {
        let api_key = api_key
            .filter(|value| !value.trim().is_empty())
            .ok_or(DemoError::MissingApiKey)?;

        Ok(Self {
            base_url: Url::parse(&base_url).context("invalid base URL")?,
            api_key: SecretString::from(api_key),
            weak_model,
            strong_model,
            threshold,
        })
    }

    pub fn responses_client(&self) -> ResponsesClient {
        ResponsesClient::new(ResponsesClientConfig {
            base_url: self.base_url.clone(),
            api_key: self.api_key.clone(),
        })
    }
}
