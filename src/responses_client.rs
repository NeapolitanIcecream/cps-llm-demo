use anyhow::{Context, Result};
use secrecy::{ExposeSecret, SecretString};
use serde::de::DeserializeOwned;
use serde_json::{Value, json};
use url::Url;

use crate::error::model_protocol_error;
use crate::schema::validate_value;

#[derive(Debug, Clone)]
pub struct ResponsesClientConfig {
    pub base_url: Url,
    pub api_key: SecretString,
}

#[derive(Clone)]
pub struct ResponsesClient {
    http: reqwest::Client,
    config: ResponsesClientConfig,
}

impl ResponsesClient {
    pub fn new(config: ResponsesClientConfig) -> Self {
        Self {
            http: reqwest::Client::new(),
            config,
        }
    }

    pub fn responses_url(&self) -> Result<Url> {
        let mut url = self.config.base_url.clone();
        let base_path = url.path().trim_end_matches('/');
        url.set_path(&format!("{base_path}/responses"));
        url.set_query(None);
        Ok(url)
    }

    pub async fn create_structured<T>(
        &self,
        model: &str,
        instructions: &str,
        input_json: &Value,
        schema_name: &str,
        schema: Value,
    ) -> Result<T>
    where
        T: DeserializeOwned,
    {
        let request_schema = schema.clone();
        let body = json!({
            "model": model,
            "instructions": instructions,
            "input": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": serde_json::to_string_pretty(input_json)?,
                        }
                    ]
                }
            ],
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": schema_name,
                    "schema": schema,
                    "strict": true
                },
                "verbosity": "low"
            },
            "store": false
        });

        let request_id = uuid::Uuid::new_v4().to_string();
        let response = self
            .http
            .post(self.responses_url()?)
            .bearer_auth(self.config.api_key.expose_secret())
            .header("X-Client-Request-Id", request_id)
            .json(&body)
            .send()
            .await
            .context("failed to call Responses API")?;

        let status = response.status();
        let value: Value = response
            .json()
            .await
            .context("Responses API returned non-JSON response")?;

        if !status.is_success() {
            return Err(anyhow::anyhow!(
                "Responses API error: status={status}, body={value}"
            ));
        }

        let output_text = extract_output_text(&value)?;
        let output_value: Value =
            serde_json::from_str(&output_text).context("model output was not valid JSON")?;
        validate_value(&request_schema, &output_value)?;
        let parsed: T = serde_json::from_value(output_value)
            .context("model output did not match target type")?;
        Ok(parsed)
    }
}

pub fn extract_output_text(response_json: &Value) -> Result<String> {
    if let Some(text) = response_json.get("output_text").and_then(Value::as_str) {
        return Ok(text.to_owned());
    }

    let mut chunks = Vec::new();
    if let Some(output) = response_json.get("output").and_then(Value::as_array) {
        for item in output {
            if item.get("type").and_then(Value::as_str) != Some("message") {
                continue;
            }

            let Some(content) = item.get("content").and_then(Value::as_array) else {
                continue;
            };

            for content_item in content {
                if content_item.get("type").and_then(Value::as_str) == Some("output_text") {
                    if let Some(text) = content_item.get("text").and_then(Value::as_str) {
                        chunks.push(text);
                    }
                }
            }
        }
    }

    if chunks.is_empty() {
        return Err(model_protocol_error("missing output_text in Responses API payload").into());
    }

    Ok(chunks.join(""))
}
