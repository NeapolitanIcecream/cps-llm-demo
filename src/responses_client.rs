use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context, Result};
use secrecy::{ExposeSecret, SecretString};
use serde::de::DeserializeOwned;
use serde_json::{Value, json};
use url::Url;

use crate::error::model_protocol_error;
use crate::model_cache::{
    ModelCache, ModelCacheEntryParts, ModelCacheMode, cache_entry_from_response, model_config_hash,
    prompt_hash, schema_hash, structured_cache_key,
};
use crate::pricing::price_catalog::PriceCatalog;
use crate::schema::validate_value;
use crate::store::budget_store::{BudgetConfig, BudgetGuard, FileBudgetStore};
use crate::store::model_call_store::{
    CacheStatus, CostBreakdown, FileModelCallStore, ModelCallRecord, ModelUsage,
    estimate_usage_from_bytes,
};
use crate::store::state_dir::{now_string, stable_hash_value};

#[derive(Debug, Clone)]
pub struct ResponsesClientConfig {
    pub base_url: Url,
    pub api_key: SecretString,
    pub runtime: Option<Arc<ModelCallRuntime>>,
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
        Ok(self
            .create_structured_with_context(
                model,
                instructions,
                input_json,
                schema_name,
                schema,
                ModelCallContext::unknown(),
            )
            .await?
            .parsed)
    }

    pub async fn create_structured_with_context<T>(
        &self,
        model: &str,
        instructions: &str,
        input_json: &Value,
        schema_name: &str,
        schema: Value,
        context: ModelCallContext,
    ) -> Result<StructuredResponse<T>>
    where
        T: DeserializeOwned,
    {
        self.create_structured_internal(
            model,
            instructions,
            input_json,
            schema_name,
            schema,
            context,
        )
        .await
    }
}

impl ResponsesClient {
    async fn create_structured_internal<T>(
        &self,
        model: &str,
        instructions: &str,
        input_json: &Value,
        schema_name: &str,
        schema: Value,
        context: ModelCallContext,
    ) -> Result<StructuredResponse<T>>
    where
        T: DeserializeOwned,
    {
        let request_schema = schema.clone();
        let request_options = json!({
            "schema_name": schema_name,
            "strict": true,
            "verbosity": "low",
            "store": false,
        });
        let body = structured_request_body(model, instructions, input_json, schema_name, schema)?;
        let body_bytes = serde_json::to_vec(&body)?;
        let base_url = self.config.base_url.to_string();
        let request_hash = structured_cache_key(
            &base_url,
            model,
            instructions,
            input_json,
            &request_schema,
            &request_options,
        )?;
        let prompt_hash = prompt_hash(instructions);
        let schema_hash = schema_hash(&request_schema)?;
        let model_config_hash = model_config_hash(&base_url, model)?;

        if let Some(runtime) = &self.config.runtime {
            if let Some(cache) = &runtime.cache {
                if let Some(entry) = cache.get(&request_hash)? {
                    let output_value = entry.parsed_output.clone();
                    validate_value(&request_schema, &output_value)?;
                    let parsed = serde_json::from_value(output_value.clone())
                        .context("cached model output did not match target type")?;
                    let metadata = StructuredCallMetadata {
                        call_id: uuid::Uuid::new_v4().to_string(),
                        request_hash: request_hash.clone(),
                        prompt_hash: prompt_hash.clone(),
                        schema_hash: schema_hash.clone(),
                        model_config_hash: model_config_hash.clone(),
                        input_bytes: body_bytes.len() as u64,
                        output_bytes: serde_json::to_vec(&entry.raw_response)?.len() as u64,
                        usage: entry.usage.clone(),
                        estimated_usage: entry.usage.clone().unwrap_or_else(|| {
                            estimate_usage_from_bytes(body_bytes.len() as u64, 0)
                        }),
                        cost: CostBreakdown::zero(false),
                        latency_ms: 0,
                        cache_status: CacheStatus::Hit,
                    };
                    runtime.record_call(model, &context, &metadata, true, None)?;
                    return Ok(StructuredResponse {
                        parsed,
                        output_value,
                        raw_response: entry.raw_response,
                        metadata,
                    });
                }
            }
            if let Some(guard) = &runtime.budget_guard {
                if let Err(err) = guard.ensure_call_allowed(model, body_bytes.len() as u64, None) {
                    let metadata = StructuredCallMetadata {
                        call_id: uuid::Uuid::new_v4().to_string(),
                        request_hash: request_hash.clone(),
                        prompt_hash: prompt_hash.clone(),
                        schema_hash: schema_hash.clone(),
                        model_config_hash: model_config_hash.clone(),
                        input_bytes: body_bytes.len() as u64,
                        output_bytes: 0,
                        usage: None,
                        estimated_usage: estimate_usage_from_bytes(body_bytes.len() as u64, 0),
                        cost: CostBreakdown::zero(true),
                        latency_ms: 0,
                        cache_status: cache_status_for_runtime(runtime),
                    };
                    runtime.record_call(
                        model,
                        &context,
                        &metadata,
                        false,
                        Some(err.to_string()),
                    )?;
                    return Err(err);
                }
            }
        }

        let request_id = uuid::Uuid::new_v4().to_string();
        let start = Instant::now();
        let response_result = self
            .http
            .post(self.responses_url()?)
            .bearer_auth(self.config.api_key.expose_secret())
            .header("X-Client-Request-Id", request_id)
            .json(&body)
            .send()
            .await;
        let latency_ms = start.elapsed().as_millis() as u64;

        let response = match response_result {
            Ok(response) => response,
            Err(err) => {
                let metadata = failure_metadata(
                    &request_hash,
                    &prompt_hash,
                    &schema_hash,
                    &model_config_hash,
                    body_bytes.len() as u64,
                    latency_ms,
                    cache_status_for_optional_runtime(self.config.runtime.as_deref()),
                );
                if let Some(runtime) = &self.config.runtime {
                    runtime.record_call(
                        model,
                        &context,
                        &metadata,
                        false,
                        Some(err.to_string()),
                    )?;
                }
                return Err(err).context("failed to call Responses API");
            }
        };

        let status = response.status();
        let value_result: Result<Value> = response
            .json()
            .await
            .context("Responses API returned non-JSON response");

        let value = match value_result {
            Ok(value) => value,
            Err(err) => {
                let metadata = failure_metadata(
                    &request_hash,
                    &prompt_hash,
                    &schema_hash,
                    &model_config_hash,
                    body_bytes.len() as u64,
                    latency_ms,
                    cache_status_for_optional_runtime(self.config.runtime.as_deref()),
                );
                if let Some(runtime) = &self.config.runtime {
                    runtime.record_call(
                        model,
                        &context,
                        &metadata,
                        false,
                        Some(err.to_string()),
                    )?;
                }
                return Err(err);
            }
        };

        let output_bytes = serde_json::to_vec(&value)?.len() as u64;
        let usage = extract_usage(&value);
        let estimated_usage = usage
            .clone()
            .unwrap_or_else(|| estimate_usage_from_bytes(body_bytes.len() as u64, output_bytes));
        let cost = cost_for_runtime(
            self.config.runtime.as_deref(),
            model,
            &usage,
            &estimated_usage,
        )?;
        let cache_status = cache_status_for_optional_runtime(self.config.runtime.as_deref());
        let metadata = StructuredCallMetadata {
            call_id: uuid::Uuid::new_v4().to_string(),
            request_hash,
            prompt_hash,
            schema_hash,
            model_config_hash,
            input_bytes: body_bytes.len() as u64,
            output_bytes,
            usage,
            estimated_usage,
            cost,
            latency_ms,
            cache_status,
        };

        if !status.is_success() {
            if let Some(runtime) = &self.config.runtime {
                runtime.record_call(
                    model,
                    &context,
                    &metadata,
                    false,
                    Some(format!(
                        "Responses API error: status={status}, body={value}"
                    )),
                )?;
            }
            return Err(anyhow::anyhow!(
                "Responses API error: status={status}, body={value}"
            ));
        }

        let output_parse = parse_structured_output::<T>(&request_schema, &value);
        match output_parse {
            Ok((parsed, output_value)) => {
                if let Some(runtime) = &self.config.runtime {
                    if let Some(cache) = &runtime.cache {
                        cache.put(&cache_entry_from_response(ModelCacheEntryParts {
                            key: metadata.request_hash.clone(),
                            model: model.to_owned(),
                            prompt_hash: metadata.prompt_hash.clone(),
                            schema_hash: metadata.schema_hash.clone(),
                            model_config_hash: metadata.model_config_hash.clone(),
                            parsed_output: output_value.clone(),
                            raw_response: value.clone(),
                            usage: metadata.usage.clone(),
                        }))?;
                    }
                    runtime.record_call(model, &context, &metadata, true, None)?;
                }
                Ok(StructuredResponse {
                    parsed,
                    output_value,
                    raw_response: value,
                    metadata,
                })
            }
            Err(err) => {
                if let Some(runtime) = &self.config.runtime {
                    runtime.record_call(
                        model,
                        &context,
                        &metadata,
                        false,
                        Some(err.to_string()),
                    )?;
                }
                Err(err)
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct ModelCallRuntime {
    pub call_store: FileModelCallStore,
    pub budget_store: Option<FileBudgetStore>,
    pub budget_guard: Option<BudgetGuard>,
    pub price_catalog: PriceCatalog,
    pub cache: Option<ModelCache>,
}

impl ModelCallRuntime {
    pub fn new(call_store: FileModelCallStore, price_catalog: PriceCatalog) -> Self {
        Self {
            call_store,
            budget_store: None,
            budget_guard: None,
            price_catalog,
            cache: None,
        }
    }

    pub fn with_budget(
        mut self,
        budget_store: FileBudgetStore,
        budget_config: BudgetConfig,
    ) -> Self {
        self.budget_guard = Some(BudgetGuard::new(
            budget_config,
            self.price_catalog.clone(),
            budget_store.clone(),
        ));
        self.budget_store = Some(budget_store);
        self
    }

    pub fn with_cache(mut self, cache: ModelCache) -> Self {
        self.cache = Some(cache);
        self
    }

    fn record_call(
        &self,
        model: &str,
        context: &ModelCallContext,
        metadata: &StructuredCallMetadata,
        success: bool,
        error: Option<String>,
    ) -> Result<()> {
        let record = ModelCallRecord {
            call_id: metadata.call_id.clone(),
            run_id: context.run_id.clone(),
            workflow_id: context.workflow_id.clone(),
            event_id: context.event_id.clone(),
            model: model.to_owned(),
            handler: context.handler.clone(),
            effect_kind: context.effect_kind.clone(),
            task_name: context.task_name.clone(),
            phase: context.phase.clone(),
            request_hash: metadata.request_hash.clone(),
            prompt_hash: metadata.prompt_hash.clone(),
            schema_hash: metadata.schema_hash.clone(),
            model_config_hash: metadata.model_config_hash.clone(),
            input_bytes: metadata.input_bytes,
            output_bytes: metadata.output_bytes,
            usage: metadata.usage.clone(),
            estimated_usage: metadata.estimated_usage.clone(),
            cost: metadata.cost.clone(),
            latency_ms: metadata.latency_ms,
            cache_status: metadata.cache_status,
            success,
            error,
            created_at: now_string(),
        };
        self.call_store.append(&record)?;
        if let Some(budget_store) = &self.budget_store {
            budget_store.record_model_call(&record)?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Default)]
pub struct ModelCallContext {
    pub run_id: Option<String>,
    pub workflow_id: Option<String>,
    pub event_id: Option<String>,
    pub handler: String,
    pub effect_kind: String,
    pub task_name: Option<String>,
    pub phase: Option<String>,
}

impl ModelCallContext {
    pub fn unknown() -> Self {
        Self {
            handler: "unknown".to_owned(),
            effect_kind: "unknown".to_owned(),
            ..Self::default()
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct StructuredCallMetadata {
    pub call_id: String,
    pub request_hash: String,
    pub prompt_hash: String,
    pub schema_hash: String,
    pub model_config_hash: String,
    pub input_bytes: u64,
    pub output_bytes: u64,
    pub usage: Option<ModelUsage>,
    pub estimated_usage: ModelUsage,
    pub cost: CostBreakdown,
    pub latency_ms: u64,
    pub cache_status: CacheStatus,
}

#[derive(Debug, Clone)]
pub struct StructuredResponse<T> {
    pub parsed: T,
    pub output_value: Value,
    pub raw_response: Value,
    pub metadata: StructuredCallMetadata,
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

pub fn extract_usage(response_json: &Value) -> Option<ModelUsage> {
    let usage = response_json.get("usage")?;
    let input_tokens = usage
        .get("input_tokens")
        .or_else(|| usage.get("prompt_tokens"))
        .and_then(Value::as_u64)
        .unwrap_or_default();
    let output_tokens = usage
        .get("output_tokens")
        .or_else(|| usage.get("completion_tokens"))
        .and_then(Value::as_u64)
        .unwrap_or_default();
    let cached_input_tokens = usage
        .get("cached_input_tokens")
        .and_then(Value::as_u64)
        .or_else(|| {
            usage
                .get("input_tokens_details")
                .and_then(|details| details.get("cached_tokens"))
                .and_then(Value::as_u64)
        })
        .unwrap_or_default();
    let reasoning_tokens = usage
        .get("reasoning_tokens")
        .and_then(Value::as_u64)
        .or_else(|| {
            usage
                .get("output_tokens_details")
                .and_then(|details| details.get("reasoning_tokens"))
                .and_then(Value::as_u64)
        })
        .unwrap_or_default();
    let total_tokens = usage
        .get("total_tokens")
        .and_then(Value::as_u64)
        .unwrap_or(input_tokens + output_tokens);

    Some(ModelUsage {
        input_tokens,
        cached_input_tokens,
        output_tokens,
        reasoning_tokens,
        total_tokens,
    })
}

fn structured_request_body(
    model: &str,
    instructions: &str,
    input_json: &Value,
    schema_name: &str,
    schema: Value,
) -> Result<Value> {
    Ok(json!({
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
    }))
}

fn parse_structured_output<T>(request_schema: &Value, response_json: &Value) -> Result<(T, Value)>
where
    T: DeserializeOwned,
{
    let output_text = extract_output_text(response_json)?;
    let output_value: Value =
        serde_json::from_str(&output_text).context("model output was not valid JSON")?;
    validate_value(request_schema, &output_value)?;
    let parsed: T = serde_json::from_value(output_value.clone())
        .context("model output did not match target type")?;
    Ok((parsed, output_value))
}

fn failure_metadata(
    request_hash: &str,
    prompt_hash: &str,
    schema_hash: &str,
    model_config_hash: &str,
    input_bytes: u64,
    latency_ms: u64,
    cache_status: CacheStatus,
) -> StructuredCallMetadata {
    StructuredCallMetadata {
        call_id: uuid::Uuid::new_v4().to_string(),
        request_hash: request_hash.to_owned(),
        prompt_hash: prompt_hash.to_owned(),
        schema_hash: schema_hash.to_owned(),
        model_config_hash: model_config_hash.to_owned(),
        input_bytes,
        output_bytes: 0,
        usage: None,
        estimated_usage: estimate_usage_from_bytes(input_bytes, 0),
        cost: CostBreakdown::zero(true),
        latency_ms,
        cache_status,
    }
}

fn cache_status_for_optional_runtime(runtime: Option<&ModelCallRuntime>) -> CacheStatus {
    runtime
        .map(cache_status_for_runtime)
        .unwrap_or(CacheStatus::Bypass)
}

fn cache_status_for_runtime(runtime: &ModelCallRuntime) -> CacheStatus {
    match runtime.cache.as_ref().map(ModelCache::mode) {
        Some(ModelCacheMode::Refresh) => CacheStatus::Refresh,
        Some(ModelCacheMode::ReadWrite | ModelCacheMode::ReadOnly) => CacheStatus::Miss,
        Some(ModelCacheMode::Disabled) | None => CacheStatus::Bypass,
    }
}

fn cost_for_runtime(
    runtime: Option<&ModelCallRuntime>,
    model: &str,
    usage: &Option<ModelUsage>,
    estimated_usage: &ModelUsage,
) -> Result<CostBreakdown> {
    let catalog = runtime
        .map(|runtime| runtime.price_catalog.clone())
        .unwrap_or_else(PriceCatalog::default_openai);
    match usage {
        Some(usage) => catalog.estimate_cost(model, usage, false),
        None => catalog.estimate_cost(model, estimated_usage, true),
    }
    .or_else(|_| Ok(CostBreakdown::zero(true)))
}

pub fn request_hash_for_debug(value: &Value) -> Result<String> {
    stable_hash_value(value)
}
