use std::collections::BTreeMap;

use anyhow::{Context, Result, anyhow};
use async_trait::async_trait;
use regex::Regex;
use schemars::{JsonSchema, schema_for};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};

use crate::schema::validate_value;
use crate::value_demo::local_tools::LocalTool;

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
pub struct FastPathSpec {
    pub fast_path_id: String,
    pub predicates: Vec<PredicateSpec>,
    pub output_template: TemplateExpr,
    pub output_schema: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum PredicateSpec {
    JsonPathExists { path: Vec<String> },
    JsonPathEquals { path: Vec<String>, value: Value },
    JsonSchemaValid { path: Vec<String>, schema: Value },
    RegexMatch { path: Vec<String>, pattern: String },
    And { predicates: Vec<PredicateSpec> },
    Or { predicates: Vec<PredicateSpec> },
    Not { predicate: Box<PredicateSpec> },
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum TemplateExpr {
    Literal {
        value: Value,
    },
    FromPath {
        path: Vec<String>,
    },
    Object {
        fields: BTreeMap<String, TemplateExpr>,
    },
    Array {
        items: Vec<TemplateExpr>,
    },
    Concat {
        parts: Vec<TemplateExpr>,
    },
    Null,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
pub struct FastPathApplyInput {
    pub payload: Value,
    pub fast_paths: Vec<FastPathSpec>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
pub struct FastPathApplyOutput {
    pub hit: bool,
    pub fast_path_id: Option<String>,
    pub value: Option<Value>,
}

pub struct FastPathApplyTool;

#[async_trait]
impl LocalTool for FastPathApplyTool {
    fn name(&self) -> &'static str {
        "fast_path_apply"
    }

    fn input_schema(&self) -> Value {
        schema_value(schema_for!(FastPathApplyInput))
    }

    fn output_schema(&self) -> Value {
        schema_value(schema_for!(FastPathApplyOutput))
    }

    async fn call(&self, input: Value) -> Result<Value> {
        let input: FastPathApplyInput =
            serde_json::from_value(input).context("invalid fast_path_apply input")?;
        let output = apply_fast_paths(input)?;
        Ok(serde_json::to_value(output)?)
    }
}

pub fn apply_fast_paths(input: FastPathApplyInput) -> Result<FastPathApplyOutput> {
    for spec in input.fast_paths {
        validate_fast_path_spec(&spec)
            .with_context(|| format!("invalid fast path {}", spec.fast_path_id))?;
        if predicates_match(&input.payload, &spec.predicates)? {
            let value = render_template(&input.payload, &spec.output_template)
                .with_context(|| format!("failed to render fast path {}", spec.fast_path_id))?;
            validate_value(&spec.output_schema, &value).with_context(|| {
                format!(
                    "fast path {} rendered output failed output_schema",
                    spec.fast_path_id
                )
            })?;
            return Ok(FastPathApplyOutput {
                hit: true,
                fast_path_id: Some(spec.fast_path_id),
                value: Some(value),
            });
        }
    }

    Ok(FastPathApplyOutput {
        hit: false,
        fast_path_id: None,
        value: None,
    })
}

pub fn validate_fast_path_spec(spec: &FastPathSpec) -> Result<()> {
    jsonschema::validator_for(&spec.output_schema)
        .context("fast path output_schema is not a valid JSON schema")?;
    validate_predicates(&spec.predicates)
}

fn validate_predicates(predicates: &[PredicateSpec]) -> Result<()> {
    for predicate in predicates {
        match predicate {
            PredicateSpec::RegexMatch { pattern, .. } => {
                Regex::new(pattern)
                    .with_context(|| format!("invalid regex pattern {pattern:?}"))?;
            }
            PredicateSpec::JsonSchemaValid { schema, .. } => {
                jsonschema::validator_for(schema)
                    .context("predicate schema is not a valid JSON schema")?;
            }
            PredicateSpec::And { predicates } | PredicateSpec::Or { predicates } => {
                validate_predicates(predicates)?;
            }
            PredicateSpec::Not { predicate } => {
                validate_predicates(std::slice::from_ref(predicate.as_ref()))?;
            }
            PredicateSpec::JsonPathExists { .. } | PredicateSpec::JsonPathEquals { .. } => {}
        }
    }
    Ok(())
}

fn predicates_match(payload: &Value, predicates: &[PredicateSpec]) -> Result<bool> {
    for predicate in predicates {
        if !predicate_matches(payload, predicate)? {
            return Ok(false);
        }
    }
    Ok(true)
}

fn predicate_matches(payload: &Value, predicate: &PredicateSpec) -> Result<bool> {
    match predicate {
        PredicateSpec::JsonPathExists { path } => Ok(json_path(payload, path).is_some()),
        PredicateSpec::JsonPathEquals { path, value } => {
            Ok(json_path(payload, path).is_some_and(|actual| actual == value))
        }
        PredicateSpec::JsonSchemaValid { path, schema } => {
            let Some(value) = json_path(payload, path) else {
                return Ok(false);
            };
            Ok(validate_value(schema, value).is_ok())
        }
        PredicateSpec::RegexMatch { path, pattern } => {
            let Some(value) = json_path(payload, path).and_then(Value::as_str) else {
                return Ok(false);
            };
            Ok(Regex::new(pattern)?.is_match(value))
        }
        PredicateSpec::And { predicates } => predicates_match(payload, predicates),
        PredicateSpec::Or { predicates } => {
            for predicate in predicates {
                if predicate_matches(payload, predicate)? {
                    return Ok(true);
                }
            }
            Ok(false)
        }
        PredicateSpec::Not { predicate } => Ok(!predicate_matches(payload, predicate)?),
    }
}

fn render_template(payload: &Value, template: &TemplateExpr) -> Result<Value> {
    match template {
        TemplateExpr::Literal { value } => Ok(value.clone()),
        TemplateExpr::FromPath { path } => json_path(payload, path)
            .cloned()
            .ok_or_else(|| anyhow!("template path {:?} was not present", path)),
        TemplateExpr::Object { fields } => {
            let mut object = Map::new();
            for (name, value) in fields {
                object.insert(name.clone(), render_template(payload, value)?);
            }
            Ok(Value::Object(object))
        }
        TemplateExpr::Array { items } => {
            let mut values = Vec::with_capacity(items.len());
            for item in items {
                values.push(render_template(payload, item)?);
            }
            Ok(Value::Array(values))
        }
        TemplateExpr::Concat { parts } => {
            let mut rendered = String::new();
            for part in parts {
                let value = render_template(payload, part)?;
                match value {
                    Value::String(text) => rendered.push_str(&text),
                    Value::Null => {}
                    other => rendered.push_str(&other.to_string()),
                }
            }
            Ok(Value::String(rendered))
        }
        TemplateExpr::Null => Ok(Value::Null),
    }
}

fn json_path<'a>(value: &'a Value, path: &[String]) -> Option<&'a Value> {
    let mut cursor = value;
    for segment in path {
        cursor = cursor.as_object()?.get(segment)?;
    }
    Some(cursor)
}

fn schema_value(schema: impl Serialize) -> Value {
    let mut value = serde_json::to_value(schema).expect("schemars schema should serialize");
    strip_schema_titles(&mut value);
    value
}

fn strip_schema_titles(value: &mut Value) {
    match value {
        Value::Object(map) => {
            map.remove("$schema");
            map.remove("title");
            for item in map.values_mut() {
                strip_schema_titles(item);
            }
        }
        Value::Array(items) => {
            for item in items {
                strip_schema_titles(item);
            }
        }
        Value::Null | Value::Bool(_) | Value::Number(_) | Value::String(_) => {}
    }
}

pub fn fast_path_apply_input_schema() -> Value {
    json!({
        "type": "object",
        "additionalProperties": false,
        "required": ["payload", "fast_paths"],
        "properties": {
            "payload": {},
            "fast_paths": { "type": "array" }
        }
    })
}
