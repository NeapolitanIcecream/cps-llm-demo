use std::collections::{BTreeMap, BTreeSet};

use anyhow::{Context, Result, anyhow};
use async_trait::async_trait;
use regex::Regex;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};

use crate::effects::{HandlerDecision, HandlerRequest};
use crate::models::EffectHandler;
use crate::program::EffectCall;
use crate::schema::validate_value;

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
pub struct FastPathInput {
    pub event: Value,
    pub rules: Vec<FastPathRule>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
pub struct FastPathRule {
    pub rule_id: String,
    pub when: PredicateExpr,
    pub emit: TemplateExpr,
    pub confidence: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum PredicateExpr {
    And {
        terms: Vec<PredicateExpr>,
    },
    Or {
        terms: Vec<PredicateExpr>,
    },
    Not {
        term: Box<PredicateExpr>,
    },
    FieldExists {
        path: Vec<String>,
    },
    FieldEquals {
        path: Vec<String>,
        value: Value,
    },
    FieldIn {
        path: Vec<String>,
        values: Vec<Value>,
    },
    RegexMatch {
        path: Vec<String>,
        pattern: String,
    },
    JsonSchemaValid {
        schema: Value,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum TemplateExpr {
    Literal {
        value: Value,
    },
    Field {
        path: Vec<String>,
    },
    Object {
        fields: BTreeMap<String, TemplateExpr>,
    },
    Array {
        items: Vec<TemplateExpr>,
    },
    StringTemplate {
        template: String,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
pub struct FastPathOutput {
    pub hit: bool,
    pub rule_id: Option<String>,
    pub value: Option<Value>,
    pub confidence: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
pub struct ValidatorSpec {
    pub validator_id: String,
    pub predicates: Vec<PredicateExpr>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
pub struct ValidatorInput {
    pub value: Value,
    pub validators: Vec<ValidatorSpec>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
pub struct ValidatorOutput {
    pub passed: bool,
    pub failed_validator_ids: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct LocalToolRegistry {
    allowed_tools: BTreeSet<String>,
}

impl LocalToolRegistry {
    pub fn new(tool_names: impl IntoIterator<Item = impl Into<String>>) -> Self {
        Self {
            allowed_tools: tool_names.into_iter().map(Into::into).collect(),
        }
    }

    pub fn with_builtin_tools() -> Self {
        Self::new(["fast_path_apply", "validator_apply"])
    }

    pub fn contains(&self, tool_name: &str) -> bool {
        self.allowed_tools.contains(tool_name)
    }
}

impl Default for LocalToolRegistry {
    fn default() -> Self {
        Self::with_builtin_tools()
    }
}

#[async_trait]
impl EffectHandler for LocalToolRegistry {
    async fn handle(&self, request: HandlerRequest) -> Result<HandlerDecision> {
        let EffectCall::LocalTool { tool_name, .. } = &request.effect else {
            return Err(anyhow!(
                "local tool registry cannot handle effect {}",
                request.effect.kind_name()
            ));
        };
        if !self.contains(tool_name) {
            return Err(anyhow!("local tool {tool_name} is not registered"));
        }

        let value = match tool_name.as_str() {
            "fast_path_apply" => serde_json::to_value(apply_fast_path(request.input)?)?,
            "validator_apply" => serde_json::to_value(apply_validators(request.input)?)?,
            _ => return Err(anyhow!("local tool {tool_name} is not implemented")),
        };

        Ok(HandlerDecision::ReturnValue {
            value,
            confidence: 1.0,
            rationale: format!("{tool_name} completed"),
        })
    }
}

pub fn apply_fast_path(input: Value) -> Result<FastPathOutput> {
    let input: FastPathInput = serde_json::from_value(input).context("invalid fast path input")?;
    for rule in input.rules {
        if predicate_matches(&input.event, &rule.when)? {
            ensure_probability(rule.confidence, "fast path rule confidence")?;
            return Ok(FastPathOutput {
                hit: true,
                rule_id: Some(rule.rule_id),
                value: Some(render_template(&input.event, &rule.emit)?),
                confidence: rule.confidence,
            });
        }
    }

    Ok(FastPathOutput {
        hit: false,
        rule_id: None,
        value: None,
        confidence: 0.0,
    })
}

pub fn apply_validators(input: Value) -> Result<ValidatorOutput> {
    let input: ValidatorInput = serde_json::from_value(input).context("invalid validator input")?;
    let mut failed_validator_ids = Vec::new();

    for validator in input.validators {
        for predicate in &validator.predicates {
            if !predicate_matches(&input.value, predicate)? {
                failed_validator_ids.push(validator.validator_id.clone());
                break;
            }
        }
    }

    Ok(ValidatorOutput {
        passed: failed_validator_ids.is_empty(),
        failed_validator_ids,
    })
}

fn predicate_matches(root: &Value, predicate: &PredicateExpr) -> Result<bool> {
    match predicate {
        PredicateExpr::And { terms } => {
            for term in terms {
                if !predicate_matches(root, term)? {
                    return Ok(false);
                }
            }
            Ok(true)
        }
        PredicateExpr::Or { terms } => {
            for term in terms {
                if predicate_matches(root, term)? {
                    return Ok(true);
                }
            }
            Ok(false)
        }
        PredicateExpr::Not { term } => Ok(!predicate_matches(root, term)?),
        PredicateExpr::FieldExists { path } => Ok(value_at_path(root, path).is_some()),
        PredicateExpr::FieldEquals { path, value } => {
            Ok(value_at_path(root, path).is_some_and(|actual| actual == value))
        }
        PredicateExpr::FieldIn { path, values } => Ok(value_at_path(root, path)
            .is_some_and(|actual| values.iter().any(|candidate| candidate == actual))),
        PredicateExpr::RegexMatch { path, pattern } => {
            let Some(text) = value_at_path(root, path).and_then(Value::as_str) else {
                return Ok(false);
            };
            Ok(Regex::new(pattern)
                .with_context(|| format!("invalid regex pattern {pattern:?}"))?
                .is_match(text))
        }
        PredicateExpr::JsonSchemaValid { schema } => Ok(validate_value(schema, root).is_ok()),
    }
}

fn render_template(root: &Value, template: &TemplateExpr) -> Result<Value> {
    match template {
        TemplateExpr::Literal { value } => Ok(value.clone()),
        TemplateExpr::Field { path } => value_at_path(root, path)
            .cloned()
            .ok_or_else(|| anyhow!("template field path {path:?} was not present")),
        TemplateExpr::Object { fields } => {
            let mut object = Map::new();
            for (key, value) in fields {
                object.insert(key.clone(), render_template(root, value)?);
            }
            Ok(Value::Object(object))
        }
        TemplateExpr::Array { items } => {
            let mut rendered = Vec::with_capacity(items.len());
            for item in items {
                rendered.push(render_template(root, item)?);
            }
            Ok(Value::Array(rendered))
        }
        TemplateExpr::StringTemplate { template } => {
            Ok(Value::String(render_string_template(root, template)?))
        }
    }
}

fn render_string_template(root: &Value, template: &str) -> Result<String> {
    let mut rendered = String::with_capacity(template.len());
    let mut cursor = 0;

    while let Some(open_offset) = template[cursor..].find('{') {
        let open = cursor + open_offset;
        rendered.push_str(&template[cursor..open]);
        let close = template[open + 1..]
            .find('}')
            .map(|offset| open + 1 + offset)
            .ok_or_else(|| anyhow!("string template has an unclosed field"))?;
        let name = template[open + 1..close].trim();
        if name.is_empty() {
            return Err(anyhow!("string template contains an empty field"));
        }
        let replacement = value_at_path(root, &[name.to_owned()])
            .ok_or_else(|| anyhow!("string template field {name} was not present"))?;
        rendered.push_str(value_to_template_string(replacement).as_str());
        cursor = close + 1;
    }

    rendered.push_str(&template[cursor..]);
    Ok(rendered)
}

fn value_to_template_string(value: &Value) -> String {
    match value {
        Value::String(value) => value.clone(),
        Value::Null => "null".to_owned(),
        Value::Bool(value) => value.to_string(),
        Value::Number(value) => value.to_string(),
        Value::Array(_) | Value::Object(_) => serde_json::to_string(value).unwrap_or_default(),
    }
}

fn value_at_path<'a>(root: &'a Value, path: &[String]) -> Option<&'a Value> {
    let mut cursor = root;
    for segment in path {
        cursor = cursor.as_object()?.get(segment)?;
    }
    Some(cursor)
}

fn ensure_probability(value: f32, name: &str) -> Result<()> {
    if value.is_finite() && (0.0..=1.0).contains(&value) {
        Ok(())
    } else {
        Err(anyhow!("{name} must be a finite probability"))
    }
}

pub fn json_schema() -> Value {
    json!({})
}
