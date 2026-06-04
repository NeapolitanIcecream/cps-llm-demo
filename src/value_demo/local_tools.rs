use std::collections::BTreeMap;
use std::sync::Arc;

use anyhow::{Result, anyhow};
use async_trait::async_trait;
use serde_json::Value;

use crate::value_demo::fast_path::FastPathApplyTool;

#[async_trait]
pub trait LocalTool: Send + Sync {
    fn name(&self) -> &'static str;
    fn input_schema(&self) -> Value;
    fn output_schema(&self) -> Value;
    async fn call(&self, input: Value) -> Result<Value>;
}

#[derive(Clone, Default)]
pub struct LocalToolRegistry {
    tools: BTreeMap<String, Arc<dyn LocalTool>>,
}

impl LocalToolRegistry {
    pub fn empty() -> Self {
        Self {
            tools: BTreeMap::new(),
        }
    }

    pub fn with_default_tools() -> Self {
        let mut registry = Self::empty();
        registry.register(Arc::new(FastPathApplyTool));
        registry
    }

    pub fn register(&mut self, tool: Arc<dyn LocalTool>) {
        self.tools.insert(tool.name().to_owned(), tool);
    }

    pub async fn call(&self, tool_name: &str, input: Value) -> Result<Value> {
        let tool = self
            .tools
            .get(tool_name)
            .ok_or_else(|| anyhow!("local tool {tool_name} is not registered"))?;
        tool.call(input).await
    }

    pub fn has_tool(&self, tool_name: &str) -> bool {
        self.tools.contains_key(tool_name)
    }

    pub fn input_schema(&self, tool_name: &str) -> Option<Value> {
        self.tools.get(tool_name).map(|tool| tool.input_schema())
    }

    pub fn output_schema(&self, tool_name: &str) -> Option<Value> {
        self.tools.get(tool_name).map(|tool| tool.output_schema())
    }
}
