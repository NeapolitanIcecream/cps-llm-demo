use anyhow::Result;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::effects::{
    Continuation, EffectFrame, EffectFrameEncoder, EncodedEffectFrame, ReturnSlot, RuntimeFrame,
};
use crate::store::state_dir::{StateDir, read_json, write_json_pretty};
use crate::store::value_store::FileValueStore;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ContinuationRef {
    pub continuation_id: String,
    pub uri: String,
}

#[derive(Debug, Clone)]
pub struct FileContinuationStore {
    state: StateDir,
    workflow_id: String,
}

impl FileContinuationStore {
    pub fn new(state: StateDir, workflow_id: impl Into<String>) -> Self {
        Self {
            state,
            workflow_id: workflow_id.into(),
        }
    }

    pub fn put(&self, continuation: &Continuation) -> Result<ContinuationRef> {
        self.state.ensure_workflow_layout(&self.workflow_id)?;
        let reference = ContinuationRef {
            continuation_id: continuation.continuation_id.clone(),
            uri: format!("continuationstore://{}", continuation.continuation_id),
        };
        write_json_pretty(
            &self
                .state
                .workflow_dir(&self.workflow_id)?
                .join("continuations")
                .join(format!("{}.json", continuation.continuation_id)),
            continuation,
        )?;
        Ok(reference)
    }

    pub fn get(&self, continuation_id: &str) -> Result<Continuation> {
        read_json(
            &self
                .state
                .workflow_dir(&self.workflow_id)?
                .join("continuations")
                .join(format!("{continuation_id}.json")),
        )
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct FrameEncodingConfig {
    pub max_inline_value_bytes: usize,
    pub max_model_visible_frame_bytes: usize,
}

impl Default for FrameEncodingConfig {
    fn default() -> Self {
        Self {
            max_inline_value_bytes: 2048,
            max_model_visible_frame_bytes: 16_384,
        }
    }
}

#[derive(Debug, Clone)]
pub struct FileEffectFrameEncoder {
    continuations: FileContinuationStore,
    values: FileValueStore,
    config: FrameEncodingConfig,
}

impl FileEffectFrameEncoder {
    pub fn new(
        state: StateDir,
        workflow_id: impl Into<String>,
        config: FrameEncodingConfig,
    ) -> Self {
        let workflow_id = workflow_id.into();
        Self {
            continuations: FileContinuationStore::new(state.clone(), workflow_id.clone()),
            values: FileValueStore::new(state, workflow_id),
            config,
        }
    }
}

impl EffectFrameEncoder for FileEffectFrameEncoder {
    fn encode(&self, frame: &EffectFrame) -> Result<EncodedEffectFrame> {
        let original_bytes = serde_json::to_vec(frame)?.len();
        let continuation_ref = self.continuations.put(&frame.continuation)?;
        let mut compact = frame.clone();

        for runtime_frame in &mut compact.continuation.stack {
            for value in runtime_frame.env.values_mut() {
                compact_value(&self.values, value, self.config.max_inline_value_bytes)?;
            }
            if let Some(ReturnSlot::MapElement { items, results, .. }) =
                &mut runtime_frame.return_to
            {
                compact_vec(&self.values, items, self.config.max_inline_value_bytes)?;
                compact_vec(&self.values, results, self.config.max_inline_value_bytes)?;
            }
        }
        for observation in &mut compact.observations {
            compact_value(
                &self.values,
                &mut observation.value,
                self.config.max_inline_value_bytes,
            )?;
        }

        let mut encoded_bytes = serde_json::to_vec(&compact)?.len();
        if encoded_bytes > self.config.max_model_visible_frame_bytes {
            for observation in &mut compact.observations {
                if let Some(text) = observation.value.as_str() {
                    observation.value = Value::String(truncate_string(text, 256));
                }
            }
            encoded_bytes = serde_json::to_vec(&compact)?.len();
        }

        if encoded_bytes > self.config.max_model_visible_frame_bytes {
            compact.observations.clear();
            encoded_bytes = serde_json::to_vec(&compact)?.len();
        }

        if encoded_bytes > self.config.max_model_visible_frame_bytes {
            elide_model_visible_continuation(&mut compact, &continuation_ref);
            encoded_bytes = serde_json::to_vec(&compact)?.len();
        }

        if encoded_bytes > self.config.max_model_visible_frame_bytes {
            anyhow::bail!(
                "encoded effect frame is {} bytes, over configured limit {}",
                encoded_bytes,
                self.config.max_model_visible_frame_bytes
            );
        }

        Ok(EncodedEffectFrame {
            model_visible_frame: compact,
            original_continuation_ref: Some(continuation_ref.uri),
            encoded_bytes,
            original_bytes,
        })
    }
}

fn compact_vec(store: &FileValueStore, values: &mut Vec<Value>, max_inline: usize) -> Result<()> {
    let value = Value::Array(values.clone());
    if serde_json::to_vec(&value)?.len() <= max_inline {
        return Ok(());
    }
    let reference = store.put(&value)?;
    *values = vec![FileValueStore::placeholder(&reference)];
    Ok(())
}

fn compact_value(store: &FileValueStore, value: &mut Value, max_inline: usize) -> Result<()> {
    if serde_json::to_vec(value)?.len() <= max_inline {
        return Ok(());
    }
    let reference = store.put(value)?;
    *value = FileValueStore::placeholder(&reference);
    Ok(())
}

fn elide_model_visible_continuation(frame: &mut EffectFrame, reference: &ContinuationRef) {
    let stack_depth = frame.continuation.stack.len();
    let (function, pc, return_to_elided) = frame
        .continuation
        .stack
        .last()
        .map(|runtime_frame| {
            (
                runtime_frame.function.clone(),
                runtime_frame.pc,
                runtime_frame.return_to.is_some(),
            )
        })
        .unwrap_or_else(|| ("<elided>".to_owned(), frame.continuation.resume_pc, false));

    let mut env = Map::new();
    env.insert(
        "$continuation_ref".to_owned(),
        Value::String(reference.uri.clone()),
    );
    env.insert(
        "$continuation_stack_depth".to_owned(),
        Value::from(stack_depth as u64),
    );
    env.insert("$continuation_env_elided".to_owned(), Value::Bool(true));
    if return_to_elided {
        env.insert(
            "$continuation_return_to_elided".to_owned(),
            Value::Bool(true),
        );
    }

    frame.continuation.stack = vec![RuntimeFrame {
        function,
        pc,
        env,
        return_to: None,
    }];
}

fn truncate_string(value: &str, max_chars: usize) -> String {
    if value.chars().count() <= max_chars {
        return value.to_owned();
    }
    let mut truncated = value.chars().take(max_chars).collect::<String>();
    truncated.push_str("...");
    truncated
}
