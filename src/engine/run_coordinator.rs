use std::sync::Arc;

use anyhow::Result;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::effects::RuntimeBudget;
use crate::engine::event_source::EventSource;
use crate::experiment::prediction::{
    PredictionRecord, PredictionStore, prediction_metadata_from_trace,
};
use crate::local_tools::LocalToolRegistry;
use crate::models::EffectHandler;
use crate::observability::metrics::MetricsAccumulator;
use crate::runtime::Runtime;
use crate::schema::validate_value;
use crate::store::continuation_store::{FileEffectFrameEncoder, FrameEncodingConfig};
use crate::store::metrics_store::{FileMetricsStore, RunMetrics};
use crate::store::patch_registry::{FilePatchRegistry, PatchMetadata, PatchSource, PatchStatus};
use crate::store::profile_store::FileProfileStore;
use crate::store::program_registry::FileProgramRegistry;
use crate::store::state_dir::{StateDir, now_string};
use crate::store::trace_store::FileTraceStore;
use crate::trace::TraceCollector;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RunSummary {
    pub run_id: String,
    pub workflow_id: String,
    pub program_version: String,
    pub events_total: u64,
    pub events_succeeded: u64,
    pub events_failed: u64,
}

#[derive(Debug, Clone)]
pub struct RunStreamOptions {
    pub predictions: Option<PredictionWriteOptions>,
}

#[derive(Debug, Clone)]
pub struct PredictionWriteOptions {
    pub store: PredictionStore,
    pub file_name: String,
    pub variant: String,
}

pub async fn run_stream<S>(
    state: StateDir,
    workflow_id: &str,
    event_source: S,
    weak: Arc<dyn EffectHandler>,
    strong: Arc<dyn EffectHandler>,
    trace_json: bool,
) -> Result<RunSummary>
where
    S: EventSource,
{
    run_stream_with_options(
        state,
        workflow_id,
        event_source,
        weak,
        strong,
        trace_json,
        RunStreamOptions { predictions: None },
    )
    .await
}

pub async fn run_stream_with_options<S>(
    state: StateDir,
    workflow_id: &str,
    mut event_source: S,
    weak: Arc<dyn EffectHandler>,
    strong: Arc<dyn EffectHandler>,
    trace_json: bool,
    options: RunStreamOptions,
) -> Result<RunSummary>
where
    S: EventSource,
{
    let programs = FileProgramRegistry::new(state.clone());
    let traces = FileTraceStore::new(state.clone());
    let metrics_store = FileMetricsStore::new(state.clone());
    let profiles = FileProfileStore::new(state.clone());
    let patch_registry = FilePatchRegistry::new(state.clone());
    let program = programs.load_latest(workflow_id)?;
    let run_id = uuid::Uuid::new_v4().to_string();
    let mut metrics = RunMetrics::new(
        run_id.clone(),
        workflow_id.to_owned(),
        "stream".to_owned(),
        program.program_id.clone(),
        program.version.clone(),
        now_string(),
    );
    let mut accumulator = MetricsAccumulator::default();
    let frame_encoder = Arc::new(FileEffectFrameEncoder::new(
        state,
        workflow_id.to_owned(),
        FrameEncodingConfig::default(),
    ));

    while let Some(event) = event_source.next_event()? {
        metrics.events_total += 1;
        let trace = TraceCollector::default();
        let event_id = event_id_or_generate(&event);
        let event = event_with_schema_permitted_id(event, &program.input_schema, &event_id);
        trace.emit(
            "stream_event",
            event_id.clone(),
            serde_json::json!({
                "event": event.clone(),
            }),
        );
        let runtime = Runtime::with_frame_encoder(
            Arc::clone(&weak),
            Arc::clone(&strong),
            LocalToolRegistry::default(),
            frame_encoder.clone(),
            trace.clone(),
            RuntimeBudget::default(),
        );
        let result = runtime
            .run_program_with_result_and_trace_id(program.clone(), event, event_id.clone())
            .await;
        if let Ok(result) = &result {
            metrics.events_succeeded += 1;
            for patch in &result.pending_patches {
                let metadata = PatchMetadata {
                    patch_id: patch.patch_id.clone(),
                    workflow_id: workflow_id.to_owned(),
                    target_program_version: program.version.clone(),
                    status: PatchStatus::Proposed,
                    source: PatchSource::RuntimeStrongThink,
                    created_at: now_string(),
                    evaluated_at: None,
                    installed_at: None,
                    rationale: patch.rationale.clone(),
                    metrics_delta: None,
                };
                patch_registry.record_proposed(workflow_id, patch.clone(), metadata.clone())?;
                patch_registry.mark_validated(workflow_id, patch.clone(), metadata)?;
            }
        } else {
            metrics.events_failed += 1;
        }
        let events = trace.events();
        if let (Ok(result), Some(predictions)) = (&result, options.predictions.as_ref()) {
            let schema_valid = validate_value(&program.output_schema, &result.output).is_ok();
            let (fast_path, model_calls) = prediction_metadata_from_trace(&events);
            predictions.store.append(
                &predictions.file_name,
                &PredictionRecord {
                    event_id: event_id.clone(),
                    variant: predictions.variant.clone(),
                    program_version: program.version.clone(),
                    output: result.output.clone(),
                    schema_valid,
                    trace_run_id: Some(run_id.clone()),
                    fast_path,
                    model_calls,
                },
            )?;
        }
        accumulator.update_from_trace(&mut metrics, &events);
        traces.append_events(workflow_id, &run_id, &events)?;
        profiles.update_from_trace(workflow_id, &events)?;
        if trace_json {
            for event in events {
                eprintln!("{}", serde_json::to_string(&event)?);
            }
        }
    }

    accumulator.finalize(&mut metrics);
    metrics.finished_at = now_string();
    metrics_store.write(&metrics)?;

    Ok(RunSummary {
        run_id,
        workflow_id: workflow_id.to_owned(),
        program_version: program.version,
        events_total: metrics.events_total,
        events_succeeded: metrics.events_succeeded,
        events_failed: metrics.events_failed,
    })
}

pub fn event_id_or_generate(event: &Value) -> String {
    event
        .get("event_id")
        .and_then(Value::as_str)
        .map(ToOwned::to_owned)
        .unwrap_or_else(|| uuid::Uuid::new_v4().to_string())
}

fn event_with_schema_permitted_id(event: Value, input_schema: &Value, event_id: &str) -> Value {
    let Value::Object(mut object) = event.clone() else {
        return event;
    };
    object.insert("event_id".to_owned(), Value::String(event_id.to_owned()));
    let candidate = Value::Object(object);
    if validate_value(input_schema, &candidate).is_ok() {
        candidate
    } else {
        event
    }
}
