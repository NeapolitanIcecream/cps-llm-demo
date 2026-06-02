use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use anyhow::{Result, anyhow};
use async_trait::async_trait;
use cps_llm_demo::effects::{Continuation, EffectFrame, ThinkDecision};
use cps_llm_demo::models::{StrongModel, WeakModel, WeakTaskResult};
use cps_llm_demo::program::{Instr, JsonExpr, JsonObjectField, Program, WeakTaskSpec};
use cps_llm_demo::runtime::Runtime;
use cps_llm_demo::schema::{action_draft_schema, message_event_schema};
use cps_llm_demo::trace::TraceCollector;
use serde_json::{Map, Value, json};

#[derive(Clone)]
struct WeakCallRecord {
    task: WeakTaskSpec,
    input: Value,
    output_schema: Value,
}

#[derive(Clone)]
struct SequenceWeak {
    results: Arc<Mutex<VecDeque<std::result::Result<WeakTaskResult, String>>>>,
    calls: Arc<Mutex<Vec<WeakCallRecord>>>,
}

impl SequenceWeak {
    fn new(results: Vec<std::result::Result<WeakTaskResult, String>>) -> Self {
        Self {
            results: Arc::new(Mutex::new(results.into())),
            calls: Arc::new(Mutex::new(Vec::new())),
        }
    }

    fn empty() -> Self {
        Self::new(Vec::new())
    }

    fn calls(&self) -> Vec<WeakCallRecord> {
        self.calls.lock().unwrap().clone()
    }
}

#[async_trait]
impl WeakModel for SequenceWeak {
    async fn run_weak_task(
        &self,
        task: &WeakTaskSpec,
        input: &Value,
        output_schema: &Value,
    ) -> Result<WeakTaskResult> {
        self.calls.lock().unwrap().push(WeakCallRecord {
            task: task.clone(),
            input: input.clone(),
            output_schema: output_schema.clone(),
        });
        self.results
            .lock()
            .unwrap()
            .pop_front()
            .unwrap_or_else(|| Err("unexpected weak call".to_owned()))
            .map_err(|message| anyhow!(message))
    }
}

#[derive(Clone)]
struct SequenceStrong {
    compile_result: Option<Program>,
    decisions: Arc<Mutex<VecDeque<std::result::Result<ThinkDecision, String>>>>,
    calls: Arc<Mutex<Vec<EffectFrame>>>,
}

impl SequenceStrong {
    fn new(decisions: Vec<std::result::Result<ThinkDecision, String>>) -> Self {
        Self {
            compile_result: None,
            decisions: Arc::new(Mutex::new(decisions.into())),
            calls: Arc::new(Mutex::new(Vec::new())),
        }
    }

    fn empty() -> Self {
        Self::new(Vec::new())
    }

    fn calls(&self) -> Vec<EffectFrame> {
        self.calls.lock().unwrap().clone()
    }
}

#[async_trait]
impl StrongModel for SequenceStrong {
    async fn compile_program(
        &self,
        _task_spec: &str,
        _input_schema: &Value,
        _output_schema: &Value,
    ) -> Result<Program> {
        self.compile_result
            .clone()
            .ok_or_else(|| anyhow!("unexpected compile_program call"))
    }

    async fn think(&self, frame: &EffectFrame) -> Result<ThinkDecision> {
        self.calls.lock().unwrap().push(frame.clone());
        self.decisions
            .lock()
            .unwrap()
            .pop_front()
            .unwrap_or_else(|| Err("unexpected strong think call".to_owned()))
            .map_err(|message| anyhow!(message))
    }
}

#[test]
fn continuation_is_serializable_program_point_data() {
    let mut env = Map::new();
    env.insert(
        "$input".to_owned(),
        json!({ "event_id": "m1", "text": "hello" }),
    );

    let continuation = Continuation {
        program_id: "message_action_v1".to_owned(),
        pc: 2,
        resume_var: Some("draft".to_owned()),
        env,
        expected_schema: action_draft_schema(),
    };

    let value = serde_json::to_value(&continuation).unwrap();
    assert_eq!(value["program_id"], "message_action_v1");
    assert_eq!(value["pc"], 2);
    assert_eq!(value["resume_var"], "draft");

    let roundtrip: Continuation = serde_json::from_value(value).unwrap();
    assert_eq!(roundtrip, continuation);
}

#[tokio::test]
async fn program_interpreter_runs_multiple_instructions() {
    let weak = SequenceWeak::empty();
    let strong = SequenceStrong::empty();
    let runtime = Runtime::new(weak.clone(), strong.clone(), TraceCollector::default());
    let program = Program {
        program_id: "pure_projection".to_owned(),
        input_schema: json!({}),
        output_schema: json!({ "type": "string" }),
        instructions: vec![
            Instr::Set {
                var: "payload".to_owned(),
                value: json!({ "answer": "ok" }),
            },
            Instr::Project {
                out: "answer".to_owned(),
                from: JsonExpr::Var {
                    name: "payload".to_owned(),
                },
                path: vec!["answer".to_owned()],
            },
            Instr::Finish {
                value: JsonExpr::Var {
                    name: "answer".to_owned(),
                },
            },
        ],
    };

    let output = runtime.run_program(program, json!({})).await.unwrap();

    assert_eq!(output, json!("ok"));
    assert!(weak.calls().is_empty());
    assert!(strong.calls().is_empty());
}

#[tokio::test]
async fn weak_call_is_instruction_not_runtime_first_step() {
    let weak = SequenceWeak::empty();
    let strong = SequenceStrong::empty();
    let runtime = Runtime::new(weak.clone(), strong.clone(), TraceCollector::default());
    let program = Program {
        program_id: "no_weak_path".to_owned(),
        input_schema: message_event_schema(),
        output_schema: json!({ "type": "string" }),
        instructions: vec![
            Instr::Set {
                var: "result".to_owned(),
                value: json!("done_without_models"),
            },
            Instr::Finish {
                value: JsonExpr::Var {
                    name: "result".to_owned(),
                },
            },
        ],
    };

    let output = runtime
        .run_program(
            program,
            json!({ "event_id": "m1", "text": "semantic text" }),
        )
        .await
        .unwrap();

    assert_eq!(output, json!("done_without_models"));
    assert!(weak.calls().is_empty());
    assert!(strong.calls().is_empty());
}

#[tokio::test]
async fn program_with_two_weak_calls_can_capture_second_call() {
    let weak = SequenceWeak::new(vec![
        Ok(weak_result(json!({ "kind": "create_task" }), 0.91)),
        Ok(weak_result(action_value("weak_model"), 0.42)),
    ]);
    let strong = SequenceStrong::new(vec![Ok(ThinkDecision::ResumeWithValue {
        value: action_value("strong_think"),
        confidence: 0.89,
        rationale: "resolved stuck draft extraction".to_owned(),
    })]);
    let trace = TraceCollector::default();
    let runtime = Runtime::new(weak.clone(), strong.clone(), trace.clone());

    let output = runtime
        .run_program(two_stage_program(), message())
        .await
        .unwrap();

    assert_eq!(output["source"], "strong_think");
    let weak_calls = weak.calls();
    assert_eq!(weak_calls.len(), 2);
    assert_eq!(weak_calls[0].task.name, "classify_intent");
    assert_eq!(weak_calls[1].task.name, "extract_action_draft_from_intent");
    assert!(weak_calls[1].input.get("intent").is_some());
    assert!(weak_calls[1].output_schema.get("required").is_some());

    let strong_calls = strong.calls();
    assert_eq!(strong_calls.len(), 1);
    let frame = &strong_calls[0];
    assert_eq!(frame.continuation.pc, 2);
    assert_eq!(frame.continuation.resume_var.as_deref(), Some("draft"));
    assert!(matches!(
        frame.failed_instruction,
        Some(Instr::WeakCall { ref out, .. }) if out == "draft"
    ));

    assert!(trace.events().iter().any(|event| {
        event.event == "capture_continuation"
            && event.detail["pc"] == 2
            && event.detail["resume_var"] == "draft"
    }));
}

#[tokio::test]
async fn strong_request_weak_probe_then_resume() {
    let weak = SequenceWeak::new(vec![
        Ok(weak_result(action_value("weak_model"), 0.20)),
        Ok(weak_result(
            json!({ "datetime_candidates": ["tomorrow 10am"] }),
            0.92,
        )),
    ]);
    let strong = SequenceStrong::new(vec![
        Ok(ThinkDecision::RequestWeakProbe {
            out: "datetime_candidates".to_owned(),
            task: WeakTaskSpec {
                name: "extract_datetime_candidates".to_owned(),
                instructions: "Extract possible datetime hints.".to_owned(),
            },
            input: JsonExpr::Var {
                name: "$input".to_owned(),
            },
            output_schema: json!({
                "type": "object",
                "additionalProperties": false,
                "required": ["datetime_candidates"],
                "properties": {
                    "datetime_candidates": {
                        "type": "array",
                        "items": { "type": "string" }
                    }
                }
            }),
            min_confidence: 0.50,
            rationale: "need a local datetime probe".to_owned(),
        }),
        Ok(ThinkDecision::ResumeWithValue {
            value: action_value("strong_think"),
            confidence: 0.86,
            rationale: "used probe observation to fill draft".to_owned(),
        }),
    ]);
    let trace = TraceCollector::default();
    let runtime = Runtime::new(weak.clone(), strong.clone(), trace.clone());

    let output = runtime
        .run_program(single_weak_program(), message())
        .await
        .unwrap();

    assert_eq!(output["source"], "strong_think");
    assert_eq!(weak.calls().len(), 2);

    let strong_calls = strong.calls();
    assert_eq!(strong_calls.len(), 2);
    assert_eq!(strong_calls[0].observations.len(), 1);
    assert_eq!(strong_calls[1].observations.len(), 2);

    let events = trace.events();
    assert!(events.iter().any(|event| {
        event.event == "strong_think" && event.detail["decision"] == "request_weak_probe"
    }));
    assert!(events.iter().any(|event| event.event == "weak_probe"));
    assert!(events.iter().any(|event| {
        event.event == "strong_think" && event.detail["decision"] == "resume_with_value"
    }));
    assert!(
        events
            .iter()
            .any(|event| event.event == "resume_continuation")
    );
}

#[tokio::test]
async fn guard_think_resumes_with_value_for_failed_schema_contract() {
    let weak = SequenceWeak::empty();
    let strong = SequenceStrong::new(vec![Ok(ThinkDecision::ResumeWithValue {
        value: action_value("strong_think"),
        confidence: 0.90,
        rationale: "repaired invalid draft".to_owned(),
    })]);
    let runtime = Runtime::new(weak.clone(), strong.clone(), TraceCollector::default());
    let program = Program {
        program_id: "guard_repair".to_owned(),
        input_schema: message_event_schema(),
        output_schema: action_draft_schema(),
        instructions: vec![
            Instr::Set {
                var: "draft".to_owned(),
                value: json!({ "event_id": "m1" }),
            },
            Instr::Guard {
                condition: cps_llm_demo::program::GuardExpr::JsonSchemaValid {
                    var: "draft".to_owned(),
                    schema: action_draft_schema(),
                },
                on_fail: cps_llm_demo::program::GuardFail::Think {
                    reason: "draft failed action schema".to_owned(),
                },
            },
            Instr::Finish {
                value: JsonExpr::Var {
                    name: "draft".to_owned(),
                },
            },
        ],
    };

    let output = runtime.run_program(program, message()).await.unwrap();

    assert_eq!(output["source"], "strong_think");
    assert!(weak.calls().is_empty());
    let strong_calls = strong.calls();
    assert_eq!(strong_calls.len(), 1);
    assert_eq!(strong_calls[0].continuation.pc, 2);
    assert_eq!(
        strong_calls[0].continuation.resume_var.as_deref(),
        Some("draft")
    );
    assert_eq!(
        strong_calls[0].continuation.expected_schema,
        action_draft_schema()
    );
}

#[tokio::test]
async fn strong_output_must_match_expected_schema() {
    let weak = SequenceWeak::new(vec![Ok(weak_result(action_value("weak_model"), 0.10))]);
    let strong = SequenceStrong::new(vec![Ok(ThinkDecision::ResumeWithValue {
        value: json!({ "event_id": "m1" }),
        confidence: 0.90,
        rationale: "invalid partial answer".to_owned(),
    })]);
    let runtime = Runtime::new(weak, strong, TraceCollector::default());

    let error = runtime
        .run_program(single_weak_program(), message())
        .await
        .unwrap_err();

    assert!(
        error
            .to_string()
            .contains("strong ResumeWithValue failed expected schema")
    );
}

fn weak_result(value: Value, confidence: f32) -> WeakTaskResult {
    WeakTaskResult {
        value,
        confidence,
        rationale: "fake weak result".to_owned(),
    }
}

fn message() -> Value {
    json!({
        "event_id": "m1",
        "text": "明天 10 点前把新版 proposal 发我一下"
    })
}

fn action_value(source: &str) -> Value {
    json!({
        "event_id": "m1",
        "kind": "create_task",
        "title": "发送新版 proposal",
        "datetime_hint": "明天 10 点前",
        "source": source
    })
}

fn single_weak_program() -> Program {
    Program {
        program_id: "message_action_v1".to_owned(),
        input_schema: message_event_schema(),
        output_schema: action_draft_schema(),
        instructions: vec![
            Instr::WeakCall {
                out: "draft".to_owned(),
                task: WeakTaskSpec {
                    name: "classify_and_extract_action_draft".to_owned(),
                    instructions: "Return a complete action draft.".to_owned(),
                },
                input: JsonExpr::Var {
                    name: "$input".to_owned(),
                },
                output_schema: action_draft_schema(),
                min_confidence: 0.80,
            },
            Instr::Finish {
                value: JsonExpr::Var {
                    name: "draft".to_owned(),
                },
            },
        ],
    }
}

fn two_stage_program() -> Program {
    let fields = vec![
        JsonObjectField {
            name: "message".to_owned(),
            value: JsonExpr::Var {
                name: "$input".to_owned(),
            },
        },
        JsonObjectField {
            name: "intent".to_owned(),
            value: JsonExpr::Var {
                name: "intent".to_owned(),
            },
        },
    ];

    Program {
        program_id: "message_action_two_stage_v1".to_owned(),
        input_schema: message_event_schema(),
        output_schema: action_draft_schema(),
        instructions: vec![
            Instr::WeakCall {
                out: "intent".to_owned(),
                task: WeakTaskSpec {
                    name: "classify_intent".to_owned(),
                    instructions: "Classify intent only.".to_owned(),
                },
                input: JsonExpr::Var {
                    name: "$input".to_owned(),
                },
                output_schema: json!({
                    "type": "object",
                    "additionalProperties": false,
                    "required": ["kind"],
                    "properties": {
                        "kind": {
                            "type": "string",
                            "enum": ["ignore", "create_task", "create_calendar_event", "draft_reply"]
                        }
                    }
                }),
                min_confidence: 0.75,
            },
            Instr::WeakCall {
                out: "draft".to_owned(),
                task: WeakTaskSpec {
                    name: "extract_action_draft_from_intent".to_owned(),
                    instructions: "Return a complete action draft.".to_owned(),
                },
                input: JsonExpr::Object { fields },
                output_schema: action_draft_schema(),
                min_confidence: 0.80,
            },
            Instr::Finish {
                value: JsonExpr::Var {
                    name: "draft".to_owned(),
                },
            },
        ],
    }
}
