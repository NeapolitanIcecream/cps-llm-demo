use std::collections::{BTreeMap, VecDeque};
use std::sync::{Arc, Mutex};

use anyhow::{Result, anyhow};
use async_trait::async_trait;
use cps_llm_demo::effects::{
    AllowedDecision, Continuation, EffectReturnMode, HandlerDecision, HandlerRequest, ReturnSlot,
    RuntimeFrame,
};
use cps_llm_demo::models::EffectHandler;
use cps_llm_demo::program::{
    AcceptancePolicy, EffectCall, EffectPermission, FailureHandler, FunctionDef, Instr, JsonExpr,
    ModelStrength, ModelTaskSpec, PatchOp, Program, ProgramFragment, ProgramPatch,
};
use cps_llm_demo::runtime::Runtime;
use cps_llm_demo::schema::{action_draft_schema, message_event_schema, program_schema};
use cps_llm_demo::trace::{TraceCollector, replay_trace_events};
use cps_llm_demo::validator::validate_program;
use serde_json::{Map, Value, json};

#[derive(Clone)]
struct SequenceHandler {
    decisions: Arc<Mutex<VecDeque<std::result::Result<HandlerDecision, String>>>>,
    calls: Arc<Mutex<Vec<HandlerRequest>>>,
}

impl SequenceHandler {
    fn new(decisions: Vec<std::result::Result<HandlerDecision, String>>) -> Self {
        Self {
            decisions: Arc::new(Mutex::new(decisions.into())),
            calls: Arc::new(Mutex::new(Vec::new())),
        }
    }

    fn empty() -> Self {
        Self::new(Vec::new())
    }

    fn calls(&self) -> Vec<HandlerRequest> {
        self.calls.lock().unwrap().clone()
    }
}

#[async_trait]
impl EffectHandler for SequenceHandler {
    async fn handle(&self, request: HandlerRequest) -> Result<HandlerDecision> {
        self.calls.lock().unwrap().push(request);
        self.decisions
            .lock()
            .unwrap()
            .pop_front()
            .unwrap_or_else(|| Err("unexpected handler call".to_owned()))
            .map_err(|message| anyhow!(message))
    }
}

#[test]
fn continuation_is_serializable_full_program_stack() {
    let mut env = Map::new();
    env.insert("message".to_owned(), message("m1", "send proposal"));
    let continuation = Continuation {
        continuation_id: "k1".to_owned(),
        boundary_id: "b1".to_owned(),
        program_id: "message_action_v2".to_owned(),
        stack: vec![RuntimeFrame {
            function: "process_message".to_owned(),
            pc: 2,
            env,
            return_to: Some(ReturnSlot::MapElement {
                caller_function: "main".to_owned(),
                caller_pc: 1,
                out: "drafts".to_owned(),
                map_index: 1,
                item_var: "message".to_owned(),
                function: "process_message".to_owned(),
                items: vec![message("m0", "ignore"), message("m1", "send proposal")],
                results: vec![action_value("m0", "weak_model")],
            }),
        }],
        resume_var: Some("draft".to_owned()),
        resume_pc: 2,
        expected_schema: action_draft_schema(),
        fuel_remaining: 999,
        effect_depth: 0,
    };

    let value = serde_json::to_value(&continuation).unwrap();
    assert_eq!(value["continuation_id"], "k1");
    assert_eq!(value["stack"][0]["function"], "process_message");
    assert_eq!(value["stack"][0]["return_to"]["map_index"], 1);

    let roundtrip: Continuation = serde_json::from_value(value).unwrap();
    assert_eq!(roundtrip, continuation);
}

#[tokio::test]
async fn program_without_perform_calls_no_model() {
    let weak = SequenceHandler::empty();
    let strong = SequenceHandler::empty();
    let runtime = Runtime::new(weak.clone(), strong.clone(), TraceCollector::default());

    let output = runtime
        .run_program(pure_program(), json!({ "answer": "ok" }))
        .await
        .unwrap();

    assert_eq!(output, json!("ok"));
    assert!(weak.calls().is_empty());
    assert!(strong.calls().is_empty());
}

#[test]
fn zero_param_entry_can_reference_runtime_input() {
    let mut functions = BTreeMap::new();
    functions.insert(
        "main".to_owned(),
        FunctionDef {
            params: Vec::new(),
            output_schema: message_event_schema(),
            body: vec![Instr::Return {
                value: JsonExpr::Var {
                    name: "$input".to_owned(),
                },
            }],
        },
    );
    let program = Program {
        program_id: "zero_param_entry_input".to_owned(),
        version: "1.0.0".to_owned(),
        entry: "main".to_owned(),
        input_schema: message_event_schema(),
        output_schema: message_event_schema(),
        functions,
        allowed_effects: Vec::new(),
    };

    validate_program(&program).unwrap();
}

#[test]
fn multi_param_entry_is_rejected_during_validation() {
    let mut functions = BTreeMap::new();
    functions.insert(
        "main".to_owned(),
        FunctionDef {
            params: vec!["left".to_owned(), "right".to_owned()],
            output_schema: message_event_schema(),
            body: vec![Instr::Return {
                value: JsonExpr::Var {
                    name: "left".to_owned(),
                },
            }],
        },
    );
    let program = Program {
        program_id: "multi_param_entry".to_owned(),
        version: "1.0.0".to_owned(),
        entry: "main".to_owned(),
        input_schema: message_event_schema(),
        output_schema: message_event_schema(),
        functions,
        allowed_effects: Vec::new(),
    };

    let error = validate_program(&program).unwrap_err();

    assert!(
        error
            .to_string()
            .contains("entry function main expects 2 params")
    );
}

#[test]
fn zero_param_helper_cannot_reference_input() {
    let mut functions = BTreeMap::new();
    functions.insert(
        "main".to_owned(),
        FunctionDef {
            params: vec!["message".to_owned()],
            output_schema: message_event_schema(),
            body: vec![
                Instr::Call {
                    out: "helper_output".to_owned(),
                    function: "helper".to_owned(),
                    args: Vec::new(),
                },
                Instr::Return {
                    value: JsonExpr::Var {
                        name: "helper_output".to_owned(),
                    },
                },
            ],
        },
    );
    functions.insert(
        "helper".to_owned(),
        FunctionDef {
            params: Vec::new(),
            output_schema: message_event_schema(),
            body: vec![Instr::Return {
                value: JsonExpr::Var {
                    name: "$input".to_owned(),
                },
            }],
        },
    );
    let program = Program {
        program_id: "zero_param_helper_input".to_owned(),
        version: "1.0.0".to_owned(),
        entry: "main".to_owned(),
        input_schema: message_event_schema(),
        output_schema: message_event_schema(),
        functions,
        allowed_effects: Vec::new(),
    };

    let error = validate_program(&program).unwrap_err();

    assert!(
        error
            .to_string()
            .contains("variable $input is used before it is defined")
    );
}

#[test]
fn duplicate_function_params_are_rejected_during_validation() {
    let mut functions = BTreeMap::new();
    functions.insert(
        "main".to_owned(),
        FunctionDef {
            params: vec!["message".to_owned()],
            output_schema: message_event_schema(),
            body: vec![
                Instr::Call {
                    out: "helper_output".to_owned(),
                    function: "helper".to_owned(),
                    args: vec![
                        JsonExpr::Var {
                            name: "message".to_owned(),
                        },
                        JsonExpr::Var {
                            name: "message".to_owned(),
                        },
                    ],
                },
                Instr::Return {
                    value: JsonExpr::Var {
                        name: "helper_output".to_owned(),
                    },
                },
            ],
        },
    );
    functions.insert(
        "helper".to_owned(),
        FunctionDef {
            params: vec!["message".to_owned(), "message".to_owned()],
            output_schema: message_event_schema(),
            body: vec![Instr::Return {
                value: JsonExpr::Var {
                    name: "message".to_owned(),
                },
            }],
        },
    );
    let program = Program {
        program_id: "duplicate_function_params".to_owned(),
        version: "1.0.0".to_owned(),
        entry: "main".to_owned(),
        input_schema: message_event_schema(),
        output_schema: message_event_schema(),
        functions,
        allowed_effects: Vec::new(),
    };

    let error = validate_program(&program).unwrap_err();

    assert!(
        error
            .to_string()
            .contains("function helper has duplicate parameter message")
    );
}

#[test]
fn functions_must_end_with_return_during_validation() {
    let mut functions = BTreeMap::new();
    functions.insert(
        "main".to_owned(),
        FunctionDef {
            params: vec!["message".to_owned()],
            output_schema: message_event_schema(),
            body: vec![Instr::Let {
                var: "copy".to_owned(),
                expr: JsonExpr::Var {
                    name: "message".to_owned(),
                },
            }],
        },
    );
    functions.insert(
        "empty_helper".to_owned(),
        FunctionDef {
            params: Vec::new(),
            output_schema: json!({}),
            body: Vec::new(),
        },
    );
    let program = Program {
        program_id: "missing_terminal_return".to_owned(),
        version: "1.0.0".to_owned(),
        entry: "main".to_owned(),
        input_schema: message_event_schema(),
        output_schema: message_event_schema(),
        functions,
        allowed_effects: Vec::new(),
    };

    let error = validate_program(&program).unwrap_err();

    assert!(
        error
            .to_string()
            .contains("function empty_helper must end with return")
            || error
                .to_string()
                .contains("function main must end with return")
    );
}

#[test]
fn local_tool_perform_is_rejected_during_validation() {
    let mut functions = BTreeMap::new();
    functions.insert(
        "main".to_owned(),
        FunctionDef {
            params: vec!["request".to_owned()],
            output_schema: json!({ "type": "string" }),
            body: vec![
                Instr::Perform {
                    out: "tool_output".to_owned(),
                    effect: EffectCall::LocalTool {
                        tool_name: "calendar.create".to_owned(),
                        args_schema: json!({ "type": "object" }),
                    },
                    input: JsonExpr::Literal {
                        value: json!({ "title": "review" }),
                    },
                    expected_schema: json!({ "type": "string" }),
                    acceptance: accept(0.0),
                },
                Instr::Return {
                    value: JsonExpr::Var {
                        name: "tool_output".to_owned(),
                    },
                },
            ],
        },
    );
    let program = Program {
        program_id: "local_tool_unsupported".to_owned(),
        version: "1.0.0".to_owned(),
        entry: "main".to_owned(),
        input_schema: json!({ "type": "object" }),
        output_schema: json!({ "type": "string" }),
        functions,
        allowed_effects: vec![EffectPermission::LocalTool {
            tool_name: "calendar.create".to_owned(),
        }],
    };

    let error = validate_program(&program).unwrap_err();

    assert!(
        error
            .to_string()
            .contains("local_tool effects are not supported")
    );
}

#[test]
fn weak_compile_program_is_rejected_during_validation() {
    let mut functions = BTreeMap::new();
    functions.insert(
        "main".to_owned(),
        FunctionDef {
            params: vec!["request".to_owned()],
            output_schema: json!({}),
            body: vec![
                Instr::Perform {
                    out: "compiled".to_owned(),
                    effect: EffectCall::CompileProgram {
                        strength: ModelStrength::Weak,
                        task_spec: "compile a tiny demo program".to_owned(),
                        input_schema: json!({ "type": "object" }),
                        output_schema: json!({ "type": "object" }),
                    },
                    input: JsonExpr::Literal { value: json!({}) },
                    expected_schema: json!({}),
                    acceptance: accept(0.0),
                },
                Instr::Return {
                    value: JsonExpr::Var {
                        name: "compiled".to_owned(),
                    },
                },
            ],
        },
    );
    let program = Program {
        program_id: "weak_compile_unsupported".to_owned(),
        version: "1.0.0".to_owned(),
        entry: "main".to_owned(),
        input_schema: json!({ "type": "object" }),
        output_schema: json!({}),
        functions,
        allowed_effects: vec![EffectPermission::CompileProgram {
            strength: ModelStrength::Weak,
        }],
    };

    let error = validate_program(&program).unwrap_err();

    assert!(
        error
            .to_string()
            .contains("weak compile_program effects are not supported")
    );
}

#[test]
fn strong_compile_program_contract_schemas_are_validated() {
    let mut functions = BTreeMap::new();
    functions.insert(
        "main".to_owned(),
        FunctionDef {
            params: vec!["request".to_owned()],
            output_schema: json!({}),
            body: vec![
                Instr::Perform {
                    out: "compiled".to_owned(),
                    effect: EffectCall::CompileProgram {
                        strength: ModelStrength::Strong,
                        task_spec: "compile a tiny demo program".to_owned(),
                        input_schema: json!({ "type": "object" }),
                        output_schema: json!({ "type": "not_a_json_schema_type" }),
                    },
                    input: JsonExpr::Literal { value: json!({}) },
                    expected_schema: json!({}),
                    acceptance: accept(0.0),
                },
                Instr::Return {
                    value: JsonExpr::Var {
                        name: "compiled".to_owned(),
                    },
                },
            ],
        },
    );
    let program = Program {
        program_id: "strong_compile_invalid_contract".to_owned(),
        version: "1.0.0".to_owned(),
        entry: "main".to_owned(),
        input_schema: json!({ "type": "object" }),
        output_schema: json!({}),
        functions,
        allowed_effects: vec![EffectPermission::CompileProgram {
            strength: ModelStrength::Strong,
        }],
    };

    let error = validate_program(&program).unwrap_err();

    assert!(
        error
            .to_string()
            .contains("compile_program output_schema is invalid")
    );
}

#[tokio::test]
async fn weak_model_only_runs_when_program_performs_weak_effect() {
    let weak = SequenceHandler::new(vec![Ok(return_value(
        action_value("m1", "weak_model"),
        0.91,
    ))]);
    let strong = SequenceHandler::empty();
    let runtime = Runtime::new(weak.clone(), strong.clone(), TraceCollector::default());

    let output = runtime
        .run_program(single_weak_program(), message("m1", "send proposal"))
        .await
        .unwrap();

    assert_eq!(output["source"], "weak_model");
    assert_eq!(weak.calls().len(), 1);
    assert!(strong.calls().is_empty());
}

#[tokio::test]
async fn continuation_captures_second_effect_inside_map_and_resumes_ordered_output() {
    let weak = SequenceHandler::new(vec![
        Ok(return_value(json!({ "kind": "ignore" }), 0.91)),
        Ok(return_value(action_value("m0", "weak_model"), 0.91)),
        Ok(return_value(json!({ "kind": "create_task" }), 0.91)),
        Ok(return_value(action_value("m1", "weak_model"), 0.20)),
        Ok(return_value(
            json!({ "kind": "create_calendar_event" }),
            0.91,
        )),
        Ok(return_value(action_value("m2", "weak_model"), 0.91)),
    ]);
    let strong = SequenceHandler::new(vec![Ok(return_value(
        action_value("m1", "strong_think"),
        0.92,
    ))]);
    let trace = TraceCollector::default();
    let runtime = Runtime::new(weak.clone(), strong.clone(), trace.clone());

    let output = runtime
        .run_program(
            message_action_program(),
            json!([
                message("m0", "ignore this"),
                message("m1", "send proposal"),
                message("m2", "Friday 3pm review")
            ]),
        )
        .await
        .unwrap();

    assert_eq!(output.as_array().unwrap().len(), 3);
    assert_eq!(output[0]["event_id"], "m0");
    assert_eq!(output[1]["event_id"], "m1");
    assert_eq!(output[1]["source"], "strong_think");
    assert_eq!(output[2]["event_id"], "m2");

    let strong_calls = strong.calls();
    assert_eq!(strong_calls.len(), 1);
    let frame = strong_calls[0].effect_frame.as_ref().unwrap();
    assert_eq!(frame.continuation.resume_pc, 2);
    assert_eq!(frame.continuation.resume_var.as_deref(), Some("draft"));
    assert_eq!(
        frame.continuation.stack.last().unwrap().function,
        "process_message"
    );
    assert!(matches!(
        frame.continuation.stack.last().unwrap().return_to,
        Some(ReturnSlot::MapElement { map_index: 1, .. })
    ));

    let events = trace.events();
    assert!(
        events.iter().any(|event| {
            event.event == "capture_continuation" && event.detail["map_index"] == 1
        })
    );
    assert!(events.iter().any(|event| {
        event.event == "resume_continuation" && event.detail["resume_var"] == "draft"
    }));
    replay_trace_events(&events).unwrap();
}

#[tokio::test]
async fn weak_handler_can_request_strong_think_via_runtime() {
    let weak = SequenceHandler::new(vec![Ok(HandlerDecision::RequestEffect {
        effect: EffectCall::Think {
            reason: "weak handler cannot resolve ambiguous action".to_owned(),
        },
        input: message("m1", "send proposal"),
        expected_schema: action_draft_schema(),
        mode: EffectReturnMode::UseAsValue,
        rationale: "needs stronger reasoning".to_owned(),
    })]);
    let strong = SequenceHandler::new(vec![Ok(return_value(
        action_value("m1", "strong_think"),
        0.93,
    ))]);
    let trace = TraceCollector::default();
    let runtime = Runtime::new(weak.clone(), strong.clone(), trace.clone());

    let output = runtime
        .run_program(single_weak_program(), message("m1", "send proposal"))
        .await
        .unwrap();

    assert_eq!(output["source"], "strong_think");
    assert_eq!(weak.calls().len(), 1);
    assert_eq!(strong.calls().len(), 1);
    assert!(strong.calls()[0].effect_frame.is_none());
    assert!(trace.events().iter().any(|event| {
        event.event == "request_nested_effect"
            && event.detail["from_handler"] == "weak_model"
            && event.detail["to_effect"] == "think"
    }));
}

#[tokio::test]
async fn nested_effect_result_is_stamped_before_requested_schema_validation() {
    let weak = SequenceHandler::new(vec![Ok(HandlerDecision::RequestEffect {
        effect: EffectCall::Think {
            reason: "weak handler cannot resolve ambiguous action".to_owned(),
        },
        input: message("m1", "send proposal"),
        expected_schema: action_draft_schema(),
        mode: EffectReturnMode::UseAsValue,
        rationale: "needs stronger reasoning".to_owned(),
    })]);
    let strong = SequenceHandler::new(vec![Ok(return_value(action_without_source("m1"), 0.93))]);
    let trace = TraceCollector::default();
    let runtime = Runtime::new(weak.clone(), strong.clone(), trace.clone());

    let output = runtime
        .run_program(single_weak_program(), message("m1", "send proposal"))
        .await
        .unwrap();

    assert_eq!(output["source"], "strong_think");
    assert_eq!(weak.calls().len(), 1);
    assert_eq!(strong.calls().len(), 1);
    assert!(trace.events().iter().any(|event| {
        event.event == "nested_effect_result"
            && event.detail["schema_valid"] == true
            && event.detail["source"] == "strong_think"
    }));
    assert!(trace.events().iter().any(|event| {
        event.event == "handler_decision"
            && event.detail["handler"] == "strong_model"
            && event.detail["decision"] == "return_value"
            && event.detail["schema_valid"] == true
    }));
    replay_trace_events(&trace.events()).unwrap();
}

#[tokio::test]
async fn handler_requested_nested_effect_must_be_allowed_by_program_boundary() {
    let weak = SequenceHandler::new(vec![Ok(HandlerDecision::RequestEffect {
        effect: EffectCall::Think {
            reason: "weak handler cannot resolve ambiguous action".to_owned(),
        },
        input: message("m1", "send proposal"),
        expected_schema: action_draft_schema(),
        mode: EffectReturnMode::UseAsValue,
        rationale: "needs stronger reasoning".to_owned(),
    })]);
    let strong = SequenceHandler::new(vec![Ok(return_value(
        action_value("m1", "strong_think"),
        0.93,
    ))]);
    let trace = TraceCollector::default();
    let runtime = Runtime::new(weak.clone(), strong.clone(), trace.clone());
    let mut program = single_weak_program();
    program.allowed_effects = vec![EffectPermission::ModelTask {
        strength: ModelStrength::Weak,
    }];

    let error = runtime
        .run_program(program, message("m1", "send proposal"))
        .await
        .unwrap_err();

    assert!(
        error
            .to_string()
            .contains("effect think is not allowed by program single_weak")
    );
    assert_eq!(weak.calls().len(), 1);
    assert!(strong.calls().is_empty());
    assert!(
        !trace
            .events()
            .iter()
            .any(|event| event.event == "request_nested_effect")
    );
}

#[tokio::test]
async fn strong_handler_can_request_weak_probe_and_reenter() {
    let weak = SequenceHandler::new(vec![
        Ok(return_value(action_value("m1", "weak_model"), 0.20)),
        Ok(return_value(
            json!({ "candidates": ["tomorrow 10am"] }),
            0.91,
        )),
    ]);
    let strong = SequenceHandler::new(vec![
        Ok(HandlerDecision::RequestEffect {
            effect: EffectCall::ModelTask {
                strength: ModelStrength::Weak,
                task: ModelTaskSpec {
                    name: "extract_datetime_candidates".to_owned(),
                    instructions: "Extract datetime candidates only.".to_owned(),
                },
            },
            input: message("m1", "send proposal tomorrow 10am"),
            expected_schema: json!({
                "type": "object",
                "additionalProperties": false,
                "required": ["candidates"],
                "properties": {
                    "candidates": {
                        "type": "array",
                        "items": { "type": "string" }
                    }
                }
            }),
            mode: EffectReturnMode::ReenterHandler {
                observation_name: "datetime_candidates".to_owned(),
            },
            rationale: "need cheap local probe".to_owned(),
        }),
        Ok(return_value(action_value("m1", "strong_think"), 0.91)),
    ]);
    let trace = TraceCollector::default();
    let runtime = Runtime::new(weak.clone(), strong.clone(), trace.clone());

    let output = runtime
        .run_program(single_weak_program(), message("m1", "send proposal"))
        .await
        .unwrap();

    assert_eq!(output["source"], "strong_think");
    assert_eq!(weak.calls().len(), 2);
    let strong_calls = strong.calls();
    assert_eq!(strong_calls.len(), 2);
    assert!(strong_calls[0].observations.is_empty());
    assert_eq!(strong_calls[1].observations[0].name, "datetime_candidates");
    assert!(trace.events().iter().any(|event| {
        event.event == "reenter_handler"
            && event.detail["observation_name"] == "datetime_candidates"
    }));
}

#[tokio::test]
async fn weak_generated_program_fragment_can_call_strong_think() {
    let weak = SequenceHandler::new(vec![Ok(HandlerDecision::ReturnProgramFragment {
        fragment: generated_processor_fragment(),
        rationale: "generated processor".to_owned(),
    })]);
    let strong = SequenceHandler::new(vec![Ok(return_value(
        action_value("m1", "strong_think"),
        0.94,
    ))]);
    let trace = TraceCollector::default();
    let runtime = Runtime::new(weak.clone(), strong.clone(), trace.clone());

    let output = runtime
        .run_program(fractal_program(), message("m1", "send proposal"))
        .await
        .unwrap();

    assert_eq!(output["source"], "strong_think");
    assert_eq!(weak.calls().len(), 1);
    assert_eq!(strong.calls().len(), 1);
    assert!(
        trace
            .events()
            .iter()
            .any(|event| event.event == "program_fragment_validated")
    );
    assert!(
        trace
            .events()
            .iter()
            .any(|event| event.event == "program_fragment_installed")
    );
    assert!(matches!(strong.calls()[0].effect, EffectCall::Think { .. }));
    assert!(strong.calls()[0].effect_frame.is_none());
    assert!(trace.events().iter().any(|event| {
        event.event == "exec_instr"
            && event.detail["function"] == "__fragment_0__generated_processor"
            && event.detail["op"] == "perform"
    }));
}

#[tokio::test]
async fn generated_program_fragment_cannot_expand_effect_boundary() {
    let weak = SequenceHandler::new(vec![Ok(HandlerDecision::ReturnProgramFragment {
        fragment: generated_processor_fragment(),
        rationale: "generated processor".to_owned(),
    })]);
    let strong = SequenceHandler::new(vec![Ok(return_value(
        action_value("m1", "strong_think"),
        0.94,
    ))]);
    let trace = TraceCollector::default();
    let runtime = Runtime::new(weak.clone(), strong.clone(), trace.clone());
    let mut program = fractal_program();
    program.allowed_effects = vec![EffectPermission::ModelTask {
        strength: ModelStrength::Weak,
    }];

    let error = runtime
        .run_program(program, message("m1", "send proposal"))
        .await
        .unwrap_err();

    assert!(
        error
            .to_string()
            .contains("program fragment declares effect think not allowed by program fractal_test")
    );
    assert_eq!(weak.calls().len(), 1);
    assert!(strong.calls().is_empty());
    assert!(
        trace
            .events()
            .iter()
            .any(|event| event.event == "program_fragment_validated")
    );
    assert!(
        !trace
            .events()
            .iter()
            .any(|event| event.event == "program_fragment_installed")
    );
}

#[tokio::test]
async fn dynamic_fragment_return_must_match_fragment_output_schema() {
    let weak = SequenceHandler::empty();
    let strong = SequenceHandler::empty();
    let runtime = Runtime::new(weak, strong, TraceCollector::default());

    let error = runtime
        .run_program(
            dynamic_fragment_schema_program(),
            message("m1", "send proposal"),
        )
        .await
        .unwrap_err();

    assert!(
        error
            .to_string()
            .contains("dynamic call output failed fragment output_schema")
    );
}

#[tokio::test]
async fn dynamic_fragment_input_must_match_fragment_input_schema() {
    let weak = SequenceHandler::empty();
    let strong = SequenceHandler::empty();
    let trace = TraceCollector::default();
    let runtime = Runtime::new(weak, strong, trace.clone());

    let error = runtime
        .run_program(dynamic_fragment_input_schema_program(), json!({}))
        .await
        .unwrap_err();

    assert!(
        error
            .to_string()
            .contains("dynamic call input failed fragment input_schema")
    );
    assert!(
        !trace
            .events()
            .iter()
            .any(|event| event.event == "program_fragment_installed")
    );
}

#[tokio::test]
async fn dynamic_fragment_rejects_unenforceable_multi_param_input_contract() {
    let weak = SequenceHandler::empty();
    let strong = SequenceHandler::empty();
    let trace = TraceCollector::default();
    let runtime = Runtime::new(weak, strong, trace.clone());

    let error = runtime
        .run_program(dynamic_fragment_multi_param_schema_program(), json!({}))
        .await
        .unwrap_err();

    assert!(
        error
            .to_string()
            .contains("fragment input_schema can only be enforced for exactly one entry parameter")
    );
    assert!(
        !trace
            .events()
            .iter()
            .any(|event| event.event == "program_fragment_installed")
    );
}

#[tokio::test]
async fn dynamic_fragment_patch_must_validate_against_host_program() {
    let weak = SequenceHandler::empty();
    let strong = SequenceHandler::new(vec![Ok(HandlerDecision::ReturnProgramPatch {
        patch: ProgramPatch {
            target_program_id: "dynamic_fragment_patch".to_owned(),
            patch_id: "p1".to_owned(),
            operations: vec![PatchOp::UpdateAcceptancePolicy {
                function: "__fragment_0__generated_processor".to_owned(),
                pc: 0,
                acceptance: accept(0.1),
            }],
            rationale: "patch should target future host runs only".to_owned(),
        },
        rationale: "propose patch from generated code".to_owned(),
    })]);
    let trace = TraceCollector::default();
    let runtime = Runtime::new(weak, strong, trace.clone());

    let error = runtime
        .run_program(dynamic_fragment_patch_program(), json!({}))
        .await
        .unwrap_err();

    assert!(
        error
            .to_string()
            .contains("handler returned invalid ProgramPatch")
    );
    let patch_invalid = trace
        .events()
        .into_iter()
        .find(|event| event.event == "patch_invalid")
        .expect("patch should be rejected against the stable host program");
    assert!(
        patch_invalid.detail["reason"]
            .as_str()
            .unwrap()
            .contains("function __fragment_0__generated_processor does not exist")
    );
    let decision = trace
        .events()
        .into_iter()
        .find(|event| {
            event.event == "handler_decision"
                && event.detail["decision"] == json!("return_program_patch")
        })
        .expect("return_program_patch decision should be traced");
    assert_eq!(decision.detail["schema_valid"], json!(false));
    assert!(
        !trace
            .events()
            .iter()
            .any(|event| event.event == "patch_validated")
    );
}

#[tokio::test]
async fn program_patch_is_validated_and_recorded_without_mutating_active_stack() {
    let weak = SequenceHandler::empty();
    let strong = SequenceHandler::new(vec![Ok(HandlerDecision::ReturnProgramPatch {
        patch: ProgramPatch {
            target_program_id: "patch_demo".to_owned(),
            patch_id: "p1".to_owned(),
            operations: vec![PatchOp::UpdateAcceptancePolicy {
                function: "main".to_owned(),
                pc: 0,
                acceptance: accept(0.1),
            }],
            rationale: "future runs can lower confidence threshold".to_owned(),
        },
        rationale: "propose patch".to_owned(),
    })]);
    let trace = TraceCollector::default();
    let runtime = Runtime::new(weak.clone(), strong.clone(), trace.clone());

    let output = runtime
        .run_program(patch_program(), json!({}))
        .await
        .unwrap();

    assert_eq!(output, Value::Null);
    assert!(weak.calls().is_empty());
    assert_eq!(strong.calls().len(), 1);
    assert!(
        trace
            .events()
            .iter()
            .any(|event| event.event == "patch_proposed")
    );
    assert!(
        trace
            .events()
            .iter()
            .any(|event| event.event == "patch_validated")
    );
}

#[tokio::test]
async fn weak_model_patch_decision_is_rejected_for_direct_requests() {
    let weak = SequenceHandler::new(vec![Ok(HandlerDecision::ReturnProgramPatch {
        patch: ProgramPatch {
            target_program_id: "direct_weak_patch".to_owned(),
            patch_id: "p1".to_owned(),
            operations: vec![PatchOp::UpdateAcceptancePolicy {
                function: "main".to_owned(),
                pc: 0,
                acceptance: accept(0.1),
            }],
            rationale: "weak handler should not be able to propose host patches".to_owned(),
        },
        rationale: "try to propose patch from direct weak task".to_owned(),
    })]);
    let strong = SequenceHandler::empty();
    let trace = TraceCollector::default();
    let runtime = Runtime::new(weak.clone(), strong.clone(), trace.clone());

    let error = runtime
        .run_program(direct_weak_patch_program(), json!({}))
        .await
        .unwrap_err();

    assert!(
        error.to_string().contains(
            "handler decision return_program_patch is not allowed for weak_model request"
        )
    );
    assert_eq!(weak.calls().len(), 1);
    assert!(strong.calls().is_empty());
    assert!(
        trace
            .events()
            .iter()
            .any(|event| event.event == "handler_decision"
                && event.detail["decision"] == json!("return_program_patch"))
    );
    assert!(
        !trace
            .events()
            .iter()
            .any(|event| event.event == "patch_proposed" || event.event == "patch_validated")
    );
}

#[tokio::test]
async fn direct_non_compile_return_program_decision_is_rejected() {
    let weak = SequenceHandler::empty();
    let strong = SequenceHandler::new(vec![Ok(HandlerDecision::ReturnProgram {
        program: pure_program(),
        rationale: "try to return program IR from a Think request".to_owned(),
    })]);
    let trace = TraceCollector::default();
    let runtime = Runtime::new(weak.clone(), strong.clone(), trace.clone());

    let error = runtime
        .run_program(direct_think_return_program_program(), json!({}))
        .await
        .unwrap_err();

    assert!(
        error.to_string().contains(
            "handler decision return_program is only allowed for compile_program request"
        )
    );
    assert!(weak.calls().is_empty());
    assert_eq!(strong.calls().len(), 1);
    assert!(
        trace
            .events()
            .iter()
            .any(|event| event.event == "handler_decision"
                && event.detail["decision"] == json!("return_program"))
    );
}

#[tokio::test]
async fn nested_non_compile_return_program_decision_is_rejected() {
    let weak = SequenceHandler::empty();
    let strong = SequenceHandler::new(vec![
        Ok(HandlerDecision::RequestEffect {
            effect: strong_task("nested_non_compile_return_program"),
            input: json!({}),
            expected_schema: json!({}),
            mode: EffectReturnMode::UseAsValue,
            rationale: "ask a nested non-compile effect".to_owned(),
        }),
        Ok(HandlerDecision::ReturnProgram {
            program: pure_program(),
            rationale: "try to return program IR from nested model task".to_owned(),
        }),
    ]);
    let trace = TraceCollector::default();
    let runtime = Runtime::new(weak.clone(), strong.clone(), trace.clone());

    let error = runtime
        .run_program(nested_non_compile_return_program_program(), json!({}))
        .await
        .unwrap_err();

    assert!(
        error.to_string().contains(
            "handler decision return_program is only allowed for compile_program request"
        )
    );
    assert!(weak.calls().is_empty());
    assert_eq!(strong.calls().len(), 2);
    assert!(
        trace
            .events()
            .iter()
            .any(|event| event.event == "request_nested_effect"
                && event.detail["to_effect"] == json!("model_task"))
    );
}

#[tokio::test]
async fn compile_program_return_program_decision_uses_requested_contract() {
    let weak = SequenceHandler::empty();
    let strong = SequenceHandler::new(vec![Ok(HandlerDecision::ReturnProgram {
        program: pure_program(),
        rationale: "compiled Program IR with stale schemas".to_owned(),
    })]);
    let runtime = Runtime::new(weak.clone(), strong.clone(), TraceCollector::default());

    let output = runtime
        .run_program(compile_program_return_program_program(), json!({}))
        .await
        .unwrap();

    assert_eq!(output["program_id"], json!("pure_projection"));
    assert_eq!(output["input_schema"], message_event_schema());
    assert_eq!(output["output_schema"], action_draft_schema());
    assert!(weak.calls().is_empty());
    assert_eq!(strong.calls().len(), 1);
    assert!(matches!(
        strong.calls()[0].effect,
        EffectCall::CompileProgram {
            strength: ModelStrength::Strong,
            ..
        }
    ));
}

#[tokio::test]
async fn nested_compile_program_return_program_decision_uses_requested_contract() {
    let weak = SequenceHandler::empty();
    let strong = SequenceHandler::new(vec![
        Ok(HandlerDecision::RequestEffect {
            effect: EffectCall::CompileProgram {
                strength: ModelStrength::Strong,
                task_spec: "compile a nested processor".to_owned(),
                input_schema: message_event_schema(),
                output_schema: action_draft_schema(),
            },
            input: json!({}),
            expected_schema: program_schema(),
            mode: EffectReturnMode::UseAsValue,
            rationale: "ask a nested compiler".to_owned(),
        }),
        Ok(HandlerDecision::ReturnProgram {
            program: pure_program(),
            rationale: "nested compiled Program IR with stale schemas".to_owned(),
        }),
    ]);
    let trace = TraceCollector::default();
    let runtime = Runtime::new(weak.clone(), strong.clone(), trace.clone());

    let output = runtime
        .run_program(nested_compile_program_return_program_program(), json!({}))
        .await
        .unwrap();

    assert_eq!(output["program_id"], json!("pure_projection"));
    assert_eq!(output["input_schema"], message_event_schema());
    assert_eq!(output["output_schema"], action_draft_schema());
    assert!(weak.calls().is_empty());
    assert_eq!(strong.calls().len(), 2);
    assert!(
        trace
            .events()
            .iter()
            .any(|event| event.event == "request_nested_effect"
                && event.detail["to_effect"] == json!("compile_program"))
    );
}

#[tokio::test]
async fn program_patch_cannot_resume_non_null_captured_continuation() {
    let weak = SequenceHandler::new(vec![Ok(return_value(
        action_value("m1", "weak_model"),
        0.20,
    ))]);
    let strong = SequenceHandler::new(vec![Ok(HandlerDecision::ReturnProgramPatch {
        patch: ProgramPatch {
            target_program_id: "single_weak".to_owned(),
            patch_id: "p1".to_owned(),
            operations: vec![PatchOp::UpdateAcceptancePolicy {
                function: "main".to_owned(),
                pc: 0,
                acceptance: accept(0.1),
            }],
            rationale: "future runs can lower confidence threshold".to_owned(),
        },
        rationale: "propose patch".to_owned(),
    })]);
    let trace = TraceCollector::default();
    let runtime = Runtime::new(weak.clone(), strong.clone(), trace.clone());

    let error = runtime
        .run_program(single_weak_program(), message("m1", "send proposal"))
        .await
        .unwrap_err();

    assert!(error.to_string().contains(
        "handler decision return_program_patch is not allowed for captured effect frame"
    ));
    let frame = strong.calls()[0].effect_frame.as_ref().unwrap().clone();
    assert!(
        !frame
            .allowed_decisions
            .contains(&AllowedDecision::ReturnProgramPatch)
    );
    assert!(
        !trace
            .events()
            .iter()
            .any(|event| event.event == "patch_validated")
    );
}

#[tokio::test]
async fn captured_frame_rejects_return_program_decision() {
    let weak = SequenceHandler::new(vec![Ok(return_value(json!({ "anything": "broad" }), 0.20))]);
    let strong = SequenceHandler::new(vec![Ok(HandlerDecision::ReturnProgram {
        program: pure_program(),
        rationale: "try to resume with a full program".to_owned(),
    })]);
    let trace = TraceCollector::default();
    let runtime = Runtime::new(weak.clone(), strong.clone(), trace.clone());

    let error = runtime
        .run_program(captured_broad_schema_program(), json!({}))
        .await
        .unwrap_err();

    assert!(
        error
            .to_string()
            .contains("handler decision return_program is not allowed for captured effect frame")
    );
    let frame = strong.calls()[0].effect_frame.as_ref().unwrap().clone();
    assert!(
        frame
            .allowed_decisions
            .contains(&AllowedDecision::ReturnValue)
    );
    assert!(
        frame
            .allowed_decisions
            .contains(&AllowedDecision::RequestEffect)
    );
    assert!(
        frame
            .allowed_decisions
            .contains(&AllowedDecision::ReturnProgramPatch)
    );
    assert!(frame.allowed_decisions.contains(&AllowedDecision::Abort));
    assert!(
        !trace
            .events()
            .iter()
            .any(|event| event.event == "resume_continuation")
    );
}

#[tokio::test]
async fn strong_return_value_must_match_expected_schema_before_resume() {
    let weak = SequenceHandler::new(vec![Ok(return_value(
        action_value("m1", "weak_model"),
        0.20,
    ))]);
    let strong = SequenceHandler::new(vec![Ok(return_value(json!({ "event_id": "m1" }), 0.91))]);
    let runtime = Runtime::new(weak, strong, TraceCollector::default());

    let error = runtime
        .run_program(single_weak_program(), message("m1", "send proposal"))
        .await
        .unwrap_err();

    assert!(
        error
            .to_string()
            .contains("strong Think return_value failed expected schema")
    );
}

fn return_value(value: Value, confidence: f32) -> HandlerDecision {
    HandlerDecision::ReturnValue {
        value,
        confidence,
        rationale: "test fixture".to_owned(),
    }
}

fn message(event_id: &str, text: &str) -> Value {
    json!({ "event_id": event_id, "text": text })
}

fn action_value(event_id: &str, source: &str) -> Value {
    json!({
        "event_id": event_id,
        "kind": "create_task",
        "title": "Send proposal",
        "datetime_hint": null,
        "source": source,
    })
}

fn action_without_source(event_id: &str) -> Value {
    json!({
        "event_id": event_id,
        "kind": "create_task",
        "title": "Send proposal",
        "datetime_hint": null,
    })
}

fn pure_program() -> Program {
    let mut functions = BTreeMap::new();
    functions.insert(
        "main".to_owned(),
        FunctionDef {
            params: vec!["input".to_owned()],
            output_schema: json!({ "type": "string" }),
            body: vec![
                Instr::Project {
                    out: "answer".to_owned(),
                    from: JsonExpr::Var {
                        name: "input".to_owned(),
                    },
                    path: vec!["answer".to_owned()],
                },
                Instr::Return {
                    value: JsonExpr::Var {
                        name: "answer".to_owned(),
                    },
                },
            ],
        },
    );
    Program {
        program_id: "pure_projection".to_owned(),
        version: "1.0.0".to_owned(),
        entry: "main".to_owned(),
        input_schema: json!({
            "type": "object",
            "additionalProperties": false,
            "required": ["answer"],
            "properties": { "answer": { "type": "string" } }
        }),
        output_schema: json!({ "type": "string" }),
        functions,
        allowed_effects: Vec::new(),
    }
}

fn single_weak_program() -> Program {
    let mut functions = BTreeMap::new();
    functions.insert(
        "main".to_owned(),
        FunctionDef {
            params: vec!["message".to_owned()],
            output_schema: action_draft_schema(),
            body: vec![
                Instr::Perform {
                    out: "draft".to_owned(),
                    effect: weak_task("classify_and_extract_action_draft"),
                    input: JsonExpr::Var {
                        name: "message".to_owned(),
                    },
                    expected_schema: action_draft_schema(),
                    acceptance: accept(0.8),
                },
                Instr::Return {
                    value: JsonExpr::Var {
                        name: "draft".to_owned(),
                    },
                },
            ],
        },
    );
    Program {
        program_id: "single_weak".to_owned(),
        version: "1.0.0".to_owned(),
        entry: "main".to_owned(),
        input_schema: message_event_schema(),
        output_schema: action_draft_schema(),
        functions,
        allowed_effects: vec![
            EffectPermission::ModelTask {
                strength: ModelStrength::Weak,
            },
            EffectPermission::Think,
        ],
    }
}

fn message_action_program() -> Program {
    serde_json::from_str(include_str!("../examples/message_action.v2.program.json")).unwrap()
}

fn fractal_program() -> Program {
    let mut functions = BTreeMap::new();
    functions.insert(
        "main".to_owned(),
        FunctionDef {
            params: vec!["message".to_owned()],
            output_schema: action_draft_schema(),
            body: vec![
                Instr::Perform {
                    out: "processor".to_owned(),
                    effect: weak_task("generate_processor_program"),
                    input: JsonExpr::Var {
                        name: "message".to_owned(),
                    },
                    expected_schema: json!({}),
                    acceptance: AcceptancePolicy {
                        min_confidence: Some(0.0),
                        require_schema_valid: false,
                        on_failure: FailureHandler::Abort {
                            reason: "fragment generation failed".to_owned(),
                        },
                    },
                },
                Instr::CallDynamic {
                    out: "draft".to_owned(),
                    fragment: JsonExpr::Var {
                        name: "processor".to_owned(),
                    },
                    args: vec![JsonExpr::Var {
                        name: "message".to_owned(),
                    }],
                },
                Instr::Return {
                    value: JsonExpr::Var {
                        name: "draft".to_owned(),
                    },
                },
            ],
        },
    );
    Program {
        program_id: "fractal_test".to_owned(),
        version: "1.0.0".to_owned(),
        entry: "main".to_owned(),
        input_schema: message_event_schema(),
        output_schema: action_draft_schema(),
        functions,
        allowed_effects: vec![
            EffectPermission::ModelTask {
                strength: ModelStrength::Weak,
            },
            EffectPermission::Think,
        ],
    }
}

fn generated_processor_fragment() -> ProgramFragment {
    let mut functions = BTreeMap::new();
    functions.insert(
        "generated_processor".to_owned(),
        FunctionDef {
            params: vec!["message".to_owned()],
            output_schema: action_draft_schema(),
            body: vec![
                Instr::Perform {
                    out: "draft".to_owned(),
                    effect: EffectCall::Think {
                        reason: "generated code needs strong reasoning for final draft".to_owned(),
                    },
                    input: JsonExpr::Var {
                        name: "message".to_owned(),
                    },
                    expected_schema: action_draft_schema(),
                    acceptance: accept(0.0),
                },
                Instr::Return {
                    value: JsonExpr::Var {
                        name: "draft".to_owned(),
                    },
                },
            ],
        },
    );
    ProgramFragment {
        functions,
        entry: "generated_processor".to_owned(),
        input_schema: message_event_schema(),
        output_schema: action_draft_schema(),
        allowed_effects: vec![EffectPermission::Think],
    }
}

fn dynamic_fragment_schema_program() -> Program {
    let mut fragment_functions = BTreeMap::new();
    fragment_functions.insert(
        "generated_processor".to_owned(),
        FunctionDef {
            params: vec!["message".to_owned()],
            output_schema: json!({}),
            body: vec![Instr::Return {
                value: JsonExpr::Literal {
                    value: json!({ "not": "a string" }),
                },
            }],
        },
    );
    let fragment = ProgramFragment {
        functions: fragment_functions,
        entry: "generated_processor".to_owned(),
        input_schema: message_event_schema(),
        output_schema: json!({ "type": "string" }),
        allowed_effects: Vec::new(),
    };

    let mut functions = BTreeMap::new();
    functions.insert(
        "main".to_owned(),
        FunctionDef {
            params: vec!["message".to_owned()],
            output_schema: json!({}),
            body: vec![
                Instr::Let {
                    var: "processor".to_owned(),
                    expr: JsonExpr::Literal {
                        value: serde_json::to_value(fragment).unwrap(),
                    },
                },
                Instr::CallDynamic {
                    out: "result".to_owned(),
                    fragment: JsonExpr::Var {
                        name: "processor".to_owned(),
                    },
                    args: vec![JsonExpr::Var {
                        name: "message".to_owned(),
                    }],
                },
                Instr::Return {
                    value: JsonExpr::Var {
                        name: "result".to_owned(),
                    },
                },
            ],
        },
    );

    Program {
        program_id: "dynamic_fragment_schema".to_owned(),
        version: "1.0.0".to_owned(),
        entry: "main".to_owned(),
        input_schema: message_event_schema(),
        output_schema: json!({}),
        functions,
        allowed_effects: Vec::new(),
    }
}

fn dynamic_fragment_input_schema_program() -> Program {
    let mut fragment_functions = BTreeMap::new();
    fragment_functions.insert(
        "generated_processor".to_owned(),
        FunctionDef {
            params: vec!["message".to_owned()],
            output_schema: json!({}),
            body: vec![Instr::Return {
                value: JsonExpr::Var {
                    name: "message".to_owned(),
                },
            }],
        },
    );
    let fragment = ProgramFragment {
        functions: fragment_functions,
        entry: "generated_processor".to_owned(),
        input_schema: message_event_schema(),
        output_schema: json!({}),
        allowed_effects: Vec::new(),
    };

    let mut functions = BTreeMap::new();
    functions.insert(
        "main".to_owned(),
        FunctionDef {
            params: Vec::new(),
            output_schema: json!({}),
            body: vec![
                Instr::Let {
                    var: "processor".to_owned(),
                    expr: JsonExpr::Literal {
                        value: serde_json::to_value(fragment).unwrap(),
                    },
                },
                Instr::CallDynamic {
                    out: "result".to_owned(),
                    fragment: JsonExpr::Var {
                        name: "processor".to_owned(),
                    },
                    args: vec![JsonExpr::Literal {
                        value: json!({ "event_id": "m1" }),
                    }],
                },
                Instr::Return {
                    value: JsonExpr::Var {
                        name: "result".to_owned(),
                    },
                },
            ],
        },
    );

    Program {
        program_id: "dynamic_fragment_input_schema".to_owned(),
        version: "1.0.0".to_owned(),
        entry: "main".to_owned(),
        input_schema: json!({}),
        output_schema: json!({}),
        functions,
        allowed_effects: Vec::new(),
    }
}

fn dynamic_fragment_multi_param_schema_program() -> Program {
    let mut fragment_functions = BTreeMap::new();
    fragment_functions.insert(
        "generated_processor".to_owned(),
        FunctionDef {
            params: vec!["left".to_owned(), "right".to_owned()],
            output_schema: json!({}),
            body: vec![Instr::Return {
                value: JsonExpr::Var {
                    name: "left".to_owned(),
                },
            }],
        },
    );
    let fragment = ProgramFragment {
        functions: fragment_functions,
        entry: "generated_processor".to_owned(),
        input_schema: message_event_schema(),
        output_schema: json!({}),
        allowed_effects: Vec::new(),
    };

    let mut functions = BTreeMap::new();
    functions.insert(
        "main".to_owned(),
        FunctionDef {
            params: Vec::new(),
            output_schema: json!({}),
            body: vec![
                Instr::Let {
                    var: "processor".to_owned(),
                    expr: JsonExpr::Literal {
                        value: serde_json::to_value(fragment).unwrap(),
                    },
                },
                Instr::CallDynamic {
                    out: "result".to_owned(),
                    fragment: JsonExpr::Var {
                        name: "processor".to_owned(),
                    },
                    args: vec![
                        JsonExpr::Literal {
                            value: json!({ "event_id": "m1", "text": "left" }),
                        },
                        JsonExpr::Literal {
                            value: json!({ "event_id": "m2", "text": "right" }),
                        },
                    ],
                },
                Instr::Return {
                    value: JsonExpr::Var {
                        name: "result".to_owned(),
                    },
                },
            ],
        },
    );

    Program {
        program_id: "dynamic_fragment_multi_param_schema".to_owned(),
        version: "1.0.0".to_owned(),
        entry: "main".to_owned(),
        input_schema: json!({}),
        output_schema: json!({}),
        functions,
        allowed_effects: Vec::new(),
    }
}

fn dynamic_fragment_patch_program() -> Program {
    let mut fragment_functions = BTreeMap::new();
    fragment_functions.insert(
        "generated_processor".to_owned(),
        FunctionDef {
            params: vec!["input".to_owned()],
            output_schema: json!({ "type": "null" }),
            body: vec![
                Instr::Perform {
                    out: "patch_ack".to_owned(),
                    effect: EffectCall::Think {
                        reason: "generated code proposes a future patch".to_owned(),
                    },
                    input: JsonExpr::Var {
                        name: "input".to_owned(),
                    },
                    expected_schema: json!({ "type": "null" }),
                    acceptance: accept(0.0),
                },
                Instr::Return {
                    value: JsonExpr::Var {
                        name: "patch_ack".to_owned(),
                    },
                },
            ],
        },
    );
    let fragment = ProgramFragment {
        functions: fragment_functions,
        entry: "generated_processor".to_owned(),
        input_schema: json!({}),
        output_schema: json!({ "type": "null" }),
        allowed_effects: vec![EffectPermission::Think],
    };

    let mut functions = BTreeMap::new();
    functions.insert(
        "main".to_owned(),
        FunctionDef {
            params: Vec::new(),
            output_schema: json!({ "type": "null" }),
            body: vec![
                Instr::Let {
                    var: "processor".to_owned(),
                    expr: JsonExpr::Literal {
                        value: serde_json::to_value(fragment).unwrap(),
                    },
                },
                Instr::CallDynamic {
                    out: "patch_ack".to_owned(),
                    fragment: JsonExpr::Var {
                        name: "processor".to_owned(),
                    },
                    args: vec![JsonExpr::Literal { value: json!({}) }],
                },
                Instr::Return {
                    value: JsonExpr::Var {
                        name: "patch_ack".to_owned(),
                    },
                },
            ],
        },
    );

    Program {
        program_id: "dynamic_fragment_patch".to_owned(),
        version: "1.0.0".to_owned(),
        entry: "main".to_owned(),
        input_schema: json!({}),
        output_schema: json!({ "type": "null" }),
        functions,
        allowed_effects: vec![EffectPermission::Think],
    }
}

fn patch_program() -> Program {
    let mut functions = BTreeMap::new();
    functions.insert(
        "main".to_owned(),
        FunctionDef {
            params: Vec::new(),
            output_schema: json!({ "type": "null" }),
            body: vec![
                Instr::Perform {
                    out: "patch_ack".to_owned(),
                    effect: EffectCall::Think {
                        reason: "propose deterministic patch".to_owned(),
                    },
                    input: JsonExpr::Literal { value: json!({}) },
                    expected_schema: json!({ "type": "null" }),
                    acceptance: accept(0.0),
                },
                Instr::Return {
                    value: JsonExpr::Var {
                        name: "patch_ack".to_owned(),
                    },
                },
            ],
        },
    );
    Program {
        program_id: "patch_demo".to_owned(),
        version: "1.0.0".to_owned(),
        entry: "main".to_owned(),
        input_schema: json!({}),
        output_schema: json!({ "type": "null" }),
        functions,
        allowed_effects: vec![EffectPermission::Think],
    }
}

fn direct_weak_patch_program() -> Program {
    let mut functions = BTreeMap::new();
    functions.insert(
        "main".to_owned(),
        FunctionDef {
            params: Vec::new(),
            output_schema: json!({ "type": "null" }),
            body: vec![
                Instr::Perform {
                    out: "patch_ack".to_owned(),
                    effect: weak_task("direct_patch_attempt"),
                    input: JsonExpr::Literal { value: json!({}) },
                    expected_schema: json!({ "type": "null" }),
                    acceptance: accept(0.0),
                },
                Instr::Return {
                    value: JsonExpr::Var {
                        name: "patch_ack".to_owned(),
                    },
                },
            ],
        },
    );
    Program {
        program_id: "direct_weak_patch".to_owned(),
        version: "1.0.0".to_owned(),
        entry: "main".to_owned(),
        input_schema: json!({}),
        output_schema: json!({ "type": "null" }),
        functions,
        allowed_effects: vec![EffectPermission::ModelTask {
            strength: ModelStrength::Weak,
        }],
    }
}

fn direct_think_return_program_program() -> Program {
    let mut functions = BTreeMap::new();
    functions.insert(
        "main".to_owned(),
        FunctionDef {
            params: Vec::new(),
            output_schema: json!({}),
            body: vec![
                Instr::Perform {
                    out: "compiled".to_owned(),
                    effect: EffectCall::Think {
                        reason: "non-compile effect should not return Program IR".to_owned(),
                    },
                    input: JsonExpr::Literal { value: json!({}) },
                    expected_schema: json!({}),
                    acceptance: accept(0.0),
                },
                Instr::Return {
                    value: JsonExpr::Var {
                        name: "compiled".to_owned(),
                    },
                },
            ],
        },
    );
    Program {
        program_id: "direct_think_return_program".to_owned(),
        version: "1.0.0".to_owned(),
        entry: "main".to_owned(),
        input_schema: json!({}),
        output_schema: json!({}),
        functions,
        allowed_effects: vec![EffectPermission::Think],
    }
}

fn nested_non_compile_return_program_program() -> Program {
    let mut functions = BTreeMap::new();
    functions.insert(
        "main".to_owned(),
        FunctionDef {
            params: Vec::new(),
            output_schema: json!({}),
            body: vec![
                Instr::Perform {
                    out: "compiled".to_owned(),
                    effect: EffectCall::Think {
                        reason: "request a nested model task".to_owned(),
                    },
                    input: JsonExpr::Literal { value: json!({}) },
                    expected_schema: json!({}),
                    acceptance: accept(0.0),
                },
                Instr::Return {
                    value: JsonExpr::Var {
                        name: "compiled".to_owned(),
                    },
                },
            ],
        },
    );
    Program {
        program_id: "nested_non_compile_return_program".to_owned(),
        version: "1.0.0".to_owned(),
        entry: "main".to_owned(),
        input_schema: json!({}),
        output_schema: json!({}),
        functions,
        allowed_effects: vec![
            EffectPermission::Think,
            EffectPermission::ModelTask {
                strength: ModelStrength::Strong,
            },
        ],
    }
}

fn compile_program_return_program_program() -> Program {
    let mut functions = BTreeMap::new();
    functions.insert(
        "main".to_owned(),
        FunctionDef {
            params: Vec::new(),
            output_schema: program_schema(),
            body: vec![
                Instr::Perform {
                    out: "compiled".to_owned(),
                    effect: EffectCall::CompileProgram {
                        strength: ModelStrength::Strong,
                        task_spec: "compile a pure projection program".to_owned(),
                        input_schema: message_event_schema(),
                        output_schema: action_draft_schema(),
                    },
                    input: JsonExpr::Literal { value: json!({}) },
                    expected_schema: program_schema(),
                    acceptance: accept(0.0),
                },
                Instr::Return {
                    value: JsonExpr::Var {
                        name: "compiled".to_owned(),
                    },
                },
            ],
        },
    );
    Program {
        program_id: "compile_program_return_program".to_owned(),
        version: "1.0.0".to_owned(),
        entry: "main".to_owned(),
        input_schema: json!({}),
        output_schema: program_schema(),
        functions,
        allowed_effects: vec![EffectPermission::CompileProgram {
            strength: ModelStrength::Strong,
        }],
    }
}

fn nested_compile_program_return_program_program() -> Program {
    let mut functions = BTreeMap::new();
    functions.insert(
        "main".to_owned(),
        FunctionDef {
            params: Vec::new(),
            output_schema: program_schema(),
            body: vec![
                Instr::Perform {
                    out: "compiled".to_owned(),
                    effect: strong_task("nested_compile_program"),
                    input: JsonExpr::Literal { value: json!({}) },
                    expected_schema: program_schema(),
                    acceptance: accept(0.0),
                },
                Instr::Return {
                    value: JsonExpr::Var {
                        name: "compiled".to_owned(),
                    },
                },
            ],
        },
    );
    Program {
        program_id: "nested_compile_program_return_program".to_owned(),
        version: "1.0.0".to_owned(),
        entry: "main".to_owned(),
        input_schema: json!({}),
        output_schema: program_schema(),
        functions,
        allowed_effects: vec![
            EffectPermission::ModelTask {
                strength: ModelStrength::Strong,
            },
            EffectPermission::CompileProgram {
                strength: ModelStrength::Strong,
            },
        ],
    }
}

fn captured_broad_schema_program() -> Program {
    let mut functions = BTreeMap::new();
    functions.insert(
        "main".to_owned(),
        FunctionDef {
            params: Vec::new(),
            output_schema: json!({}),
            body: vec![
                Instr::Perform {
                    out: "repair".to_owned(),
                    effect: weak_task("broad_repair"),
                    input: JsonExpr::Literal { value: json!({}) },
                    expected_schema: json!({}),
                    acceptance: accept(0.8),
                },
                Instr::Return {
                    value: JsonExpr::Var {
                        name: "repair".to_owned(),
                    },
                },
            ],
        },
    );
    Program {
        program_id: "captured_broad_schema".to_owned(),
        version: "1.0.0".to_owned(),
        entry: "main".to_owned(),
        input_schema: json!({}),
        output_schema: json!({}),
        functions,
        allowed_effects: vec![
            EffectPermission::ModelTask {
                strength: ModelStrength::Weak,
            },
            EffectPermission::Think,
        ],
    }
}

fn weak_task(name: &str) -> EffectCall {
    EffectCall::ModelTask {
        strength: ModelStrength::Weak,
        task: ModelTaskSpec {
            name: name.to_owned(),
            instructions: name.to_owned(),
        },
    }
}

fn strong_task(name: &str) -> EffectCall {
    EffectCall::ModelTask {
        strength: ModelStrength::Strong,
        task: ModelTaskSpec {
            name: name.to_owned(),
            instructions: name.to_owned(),
        },
    }
}

fn accept(min_confidence: f32) -> AcceptancePolicy {
    AcceptancePolicy {
        min_confidence: Some(min_confidence),
        require_schema_valid: true,
        on_failure: FailureHandler::CaptureToThink {
            reason: "effect did not satisfy acceptance".to_owned(),
        },
    }
}
