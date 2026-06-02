use anyhow::{Context, Result, anyhow};
use serde_json::{Map, Value, json};

use crate::effects::{Continuation, EffectFrame, ThinkDecision};
use crate::models::{StrongModel, WeakModel, WeakTaskResult};
use crate::program::{GuardExpr, GuardFail, Instr, JsonExpr, Program};
use crate::schema::validate_value;
use crate::trace::TraceCollector;

const INPUT_VAR: &str = "$input";
const MAX_THINK_TURNS_PER_FRAME: usize = 16;

#[derive(Debug, Clone, PartialEq)]
pub enum StepOutcome {
    Continue,
    Finished(Value),
    Effect(Box<EffectFrame>),
}

#[derive(Debug, Clone)]
pub struct RuntimeState {
    program: Program,
    pc: usize,
    env: Map<String, Value>,
    trace_id: String,
}

impl RuntimeState {
    pub fn new(program: Program, input: Value) -> Result<Self> {
        validate_value(&program.input_schema, &input).context("program input failed schema")?;

        let trace_id = input
            .get("event_id")
            .and_then(Value::as_str)
            .unwrap_or(&program.program_id)
            .to_owned();

        let mut env = Map::new();
        env.insert(INPUT_VAR.to_owned(), input);

        Ok(Self {
            program,
            pc: 0,
            env,
            trace_id,
        })
    }

    fn resume_with(&mut self, continuation: Continuation, value: Value) -> Result<()> {
        if continuation.program_id != self.program.program_id {
            return Err(anyhow!(
                "continuation program_id {} does not match running program {}",
                continuation.program_id,
                self.program.program_id
            ));
        }

        self.pc = continuation.pc;
        self.env = continuation.env;
        if let Some(var) = continuation.resume_var {
            self.env.insert(var, value);
        }
        Ok(())
    }
}

struct EffectCapture {
    resume_pc: usize,
    resume_var: Option<String>,
    expected_schema: Value,
    reason: String,
    failed_instruction: Option<Instr>,
    observations: Vec<Value>,
}

pub struct Runtime<W, S> {
    pub weak: W,
    pub strong: S,
    pub trace: TraceCollector,
}

impl<W, S> Runtime<W, S>
where
    W: WeakModel,
    S: StrongModel,
{
    pub fn new(weak: W, strong: S, trace: TraceCollector) -> Self {
        Self {
            weak,
            strong,
            trace,
        }
    }

    pub async fn run_program(&self, program: Program, input: Value) -> Result<Value> {
        let mut state = RuntimeState::new(program, input)?;
        self.trace.emit(
            "program_start",
            &state.trace_id,
            json!({ "program_id": &state.program.program_id }),
        );

        loop {
            match self.step(&mut state).await? {
                StepOutcome::Continue => continue,
                StepOutcome::Finished(value) => return Ok(value),
                StepOutcome::Effect(frame) => {
                    let frame = *frame;
                    let continuation = frame.continuation.clone();
                    let value = self.handle_think(frame, &state.trace_id).await?;
                    self.trace.emit(
                        "resume_continuation",
                        &state.trace_id,
                        json!({
                            "pc": continuation.pc,
                            "resume_var": continuation.resume_var.as_ref(),
                        }),
                    );
                    state.resume_with(continuation, value)?;
                }
            }
        }
    }

    async fn step(&self, state: &mut RuntimeState) -> Result<StepOutcome> {
        if state.pc >= state.program.instructions.len() {
            return Err(anyhow!(
                "program counter {} is outside instruction range",
                state.pc
            ));
        }

        let pc = state.pc;
        let instr = state.program.instructions[pc].clone();
        self.trace.emit(
            "exec_instr",
            &state.trace_id,
            json!({
                "pc": pc,
                "op": instr.op_name(),
                "out": instr.output_var(),
            }),
        );

        match instr {
            Instr::Set { var, value } => {
                state.env.insert(var, value);
                state.pc += 1;
                Ok(StepOutcome::Continue)
            }
            Instr::Project { out, from, path } => {
                let projected = project_path(&eval_expr(&state.env, &from)?, &path)?;
                state.env.insert(out, projected);
                state.pc += 1;
                Ok(StepOutcome::Continue)
            }
            Instr::Guard { condition, on_fail } => {
                if guard_passes(&state.env, &condition)? {
                    state.pc += 1;
                    return Ok(StepOutcome::Continue);
                }

                let failed_instruction = Instr::Guard {
                    condition: condition.clone(),
                    on_fail: on_fail.clone(),
                };
                let (resume_var, expected_schema) = guard_resume_contract(&condition);
                match on_fail {
                    GuardFail::Abort { reason } => Err(anyhow!("guard failed: {reason}")),
                    GuardFail::Think { reason } => {
                        Ok(StepOutcome::Effect(Box::new(self.make_effect_frame(
                            state,
                            EffectCapture {
                                resume_pc: pc + 1,
                                resume_var,
                                expected_schema,
                                reason,
                                failed_instruction: Some(failed_instruction),
                                observations: Vec::new(),
                            },
                        ))))
                    }
                }
            }
            Instr::WeakCall {
                out,
                task,
                input,
                output_schema,
                min_confidence,
            } => {
                ensure_probability(min_confidence, "weak_call min_confidence")?;

                let input_value = eval_expr(&state.env, &input)?;
                self.trace.emit(
                    "weak_call",
                    &state.trace_id,
                    json!({ "pc": pc, "task": &task.name, "out": &out }),
                );

                match self
                    .weak
                    .run_weak_task(&task, &input_value, &output_schema)
                    .await
                {
                    Ok(result) => {
                        let schema_valid = validate_value(&output_schema, &result.value).is_ok();
                        let confidence_valid = is_probability(result.confidence);
                        self.trace.emit(
                            "weak_result",
                            &state.trace_id,
                            json!({
                                "pc": pc,
                                "confidence": result.confidence,
                                "schema_valid": schema_valid,
                            }),
                        );

                        if confidence_valid && result.confidence >= min_confidence && schema_valid {
                            state.env.insert(out, result.value);
                            state.pc += 1;
                            Ok(StepOutcome::Continue)
                        } else {
                            let observation = weak_observation(
                                "weak_result",
                                &task.name,
                                Some(result),
                                None,
                                schema_valid,
                            );
                            let failed_instruction = Instr::WeakCall {
                                out: out.clone(),
                                task,
                                input,
                                output_schema: output_schema.clone(),
                                min_confidence,
                            };
                            Ok(StepOutcome::Effect(Box::new(self.make_effect_frame(
                                state,
                                EffectCapture {
                                    resume_pc: pc + 1,
                                    resume_var: Some(out),
                                    expected_schema: output_schema,
                                    reason: "weak_effect_failed".to_owned(),
                                    failed_instruction: Some(failed_instruction),
                                    observations: vec![observation],
                                },
                            ))))
                        }
                    }
                    Err(err) => {
                        self.trace.emit(
                            "weak_error",
                            &state.trace_id,
                            json!({ "pc": pc, "error": err.to_string() }),
                        );
                        let observation = weak_observation(
                            "weak_error",
                            &task.name,
                            None,
                            Some(err.to_string()),
                            false,
                        );
                        Ok(StepOutcome::Effect(Box::new(self.make_effect_frame(
                            state,
                            EffectCapture {
                                resume_pc: pc + 1,
                                resume_var: Some(out.clone()),
                                expected_schema: output_schema.clone(),
                                reason: "weak_model_error".to_owned(),
                                failed_instruction: Some(Instr::WeakCall {
                                    out,
                                    task,
                                    input,
                                    output_schema,
                                    min_confidence,
                                }),
                                observations: vec![observation],
                            },
                        ))))
                    }
                }
            }
            Instr::Finish { value } => {
                let output = eval_expr(&state.env, &value)?;
                validate_value(&state.program.output_schema, &output)
                    .context("program output failed schema")?;
                self.trace.emit(
                    "program_finished",
                    &state.trace_id,
                    json!({ "program_id": &state.program.program_id }),
                );
                Ok(StepOutcome::Finished(output))
            }
        }
    }

    async fn handle_think(&self, mut frame: EffectFrame, trace_id: &str) -> Result<Value> {
        for _ in 0..MAX_THINK_TURNS_PER_FRAME {
            let decision = self.strong.think(&frame).await?;
            self.trace.emit(
                "strong_think",
                trace_id,
                json!({
                    "effect_id": frame.effect_id,
                    "decision": decision.decision_name(),
                }),
            );

            match decision {
                ThinkDecision::ResumeWithValue { value, .. } => {
                    validate_value(&frame.continuation.expected_schema, &value)
                        .context("strong ResumeWithValue failed expected schema")?;
                    return Ok(value);
                }
                ThinkDecision::RequestWeakProbe {
                    out,
                    task,
                    input,
                    output_schema,
                    min_confidence,
                    ..
                } => {
                    ensure_probability(min_confidence, "weak_probe min_confidence")?;
                    let input_value = eval_expr(&frame.continuation.env, &input)?;
                    let probe = self
                        .weak
                        .run_weak_task(&task, &input_value, &output_schema)
                        .await;

                    match probe {
                        Ok(result) => {
                            let schema_valid =
                                validate_value(&output_schema, &result.value).is_ok();
                            self.trace.emit(
                                "weak_probe",
                                trace_id,
                                json!({
                                    "task": &task.name,
                                    "out": &out,
                                    "confidence": result.confidence,
                                    "schema_valid": schema_valid,
                                }),
                            );
                            frame.observations.push(weak_observation(
                                "weak_probe",
                                &task.name,
                                Some(result),
                                None,
                                schema_valid,
                            ));
                        }
                        Err(err) => {
                            self.trace.emit(
                                "weak_probe",
                                trace_id,
                                json!({
                                    "task": &task.name,
                                    "out": &out,
                                    "error": err.to_string(),
                                }),
                            );
                            frame.observations.push(weak_observation(
                                "weak_probe_error",
                                &task.name,
                                None,
                                Some(err.to_string()),
                                false,
                            ));
                        }
                    }
                }
                ThinkDecision::Abort { reason } => {
                    return Err(anyhow!("strong think aborted: {reason}"));
                }
            }
        }

        Err(anyhow!(
            "strong think exceeded {MAX_THINK_TURNS_PER_FRAME} turns for one effect frame"
        ))
    }

    fn make_effect_frame(&self, state: &RuntimeState, capture: EffectCapture) -> EffectFrame {
        self.trace.emit(
            "capture_continuation",
            &state.trace_id,
            json!({
                "program_id": &state.program.program_id,
                "pc": capture.resume_pc,
                "resume_var": capture.resume_var.as_ref(),
                "reason": &capture.reason,
            }),
        );

        EffectFrame {
            effect_id: uuid::Uuid::new_v4().to_string(),
            reason: capture.reason,
            failed_instruction: capture.failed_instruction,
            continuation: Continuation {
                program_id: state.program.program_id.clone(),
                pc: capture.resume_pc,
                resume_var: capture.resume_var,
                env: state.env.clone(),
                expected_schema: capture.expected_schema,
            },
            observations: capture.observations,
        }
    }
}

fn eval_expr(env: &Map<String, Value>, expr: &JsonExpr) -> Result<Value> {
    match expr {
        JsonExpr::Literal { value } => Ok(value.clone()),
        JsonExpr::Var { name } => env
            .get(name)
            .cloned()
            .ok_or_else(|| anyhow!("variable {name} is not defined")),
        JsonExpr::Object { fields } => {
            let mut object = Map::new();
            for (key, value_expr) in fields {
                object.insert(key.clone(), eval_expr(env, value_expr)?);
            }
            Ok(Value::Object(object))
        }
    }
}

fn project_path(value: &Value, path: &[String]) -> Result<Value> {
    let mut cursor = value;
    for segment in path {
        cursor = cursor
            .as_object()
            .and_then(|object| object.get(segment))
            .ok_or_else(|| anyhow!("project path segment {segment} was not present"))?;
    }
    Ok(cursor.clone())
}

fn guard_passes(env: &Map<String, Value>, condition: &GuardExpr) -> Result<bool> {
    match condition {
        GuardExpr::VarExists { name } => Ok(env.contains_key(name)),
        GuardExpr::JsonSchemaValid { var, schema } => {
            let Some(value) = env.get(var) else {
                return Ok(false);
            };
            Ok(validate_value(schema, value).is_ok())
        }
    }
}

fn guard_resume_contract(condition: &GuardExpr) -> (Option<String>, Value) {
    match condition {
        GuardExpr::VarExists { name } => (Some(name.clone()), json!({})),
        GuardExpr::JsonSchemaValid { var, schema } => (Some(var.clone()), schema.clone()),
    }
}

fn weak_observation(
    kind: &str,
    task: &str,
    result: Option<WeakTaskResult>,
    error: Option<String>,
    schema_valid: bool,
) -> Value {
    json!({
        "kind": kind,
        "task": task,
        "result": result,
        "error": error,
        "schema_valid": schema_valid,
    })
}

fn is_probability(value: f32) -> bool {
    value.is_finite() && (0.0..=1.0).contains(&value)
}

fn ensure_probability(value: f32, name: &str) -> Result<()> {
    if is_probability(value) {
        Ok(())
    } else {
        Err(anyhow!("{name} must be a finite probability"))
    }
}
