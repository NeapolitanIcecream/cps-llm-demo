use std::collections::BTreeMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use anyhow::{Context, Result, anyhow};
use serde_json::{Map, Value, json};

use crate::effects::{
    AllowedDecision, Continuation, ContinuationSummary, EffectFrame, EffectFrameEncoder,
    EffectReturnMode, HandlerBudget, HandlerDecision, HandlerRequest, Observation,
    ObservationSource, ReturnSlot, RuntimeBudget, RuntimeFrame,
};
use crate::local_tools::{FAST_PATH_APPLY_TOOL_NAME, LocalToolRegistry};
use crate::models::EffectHandler;
use crate::program::{
    AcceptancePolicy, EffectCall, FailureHandler, FunctionDef, GuardExpr, GuardFail, Instr,
    JsonExpr, ModelStrength, Program, ProgramFragment, ProgramPatch,
};
use crate::schema::{program_schema, validate_value};
use crate::trace::TraceCollector;
use crate::validator::{effect_allowed, validate_fragment, validate_patch, validate_program};

const INPUT_VAR: &str = "$input";

#[derive(Debug, Clone, PartialEq)]
pub enum StepOutcome {
    Continue,
    Finished(Value),
}

#[derive(Debug, Clone, PartialEq)]
pub struct ProgramRunResult {
    pub output: Value,
    pub pending_patches: Vec<ProgramPatch>,
}

#[derive(Debug, Clone)]
struct ProgramState {
    boundary_id: String,
    host_program: Program,
    program: Program,
    stack: Vec<RuntimeFrame>,
    trace_id: String,
    fuel_remaining: u64,
    effects_remaining: u64,
    budget: RuntimeBudget,
    fragment_count: u32,
    patch_attempts: u32,
    pending_patches: Vec<ProgramPatch>,
}

impl ProgramState {
    fn new(program: Program, input: Value, budget: RuntimeBudget) -> Result<Self> {
        validate_program(&program).context("program validation failed")?;
        validate_value(&program.input_schema, &input).context("program input failed schema")?;

        let trace_id = input
            .get("event_id")
            .and_then(Value::as_str)
            .unwrap_or(&program.program_id)
            .to_owned();
        let entry = program
            .functions
            .get(&program.entry)
            .ok_or_else(|| anyhow!("entry function {} does not exist", program.entry))?;
        if entry.params.len() > 1 {
            return Err(anyhow!(
                "entry function {} expects {} params; CLI runtime supplies one input value",
                program.entry,
                entry.params.len()
            ));
        }

        let mut env = Map::new();
        env.insert(INPUT_VAR.to_owned(), input.clone());
        if let Some(param) = entry.params.first() {
            env.insert(param.clone(), input);
        }

        let boundary_id = uuid::Uuid::new_v4().to_string();
        let entry_frame = RuntimeFrame {
            function: program.entry.clone(),
            pc: 0,
            env,
            return_to: None,
        };
        let host_program = program.clone();

        Ok(Self {
            boundary_id,
            host_program,
            program,
            stack: vec![entry_frame],
            trace_id,
            fuel_remaining: budget.max_instructions,
            effects_remaining: budget.max_effects,
            budget,
            fragment_count: 0,
            patch_attempts: 0,
            pending_patches: Vec::new(),
        })
    }

    fn top_frame(&self) -> Result<&RuntimeFrame> {
        self.stack
            .last()
            .ok_or_else(|| anyhow!("runtime stack is empty"))
    }

    fn top_frame_mut(&mut self) -> Result<&mut RuntimeFrame> {
        self.stack
            .last_mut()
            .ok_or_else(|| anyhow!("runtime stack is empty"))
    }

    fn current_function(&self) -> Result<&FunctionDef> {
        let function_name = &self.top_frame()?.function;
        self.program
            .functions
            .get(function_name)
            .ok_or_else(|| anyhow!("function {function_name} does not exist"))
    }

    fn current_instr(&self) -> Result<Instr> {
        let frame = self.top_frame()?;
        let body = &self.current_function()?.body;
        body.get(frame.pc).cloned().ok_or_else(|| {
            anyhow!(
                "program counter {} is outside function {} instruction range",
                frame.pc,
                frame.function
            )
        })
    }

    fn consume_instruction_fuel(&mut self) -> Result<()> {
        if self.fuel_remaining == 0 {
            return Err(anyhow!("runtime instruction fuel exhausted"));
        }
        self.fuel_remaining -= 1;
        Ok(())
    }

    fn consume_effect_budget(&mut self) -> Result<()> {
        if self.effects_remaining == 0 {
            return Err(anyhow!("runtime effect budget exhausted"));
        }
        self.effects_remaining -= 1;
        Ok(())
    }

    fn resume_with(&mut self, continuation: Continuation, value: Value) -> Result<()> {
        if continuation.boundary_id != self.boundary_id {
            return Err(anyhow!(
                "continuation boundary_id {} does not match running boundary {}",
                continuation.boundary_id,
                self.boundary_id
            ));
        }
        if continuation.program_id != self.program.program_id {
            return Err(anyhow!(
                "continuation program_id {} does not match running program {}",
                continuation.program_id,
                self.program.program_id
            ));
        }

        self.stack = continuation.stack;
        self.fuel_remaining = continuation.fuel_remaining;
        let top = self.top_frame_mut()?;
        top.pc = continuation.resume_pc;
        if let Some(var) = continuation.resume_var {
            top.env.insert(var, value);
        }
        Ok(())
    }
}

struct EffectCapture {
    resume_pc: usize,
    resume_var: Option<String>,
    expected_schema: Value,
    reason: String,
    failed_effect: Option<EffectCall>,
    failed_instruction: Option<Instr>,
    observations: Vec<Observation>,
    effect_depth: u32,
}

#[derive(Debug, Clone)]
struct EffectResolution {
    value: Value,
    confidence: f32,
    observations: Vec<Observation>,
    source: ObservationSource,
}

struct EffectWork {
    effect: EffectCall,
    input: Value,
    expected_schema: Value,
    effect_frame: Option<EffectFrame>,
    observations: Vec<Observation>,
    depth: u32,
}

pub struct Runtime<W, S> {
    pub weak: W,
    pub strong: S,
    pub local_tools: LocalToolRegistry,
    pub frame_encoder: Option<Arc<dyn EffectFrameEncoder>>,
    pub trace: TraceCollector,
    pub budget: RuntimeBudget,
}

impl<W, S> Runtime<W, S>
where
    W: EffectHandler,
    S: EffectHandler,
{
    pub fn new(weak: W, strong: S, trace: TraceCollector) -> Self {
        Self {
            weak,
            strong,
            local_tools: LocalToolRegistry::default(),
            frame_encoder: None,
            trace,
            budget: RuntimeBudget::default(),
        }
    }

    pub fn with_budget(weak: W, strong: S, trace: TraceCollector, budget: RuntimeBudget) -> Self {
        Self {
            weak,
            strong,
            local_tools: LocalToolRegistry::default(),
            frame_encoder: None,
            trace,
            budget,
        }
    }

    pub fn with_local_tools(
        weak: W,
        strong: S,
        local_tools: LocalToolRegistry,
        trace: TraceCollector,
        budget: RuntimeBudget,
    ) -> Self {
        Self {
            weak,
            strong,
            local_tools,
            frame_encoder: None,
            trace,
            budget,
        }
    }

    pub fn with_frame_encoder(
        weak: W,
        strong: S,
        local_tools: LocalToolRegistry,
        frame_encoder: Arc<dyn EffectFrameEncoder>,
        trace: TraceCollector,
        budget: RuntimeBudget,
    ) -> Self {
        Self {
            weak,
            strong,
            local_tools,
            frame_encoder: Some(frame_encoder),
            trace,
            budget,
        }
    }

    pub async fn run_program(&self, program: Program, input: Value) -> Result<Value> {
        Ok(self.run_program_with_result(program, input).await?.output)
    }

    pub async fn run_program_with_result(
        &self,
        program: Program,
        input: Value,
    ) -> Result<ProgramRunResult> {
        let mut state = ProgramState::new(program, input, self.budget.clone())?;
        self.trace.emit(
            "program_validated",
            &state.trace_id,
            json!({
                "program_id": &state.program.program_id,
                "version": &state.program.version,
            }),
        );
        self.trace.emit(
            "program_start",
            &state.trace_id,
            json!({
                "program_id": &state.program.program_id,
                "version": &state.program.version,
                "boundary_id": &state.boundary_id,
            }),
        );
        self.trace.emit(
            "enter_function",
            &state.trace_id,
            json!({ "function": &state.program.entry, "stack_depth": 1 }),
        );

        loop {
            match self.step(&mut state).await {
                Ok(StepOutcome::Continue) => continue,
                Ok(StepOutcome::Finished(output)) => {
                    return Ok(ProgramRunResult {
                        output,
                        pending_patches: state.pending_patches.clone(),
                    });
                }
                Err(err) => {
                    self.trace.emit(
                        "program_aborted",
                        &state.trace_id,
                        json!({ "reason": err.to_string() }),
                    );
                    return Err(err);
                }
            }
        }
    }

    async fn step(&self, state: &mut ProgramState) -> Result<StepOutcome> {
        state.consume_instruction_fuel()?;

        let function = state.top_frame()?.function.clone();
        let pc = state.top_frame()?.pc;
        let instr = state.current_instr()?;
        self.trace.emit(
            "exec_instr",
            &state.trace_id,
            json!({
                "function": &function,
                "pc": pc,
                "op": instr.op_name(),
                "out": instr.output_var(),
                "fuel_remaining": state.fuel_remaining,
            }),
        );

        match instr {
            Instr::Let { var, expr } => {
                let value = eval_expr(&state.top_frame()?.env, &expr)?;
                let frame = state.top_frame_mut()?;
                frame.env.insert(var, value);
                frame.pc += 1;
                Ok(StepOutcome::Continue)
            }
            Instr::Project { out, from, path } => {
                let projected = project_path(&eval_expr(&state.top_frame()?.env, &from)?, &path)?;
                let frame = state.top_frame_mut()?;
                frame.env.insert(out, projected);
                frame.pc += 1;
                Ok(StepOutcome::Continue)
            }
            Instr::Guard { condition, on_fail } => {
                if guard_passes(&state.top_frame()?.env, &condition)? {
                    state.top_frame_mut()?.pc += 1;
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
                        let frame = self.make_effect_frame(
                            state,
                            EffectCapture {
                                resume_pc: pc + 1,
                                resume_var,
                                expected_schema,
                                reason: reason.clone(),
                                failed_effect: Some(EffectCall::Think { reason }),
                                failed_instruction: Some(failed_instruction),
                                observations: Vec::new(),
                                effect_depth: 0,
                            },
                        )?;
                        self.handle_captured_think(state, frame).await?;
                        Ok(StepOutcome::Continue)
                    }
                }
            }
            Instr::Branch {
                condition,
                then_pc,
                else_pc,
            } => {
                let target = if guard_passes(&state.top_frame()?.env, &condition)? {
                    then_pc
                } else {
                    else_pc
                };
                self.trace.emit(
                    "branch_decision",
                    &state.trace_id,
                    json!({
                        "function": &function,
                        "pc": pc,
                        "target_pc": target,
                    }),
                );
                state.top_frame_mut()?.pc = target;
                Ok(StepOutcome::Continue)
            }
            Instr::Jump { pc: target_pc } => {
                self.trace.emit(
                    "jump",
                    &state.trace_id,
                    json!({
                        "function": &function,
                        "pc": pc,
                        "target_pc": target_pc,
                    }),
                );
                state.top_frame_mut()?.pc = target_pc;
                Ok(StepOutcome::Continue)
            }
            Instr::Perform {
                out,
                effect,
                input,
                expected_schema,
                acceptance,
            } => {
                let failed_instruction = Instr::Perform {
                    out: out.clone(),
                    effect: effect.clone(),
                    input: input.clone(),
                    expected_schema: expected_schema.clone(),
                    acceptance: acceptance.clone(),
                };
                if !effect_allowed(&state.program.allowed_effects, &effect) {
                    return Err(anyhow!(
                        "effect {} is not allowed by program {}",
                        effect.kind_name(),
                        state.program.program_id
                    ));
                }

                let input_value = eval_expr(&state.top_frame()?.env, &input)?;
                self.trace.emit(
                    "perform_effect",
                    &state.trace_id,
                    json!({
                        "function": &function,
                        "pc": pc,
                        "out": &out,
                        "effect": effect.kind_name(),
                        "strength": effect.strength().map(|strength| match strength {
                            ModelStrength::Weak => "weak",
                            ModelStrength::Strong => "strong",
                        }),
                        "task": effect.model_task_name(),
                    }),
                );

                let resolution = match self
                    .resolve_effect(
                        state,
                        EffectWork {
                            effect: effect.clone(),
                            input: input_value,
                            expected_schema: expected_schema.clone(),
                            effect_frame: None,
                            observations: Vec::new(),
                            depth: 0,
                        },
                    )
                    .await
                {
                    Ok(resolution) => resolution,
                    Err(err) => match acceptance.on_failure.clone() {
                        _ if !is_handler_failure(&err) => {
                            return Err(err);
                        }
                        FailureHandler::Abort { reason } => {
                            return Err(err.context(format!("perform failed: {reason}")));
                        }
                        FailureHandler::CaptureToThink { reason } => {
                            let error = root_error_message(&err);
                            let observation = Observation {
                                name: "perform_error".to_owned(),
                                value: json!({
                                    "error": error,
                                    "source": source_for_effect(&effect).as_str(),
                                }),
                                source: ObservationSource::Runtime,
                            };
                            let frame = self.make_effect_frame(
                                state,
                                EffectCapture {
                                    resume_pc: pc + 1,
                                    resume_var: Some(out),
                                    expected_schema,
                                    reason,
                                    failed_effect: Some(effect),
                                    failed_instruction: Some(failed_instruction),
                                    observations: vec![observation],
                                    effect_depth: 0,
                                },
                            )?;
                            self.handle_captured_think(state, frame).await?;
                            return Ok(StepOutcome::Continue);
                        }
                    },
                };

                if accepted_by_policy(
                    &expected_schema,
                    &acceptance,
                    &resolution.value,
                    resolution.confidence,
                ) {
                    let frame = state.top_frame_mut()?;
                    frame.env.insert(out, resolution.value);
                    frame.pc += 1;
                    Ok(StepOutcome::Continue)
                } else {
                    let schema_valid = validate_value(&expected_schema, &resolution.value).is_ok();
                    let observation = Observation {
                        name: "perform_result".to_owned(),
                        value: json!({
                            "value": resolution.value,
                            "confidence": resolution.confidence,
                            "schema_valid": schema_valid,
                            "source": resolution.source.as_str(),
                        }),
                        source: resolution.source,
                    };
                    let mut observations = resolution.observations;
                    observations.push(observation);

                    match acceptance.on_failure.clone() {
                        FailureHandler::Abort { reason } => {
                            Err(anyhow!("perform failed: {reason}"))
                        }
                        FailureHandler::CaptureToThink { reason } => {
                            let frame = self.make_effect_frame(
                                state,
                                EffectCapture {
                                    resume_pc: pc + 1,
                                    resume_var: Some(out),
                                    expected_schema,
                                    reason,
                                    failed_effect: Some(effect),
                                    failed_instruction: Some(failed_instruction),
                                    observations,
                                    effect_depth: 0,
                                },
                            )?;
                            self.handle_captured_think(state, frame).await?;
                            Ok(StepOutcome::Continue)
                        }
                    }
                }
            }
            Instr::Call {
                out,
                function,
                args,
            } => {
                let arg_values = eval_exprs(&state.top_frame()?.env, &args)?;
                self.push_call_frame(
                    state,
                    function,
                    arg_values,
                    ReturnSlot::Call {
                        caller_function: state.top_frame()?.function.clone(),
                        caller_pc: pc + 1,
                        var: out,
                        expected_schema: None,
                    },
                )?;
                Ok(StepOutcome::Continue)
            }
            Instr::Map {
                out,
                items,
                item_var,
                function,
            } => {
                let items_value = eval_expr(&state.top_frame()?.env, &items)?;
                let items = items_value
                    .as_array()
                    .cloned()
                    .ok_or_else(|| anyhow!("map items expression did not evaluate to an array"))?;
                if items.is_empty() {
                    let frame = state.top_frame_mut()?;
                    frame.env.insert(out, Value::Array(Vec::new()));
                    frame.pc += 1;
                    return Ok(StepOutcome::Continue);
                }

                self.trace.emit(
                    "map_item_start",
                    &state.trace_id,
                    json!({
                        "function": &function,
                        "caller_function": &state.top_frame()?.function,
                        "index": 0,
                    }),
                );
                let return_to = ReturnSlot::MapElement {
                    caller_function: state.top_frame()?.function.clone(),
                    caller_pc: pc + 1,
                    out,
                    map_index: 0,
                    item_var,
                    function: function.clone(),
                    items: items.clone(),
                    results: Vec::new(),
                };
                self.push_map_item_frame(state, function, items[0].clone(), return_to)?;
                Ok(StepOutcome::Continue)
            }
            Instr::CallDynamic {
                out,
                fragment,
                args,
            } => {
                let fragment_value = eval_expr(&state.top_frame()?.env, &fragment)?;
                let fragment: ProgramFragment = serde_json::from_value(fragment_value)
                    .context("invalid ProgramFragment value")?;
                validate_fragment(&fragment).context("program fragment validation failed")?;
                self.trace.emit(
                    "program_fragment_validated",
                    &state.trace_id,
                    json!({
                        "entry": &fragment.entry,
                        "function_count": fragment.functions.len(),
                    }),
                );
                let fragment_output_schema = fragment.output_schema.clone();
                let arg_values = eval_exprs(&state.top_frame()?.env, &args)?;
                validate_dynamic_fragment_input_contract(&fragment, &arg_values)?;
                let entry = self.install_fragment(state, fragment)?;
                self.push_call_frame(
                    state,
                    entry,
                    arg_values,
                    ReturnSlot::Call {
                        caller_function: state.top_frame()?.function.clone(),
                        caller_pc: pc + 1,
                        var: out,
                        expected_schema: Some(fragment_output_schema),
                    },
                )?;
                Ok(StepOutcome::Continue)
            }
            Instr::Return { value } => {
                let output = eval_expr(&state.top_frame()?.env, &value)?;
                self.return_from_function(state, output)
            }
        }
    }

    fn resolve_effect<'a>(
        &'a self,
        state: &'a mut ProgramState,
        work: EffectWork,
    ) -> Pin<Box<dyn Future<Output = Result<EffectResolution>> + Send + 'a>> {
        Box::pin(async move {
            let EffectWork {
                effect,
                input,
                expected_schema,
                effect_frame,
                observations,
                depth,
            } = work;
            if depth > state.budget.max_effect_depth {
                return Err(anyhow!(
                    "effect depth {depth} exceeded max {}",
                    state.budget.max_effect_depth
                ));
            }
            state.consume_effect_budget()?;

            validate_effect_input_contract(&effect, &input)?;

            let handler_name = effect.handler_name();
            self.trace.emit(
                "handler_request",
                &state.trace_id,
                json!({
                    "handler": handler_name,
                    "effect": effect.kind_name(),
                    "strength": effect.strength().map(|strength| match strength {
                        ModelStrength::Weak => "weak",
                        ModelStrength::Strong => "strong",
                    }),
                    "task": effect.model_task_name(),
                    "source": source_for_effect(&effect).as_str(),
                    "depth": depth,
                }),
            );

            let mut request = HandlerRequest {
                effect: effect.clone(),
                input,
                expected_schema: expected_schema.clone(),
                continuation_summary: effect_frame
                    .as_ref()
                    .map(|frame| continuation_summary(&frame.continuation)),
                effect_frame,
                observations,
                budget: HandlerBudget {
                    effect_depth: depth,
                    effects_remaining: state.effects_remaining,
                    handler_reentries_remaining: state.budget.max_handler_reentries,
                },
            };

            let mut reentries = 0;
            loop {
                let decision = match handler_name {
                    "weak_model" => self
                        .weak
                        .handle(request.clone())
                        .await
                        .with_context(|| format!("{handler_name} handler failed"))?,
                    "strong_model" => self
                        .strong
                        .handle(request.clone())
                        .await
                        .with_context(|| format!("{handler_name} handler failed"))?,
                    "local_tool" => self
                        .local_tools
                        .handle(request.clone())
                        .await
                        .with_context(|| format!("{handler_name} handler failed"))?,
                    _ => return Err(anyhow!("unknown handler {handler_name}")),
                };

                let schema_valid = match &decision {
                    HandlerDecision::ReturnValue { value, .. } => {
                        Some(trace_return_value_schema_valid(&expected_schema, value))
                    }
                    HandlerDecision::ReturnProgram { program, .. } => Some(
                        validate_value(
                            &program_schema(),
                            &serde_json::to_value(program_with_compile_contract(
                                &effect,
                                program.clone(),
                            ))?,
                        )
                        .is_ok(),
                    ),
                    HandlerDecision::ReturnProgramFragment { fragment, .. } => {
                        Some(validate_fragment(fragment).is_ok())
                    }
                    HandlerDecision::ReturnProgramPatch { patch, .. } => {
                        Some(validate_patch(&state.host_program, patch).is_ok())
                    }
                    HandlerDecision::RequestEffect { .. } | HandlerDecision::Abort { .. } => None,
                };
                self.trace.emit(
                    "handler_decision",
                    &state.trace_id,
                    json!({
                        "handler": handler_name,
                        "effect": effect.kind_name(),
                        "decision": decision.decision_name(),
                        "confidence": decision.confidence(),
                        "schema_valid": schema_valid,
                        "source": source_for_effect(&effect).as_str(),
                        "depth": depth,
                    }),
                );
                ensure_handler_decision_allowed(
                    handler_name,
                    &request.effect,
                    request.effect_frame.as_ref(),
                    &decision,
                )?;

                match decision {
                    HandlerDecision::ReturnValue {
                        value, confidence, ..
                    } => {
                        ensure_probability(confidence, "handler return_value confidence")?;
                        self.emit_local_tool_result_trace(state, &effect, &value);
                        let source = source_for_effect(&effect);
                        return Ok(EffectResolution {
                            value,
                            confidence,
                            observations: request.observations,
                            source,
                        });
                    }
                    HandlerDecision::RequestEffect {
                        effect: requested_effect,
                        input,
                        expected_schema: requested_schema,
                        mode,
                        ..
                    } => {
                        if matches!(mode, EffectReturnMode::ReenterHandler { .. })
                            && reentries >= state.budget.max_handler_reentries
                        {
                            return Err(anyhow!(
                                "handler reentry limit {} exceeded",
                                state.budget.max_handler_reentries
                            ));
                        }
                        if !effect_allowed(&state.program.allowed_effects, &requested_effect) {
                            return Err(anyhow!(
                                "effect {} is not allowed by program {}",
                                requested_effect.kind_name(),
                                state.program.program_id
                            ));
                        }
                        self.trace.emit(
                            "request_nested_effect",
                            &state.trace_id,
                            json!({
                                "from_handler": handler_name,
                                "from_effect": effect.kind_name(),
                                "to_effect": requested_effect.kind_name(),
                                "to_handler": requested_effect.handler_name(),
                                "mode": mode.mode_name(),
                                "depth": depth + 1,
                            }),
                        );
                        let nested = self
                            .resolve_effect(
                                state,
                                EffectWork {
                                    effect: requested_effect,
                                    input,
                                    expected_schema: requested_schema.clone(),
                                    effect_frame: None,
                                    observations: Vec::new(),
                                    depth: depth + 1,
                                },
                            )
                            .await?;
                        let schema_valid = validate_value(&requested_schema, &nested.value).is_ok();
                        self.trace.emit(
                            "nested_effect_result",
                            &state.trace_id,
                            json!({
                                "mode": mode.mode_name(),
                                "schema_valid": schema_valid,
                                "source": nested.source.as_str(),
                                "depth": depth + 1,
                            }),
                        );

                        match mode {
                            EffectReturnMode::UseAsValue => {
                                if !schema_valid {
                                    return Err(anyhow!(
                                        "nested effect result failed requested expected_schema"
                                    ));
                                }
                                return Ok(nested);
                            }
                            EffectReturnMode::ReenterHandler { observation_name } => {
                                if !schema_valid {
                                    return Err(anyhow!(
                                        "nested effect result failed requested expected_schema"
                                    ));
                                }
                                reentries += 1;
                                request.observations.push(Observation {
                                    name: observation_name.clone(),
                                    value: nested.value,
                                    source: nested.source,
                                });
                                request.budget.effects_remaining = state.effects_remaining;
                                request.budget.handler_reentries_remaining =
                                    state.budget.max_handler_reentries - reentries;
                                self.trace.emit(
                                    "reenter_handler",
                                    &state.trace_id,
                                    json!({
                                        "handler": handler_name,
                                        "effect": effect.kind_name(),
                                        "observation_name": observation_name,
                                        "reentry": reentries,
                                    }),
                                );
                            }
                        }
                    }
                    HandlerDecision::ReturnProgram { program, .. } => {
                        let program = program_with_compile_contract(&effect, program);
                        validate_program(&program).context("handler returned invalid Program")?;
                        return Ok(EffectResolution {
                            value: serde_json::to_value(program)?,
                            confidence: 1.0,
                            observations: request.observations,
                            source: ObservationSource::StrongModel,
                        });
                    }
                    HandlerDecision::ReturnProgramFragment { fragment, .. } => {
                        validate_fragment(&fragment)
                            .context("handler returned invalid ProgramFragment")?;
                        self.trace.emit(
                            "program_fragment_validated",
                            &state.trace_id,
                            json!({
                                "entry": &fragment.entry,
                                "function_count": fragment.functions.len(),
                            }),
                        );
                        return Ok(EffectResolution {
                            value: serde_json::to_value(fragment)?,
                            confidence: 1.0,
                            observations: request.observations,
                            source: source_for_effect(&effect),
                        });
                    }
                    HandlerDecision::ReturnProgramPatch { patch, .. } => {
                        if !schema_accepts_null(&expected_schema) {
                            return Err(anyhow!(
                                "return_program_patch cannot satisfy expected_schema that rejects null"
                            ));
                        }
                        if state.patch_attempts >= state.budget.max_patch_attempts {
                            return Err(anyhow!(
                                "program patch attempt limit {} exceeded",
                                state.budget.max_patch_attempts
                            ));
                        }
                        state.patch_attempts += 1;
                        self.trace.emit(
                            "patch_proposed",
                            &state.trace_id,
                            json!({
                                "patch_id": &patch.patch_id,
                                "target_program_id": &patch.target_program_id,
                                "operation_count": patch.operations.len(),
                            }),
                        );
                        match validate_patch(&state.host_program, &patch) {
                            Ok(patched) => {
                                self.trace.emit(
                                    "patch_validated",
                                    &state.trace_id,
                                    json!({
                                        "patch_id": &patch.patch_id,
                                        "new_version": patched.version,
                                    }),
                                );
                                state.pending_patches.push(patch);
                                return Ok(EffectResolution {
                                    value: Value::Null,
                                    confidence: 1.0,
                                    observations: request.observations,
                                    source: ObservationSource::Runtime,
                                });
                            }
                            Err(err) => {
                                self.trace.emit(
                                    "patch_invalid",
                                    &state.trace_id,
                                    json!({
                                        "patch_id": &patch.patch_id,
                                        "reason": err.to_string(),
                                    }),
                                );
                                return Err(err).context("handler returned invalid ProgramPatch");
                            }
                        }
                    }
                    HandlerDecision::Abort { reason } => {
                        return Err(anyhow!("{handler_name} aborted: {reason}"));
                    }
                }
            }
        })
    }

    async fn handle_captured_think(
        &self,
        state: &mut ProgramState,
        frame: EffectFrame,
    ) -> Result<()> {
        let expected_schema = frame.continuation.expected_schema.clone();
        let continuation = frame.continuation.clone();
        let mut model_visible_frame = frame;
        if let Some(encoder) = &self.frame_encoder {
            let encoded = encoder.encode(&model_visible_frame)?;
            self.trace.emit(
                "continuation_frame_encoded",
                &state.trace_id,
                json!({
                    "continuation_id": &continuation.continuation_id,
                    "original_continuation_ref": encoded.original_continuation_ref,
                    "original_bytes": encoded.original_bytes,
                    "encoded_bytes": encoded.encoded_bytes,
                }),
            );
            model_visible_frame = encoded.model_visible_frame;
        }
        let reason = model_visible_frame.reason.clone();
        let effect = EffectCall::Think { reason };
        if !effect_allowed(&state.program.allowed_effects, &effect) {
            return Err(anyhow!(
                "effect {} is not allowed by program {}",
                effect.kind_name(),
                state.program.program_id
            ));
        }
        let resolution = self
            .resolve_effect(
                state,
                EffectWork {
                    effect,
                    input: serde_json::to_value(&model_visible_frame)?,
                    expected_schema: expected_schema.clone(),
                    effect_frame: Some(model_visible_frame),
                    observations: Vec::new(),
                    depth: continuation.effect_depth + 1,
                },
            )
            .await?;
        validate_value(&expected_schema, &resolution.value)
            .context("strong Think return_value failed expected schema")?;

        self.trace.emit(
            "resume_continuation",
            &state.trace_id,
            json!({
                "continuation_id": &continuation.continuation_id,
                "function": continuation.stack.last().map(|frame| frame.function.as_str()),
                "pc": continuation.resume_pc,
                "resume_var": continuation.resume_var.as_ref(),
            }),
        );
        state.resume_with(continuation, resolution.value)?;
        Ok(())
    }

    fn make_effect_frame(
        &self,
        state: &ProgramState,
        capture: EffectCapture,
    ) -> Result<EffectFrame> {
        let continuation_id = uuid::Uuid::new_v4().to_string();
        let effect_id = uuid::Uuid::new_v4().to_string();
        let top = state.top_frame()?;
        let map_index = state
            .stack
            .iter()
            .rev()
            .find_map(|frame| match &frame.return_to {
                Some(ReturnSlot::MapElement { map_index, .. }) => Some(*map_index),
                _ => None,
            });
        self.trace.emit(
            "capture_continuation",
            &state.trace_id,
            json!({
                "effect_id": &effect_id,
                "continuation_id": &continuation_id,
                "boundary_id": &state.boundary_id,
                "program_id": &state.program.program_id,
                "program_version": &state.program.version,
                "function": &top.function,
                "pc": top.pc,
                "resume_pc": capture.resume_pc,
                "resume_var": capture.resume_var.as_ref(),
                "reason": &capture.reason,
                "stack_depth": state.stack.len(),
                "map_index": map_index,
                "failed_effect_kind": capture.failed_effect.as_ref().map(EffectCall::kind_name),
                "failed_task_name": capture.failed_effect.as_ref().and_then(EffectCall::model_task_name),
                "expected_schema": &capture.expected_schema,
                "observations": &capture.observations,
            }),
        );

        let mut allowed_decisions =
            vec![AllowedDecision::ReturnValue, AllowedDecision::RequestEffect];
        if schema_accepts_null(&capture.expected_schema) {
            allowed_decisions.push(AllowedDecision::ReturnProgramPatch);
        }
        allowed_decisions.push(AllowedDecision::Abort);

        Ok(EffectFrame {
            effect_id,
            boundary_id: state.boundary_id.clone(),
            reason: capture.reason,
            failed_effect: capture.failed_effect,
            failed_instruction: capture.failed_instruction,
            continuation: Continuation {
                continuation_id,
                boundary_id: state.boundary_id.clone(),
                program_id: state.program.program_id.clone(),
                stack: state.stack.clone(),
                resume_var: capture.resume_var,
                resume_pc: capture.resume_pc,
                expected_schema: capture.expected_schema,
                fuel_remaining: state.fuel_remaining,
                effect_depth: capture.effect_depth,
            },
            observations: capture.observations,
            allowed_decisions,
        })
    }

    fn push_call_frame(
        &self,
        state: &mut ProgramState,
        function: String,
        args: Vec<Value>,
        return_to: ReturnSlot,
    ) -> Result<()> {
        let target = state
            .program
            .functions
            .get(&function)
            .ok_or_else(|| anyhow!("function {function} does not exist"))?;
        if target.params.len() != args.len() {
            return Err(anyhow!(
                "function {function} expects {} args but got {}",
                target.params.len(),
                args.len()
            ));
        }

        let env = bind_params(&target.params, args);
        state.stack.push(RuntimeFrame {
            function: function.clone(),
            pc: 0,
            env,
            return_to: Some(return_to),
        });
        self.trace.emit(
            "enter_function",
            &state.trace_id,
            json!({
                "function": function,
                "stack_depth": state.stack.len(),
            }),
        );
        Ok(())
    }

    fn push_map_item_frame(
        &self,
        state: &mut ProgramState,
        function: String,
        item: Value,
        return_to: ReturnSlot,
    ) -> Result<()> {
        let target = state
            .program
            .functions
            .get(&function)
            .ok_or_else(|| anyhow!("function {function} does not exist"))?;
        if target.params.len() != 1 {
            return Err(anyhow!(
                "map target function {function} must have exactly one param"
            ));
        }
        let mut env = Map::new();
        env.insert(target.params[0].clone(), item.clone());
        if let ReturnSlot::MapElement { item_var, .. } = &return_to {
            env.insert(item_var.clone(), item.clone());
        }
        env.insert(INPUT_VAR.to_owned(), item);
        state.stack.push(RuntimeFrame {
            function: function.clone(),
            pc: 0,
            env,
            return_to: Some(return_to),
        });
        self.trace.emit(
            "enter_function",
            &state.trace_id,
            json!({
                "function": function,
                "stack_depth": state.stack.len(),
            }),
        );
        Ok(())
    }

    fn return_from_function(&self, state: &mut ProgramState, output: Value) -> Result<StepOutcome> {
        let frame = state.top_frame()?.clone();
        let function = state.current_function()?.clone();
        validate_value(&function.output_schema, &output)
            .with_context(|| format!("function {} output failed schema", frame.function))?;

        self.trace.emit(
            "return_from_function",
            &state.trace_id,
            json!({
                "function": &frame.function,
                "stack_depth": state.stack.len(),
            }),
        );

        let Some(return_to) = frame.return_to else {
            validate_value(&state.program.output_schema, &output)
                .context("program output failed schema")?;
            self.trace.emit(
                "program_finished",
                &state.trace_id,
                json!({
                    "program_id": &state.program.program_id,
                    "version": &state.program.version,
                    "pending_patch_count": state.pending_patches.len(),
                }),
            );
            return Ok(StepOutcome::Finished(output));
        };

        state.stack.pop();
        match return_to {
            ReturnSlot::Call {
                caller_pc,
                var,
                expected_schema,
                ..
            } => {
                if let Some(expected_schema) = expected_schema {
                    validate_value(&expected_schema, &output)
                        .context("dynamic call output failed fragment output_schema")?;
                }
                let caller = state.top_frame_mut()?;
                caller.env.insert(var, output);
                caller.pc = caller_pc;
            }
            ReturnSlot::MapElement {
                caller_pc,
                out,
                map_index,
                item_var,
                function,
                items,
                mut results,
                ..
            } => {
                self.trace.emit(
                    "map_item_done",
                    &state.trace_id,
                    json!({ "function": &function, "index": map_index }),
                );
                results.push(output);
                let next_index = map_index + 1;
                if next_index < items.len() {
                    self.trace.emit(
                        "map_item_start",
                        &state.trace_id,
                        json!({ "function": &function, "index": next_index }),
                    );
                    let return_to = ReturnSlot::MapElement {
                        caller_function: state.top_frame()?.function.clone(),
                        caller_pc,
                        out,
                        map_index: next_index,
                        item_var,
                        function: function.clone(),
                        items: items.clone(),
                        results,
                    };
                    self.push_map_item_frame(
                        state,
                        function,
                        items[next_index].clone(),
                        return_to,
                    )?;
                    return Ok(StepOutcome::Continue);
                }

                let caller = state.top_frame_mut()?;
                caller.env.insert(out, Value::Array(results));
                caller.pc = caller_pc;
            }
        }

        Ok(StepOutcome::Continue)
    }

    fn install_fragment(
        &self,
        state: &mut ProgramState,
        fragment: ProgramFragment,
    ) -> Result<String> {
        if state.fragment_count >= state.budget.max_program_fragments {
            return Err(anyhow!(
                "program fragment limit {} exceeded",
                state.budget.max_program_fragments
            ));
        }
        for permission in &fragment.allowed_effects {
            if !state.program.allowed_effects.contains(permission) {
                return Err(anyhow!(
                    "program fragment declares effect {} not allowed by program {}",
                    effect_permission_name(permission),
                    state.program.program_id
                ));
            }
        }
        let prefix = format!(
            "__fragment_{}__",
            uuid::Uuid::new_v4().to_string().replace('-', "")
        );
        state.fragment_count += 1;

        let mut functions = BTreeMap::new();
        for (name, mut function) in fragment.functions {
            prefix_function_refs(&mut function, &prefix);
            functions.insert(format!("{prefix}{name}"), function);
        }
        let entry = format!("{prefix}{}", fragment.entry);
        for (name, function) in functions {
            if state.program.functions.contains_key(&name) {
                return Err(anyhow!("generated fragment function {name} already exists"));
            }
            state.program.functions.insert(name, function);
        }
        self.trace.emit(
            "program_fragment_installed",
            &state.trace_id,
            json!({
                "entry": &entry,
                "fragment_index": state.fragment_count - 1,
            }),
        );
        Ok(entry)
    }
}

fn effect_permission_name(permission: &crate::program::EffectPermission) -> &'static str {
    match permission {
        crate::program::EffectPermission::ModelTask { .. } => "model_task",
        crate::program::EffectPermission::Think => "think",
        crate::program::EffectPermission::CompileProgram { .. } => "compile_program",
        crate::program::EffectPermission::LocalTool { .. } => "local_tool",
    }
}

fn continuation_summary(continuation: &Continuation) -> ContinuationSummary {
    ContinuationSummary {
        boundary_id: continuation.boundary_id.clone(),
        program_id: continuation.program_id.clone(),
        continuation_id: continuation.continuation_id.clone(),
        current_function: continuation
            .stack
            .last()
            .map(|frame| frame.function.clone()),
        resume_pc: continuation.resume_pc,
        resume_var: continuation.resume_var.clone(),
        stack_depth: continuation.stack.len(),
        expected_schema: continuation.expected_schema.clone(),
    }
}

fn bind_params(params: &[String], args: Vec<Value>) -> Map<String, Value> {
    let mut env = Map::new();
    for (param, arg) in params.iter().zip(args) {
        env.insert(param.clone(), arg.clone());
    }
    if let Some(first) = params.first().and_then(|param| env.get(param)).cloned() {
        env.insert(INPUT_VAR.to_owned(), first);
    }
    env
}

fn validate_dynamic_fragment_input_contract(
    fragment: &ProgramFragment,
    args: &[Value],
) -> Result<()> {
    let entry = fragment
        .functions
        .get(&fragment.entry)
        .ok_or_else(|| anyhow!("fragment entry function {} does not exist", fragment.entry))?;

    if entry.params.len() != 1 {
        return Err(anyhow!(
            "dynamic fragment entry {} has {} params; fragment input_schema can only be enforced for exactly one entry parameter",
            fragment.entry,
            entry.params.len()
        ));
    }
    if args.len() != 1 {
        return Err(anyhow!(
            "dynamic fragment entry {} expects 1 arg but got {}",
            fragment.entry,
            args.len()
        ));
    }

    validate_value(&fragment.input_schema, &args[0])
        .context("dynamic call input failed fragment input_schema")
}

fn prefix_function_refs(function: &mut FunctionDef, prefix: &str) {
    for instr in &mut function.body {
        match instr {
            Instr::Call { function, .. } | Instr::Map { function, .. } => {
                *function = format!("{prefix}{function}");
            }
            Instr::Let { .. }
            | Instr::Project { .. }
            | Instr::Perform { .. }
            | Instr::Guard { .. }
            | Instr::Branch { .. }
            | Instr::Jump { .. }
            | Instr::CallDynamic { .. }
            | Instr::Return { .. } => {}
        }
    }
}

impl<W, S> Runtime<W, S> {
    fn emit_local_tool_result_trace(
        &self,
        state: &ProgramState,
        effect: &EffectCall,
        value: &Value,
    ) {
        let EffectCall::LocalTool { tool_name, .. } = effect else {
            return;
        };
        if tool_name != FAST_PATH_APPLY_TOOL_NAME {
            return;
        }
        let hit = value.get("hit").and_then(Value::as_bool).unwrap_or(false);
        self.trace.emit(
            if hit {
                "fast_path_hit"
            } else {
                "fast_path_miss"
            },
            &state.trace_id,
            json!({
                "tool": tool_name,
                "rule_id": value.get("rule_id").and_then(Value::as_str),
            }),
        );
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
            for (name, value) in fields {
                object.insert(name.clone(), eval_expr(env, value)?);
            }
            Ok(Value::Object(object))
        }
        JsonExpr::Array { items } => Ok(Value::Array(eval_exprs(env, items)?)),
    }
}

fn eval_exprs(env: &Map<String, Value>, exprs: &[JsonExpr]) -> Result<Vec<Value>> {
    exprs.iter().map(|expr| eval_expr(env, expr)).collect()
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
        GuardExpr::FieldEquals { var, path, value } => {
            let Some(root) = env.get(var) else {
                return Ok(false);
            };
            Ok(value_at_path(root, path).is_some_and(|actual| actual == value))
        }
        GuardExpr::FieldIsTruthy { var, path } => {
            let Some(root) = env.get(var) else {
                return Ok(false);
            };
            Ok(value_at_path(root, path).is_some_and(value_is_truthy))
        }
    }
}

fn guard_resume_contract(condition: &GuardExpr) -> (Option<String>, Value) {
    match condition {
        GuardExpr::VarExists { name } => (Some(name.clone()), json!({})),
        GuardExpr::JsonSchemaValid { var, schema } => (Some(var.clone()), schema.clone()),
        GuardExpr::FieldEquals { var, .. } | GuardExpr::FieldIsTruthy { var, .. } => {
            (Some(var.clone()), json!({}))
        }
    }
}

fn value_at_path<'a>(root: &'a Value, path: &[String]) -> Option<&'a Value> {
    let mut cursor = root;
    for segment in path {
        cursor = cursor.as_object()?.get(segment)?;
    }
    Some(cursor)
}

fn value_is_truthy(value: &Value) -> bool {
    match value {
        Value::Bool(value) => *value,
        Value::Null => false,
        Value::Number(number) => number.as_f64().is_some_and(|value| value != 0.0),
        Value::String(value) => !value.is_empty(),
        Value::Array(values) => !values.is_empty(),
        Value::Object(values) => !values.is_empty(),
    }
}

fn accepted_by_policy(
    expected_schema: &Value,
    acceptance: &AcceptancePolicy,
    value: &Value,
    confidence: f32,
) -> bool {
    let confidence_valid = is_probability(confidence);
    let confidence_pass = acceptance
        .min_confidence
        .map(|min| confidence_valid && confidence >= min)
        .unwrap_or(confidence_valid);
    let schema_pass =
        !acceptance.require_schema_valid || validate_value(expected_schema, value).is_ok();
    confidence_pass && schema_pass
}

fn trace_return_value_schema_valid(expected_schema: &Value, value: &Value) -> bool {
    validate_value(expected_schema, value).is_ok()
}

fn validate_effect_input_contract(effect: &EffectCall, input: &Value) -> Result<()> {
    let EffectCall::LocalTool {
        tool_name,
        args_schema,
    } = effect
    else {
        return Ok(());
    };
    validate_value(args_schema, input)
        .with_context(|| format!("local_tool {tool_name} input failed args_schema"))
}

fn is_handler_failure(err: &anyhow::Error) -> bool {
    err.chain().any(|cause| {
        matches!(
            cause.to_string().as_str(),
            "weak_model handler failed"
                | "strong_model handler failed"
                | "local_tool handler failed"
        )
    })
}

fn root_error_message(err: &anyhow::Error) -> String {
    err.chain()
        .last()
        .map(ToString::to_string)
        .unwrap_or_else(|| err.to_string())
}

fn ensure_captured_decision_allowed(
    effect_frame: Option<&EffectFrame>,
    decision: &HandlerDecision,
) -> Result<()> {
    let Some(frame) = effect_frame else {
        return Ok(());
    };
    let Some(allowed) = allowed_decision_for(decision) else {
        return Err(anyhow!(
            "handler decision {} is not allowed for captured effect frame {}",
            decision.decision_name(),
            frame.effect_id
        ));
    };
    if frame.allowed_decisions.contains(&allowed) {
        Ok(())
    } else {
        Err(anyhow!(
            "handler decision {} is not allowed for captured effect frame {}",
            decision.decision_name(),
            frame.effect_id
        ))
    }
}

fn ensure_handler_decision_allowed(
    handler_name: &str,
    effect: &EffectCall,
    effect_frame: Option<&EffectFrame>,
    decision: &HandlerDecision,
) -> Result<()> {
    ensure_captured_decision_allowed(effect_frame, decision)?;
    if matches!(decision, HandlerDecision::ReturnProgram { .. })
        && !matches!(effect, EffectCall::CompileProgram { .. })
    {
        return Err(anyhow!(
            "handler decision return_program is only allowed for compile_program request"
        ));
    }
    if effect_frame.is_none()
        && handler_name != "strong_model"
        && matches!(decision, HandlerDecision::ReturnProgramPatch { .. })
    {
        return Err(anyhow!(
            "handler decision return_program_patch is not allowed for {handler_name} request"
        ));
    }
    Ok(())
}

fn allowed_decision_for(decision: &HandlerDecision) -> Option<AllowedDecision> {
    match decision {
        HandlerDecision::ReturnValue { .. } => Some(AllowedDecision::ReturnValue),
        HandlerDecision::RequestEffect { .. } => Some(AllowedDecision::RequestEffect),
        HandlerDecision::ReturnProgram { .. } => None,
        HandlerDecision::ReturnProgramFragment { .. } => {
            Some(AllowedDecision::ReturnProgramFragment)
        }
        HandlerDecision::ReturnProgramPatch { .. } => Some(AllowedDecision::ReturnProgramPatch),
        HandlerDecision::Abort { .. } => Some(AllowedDecision::Abort),
    }
}

fn schema_accepts_null(schema: &Value) -> bool {
    validate_value(schema, &Value::Null).is_ok()
}

fn source_for_effect(effect: &EffectCall) -> ObservationSource {
    match effect {
        EffectCall::ModelTask {
            strength: ModelStrength::Weak,
            ..
        } => ObservationSource::WeakModel,
        EffectCall::Think { .. } => ObservationSource::StrongModel,
        EffectCall::ModelTask {
            strength: ModelStrength::Strong,
            ..
        } => ObservationSource::StrongModel,
        EffectCall::CompileProgram {
            strength: ModelStrength::Strong,
            ..
        } => ObservationSource::StrongModel,
        EffectCall::CompileProgram {
            strength: ModelStrength::Weak,
            ..
        } => ObservationSource::WeakModel,
        EffectCall::LocalTool { .. } => ObservationSource::LocalTool,
    }
}

fn program_with_compile_contract(effect: &EffectCall, mut program: Program) -> Program {
    let EffectCall::CompileProgram {
        input_schema,
        output_schema,
        ..
    } = effect
    else {
        return program;
    };

    program.input_schema = input_schema.clone();
    program.output_schema = output_schema.clone();
    program
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
