use std::collections::{BTreeMap, BTreeSet};

use anyhow::{Result, anyhow};
use serde_json::Value;

use crate::program::{
    EffectCall, EffectPermission, FailureHandler, FunctionDef, GuardExpr, GuardFail, Instr,
    JsonExpr, ModelStrength, PatchOp, Program, ProgramFragment, ProgramPatch,
};

const MAX_INSTRUCTIONS_PER_FUNCTION: usize = 1024;
const INPUT_VAR: &str = "$input";

pub fn validate_program(program: &Program) -> Result<()> {
    validate_program_with_entry_input(program, true)
}

fn validate_program_with_entry_input(program: &Program, entry_binds_input: bool) -> Result<()> {
    let entry = program
        .functions
        .get(&program.entry)
        .ok_or_else(|| anyhow!("entry function {} does not exist", program.entry))?;
    if entry_binds_input && entry.params.len() > 1 {
        return Err(anyhow!(
            "entry function {} expects {} params; runtime supplies one input value",
            program.entry,
            entry.params.len()
        ));
    }

    validate_json_schema(&program.input_schema)
        .map_err(|err| anyhow!("program input_schema is not a valid JSON schema: {err}"))?;
    validate_json_schema(&program.output_schema)
        .map_err(|err| anyhow!("program output_schema is not a valid JSON schema: {err}"))?;

    validate_supported_effect_permissions(program)?;

    for (name, function) in &program.functions {
        let input_is_bound =
            !function.params.is_empty() || (entry_binds_input && name == &program.entry);
        validate_function(program, name, function, input_is_bound)?;
    }

    ensure_no_static_recursion(program)?;
    Ok(())
}

pub fn validate_fragment(fragment: &ProgramFragment) -> Result<()> {
    let program = Program {
        program_id: "fragment_validation".to_owned(),
        version: "fragment".to_owned(),
        entry: fragment.entry.clone(),
        input_schema: fragment.input_schema.clone(),
        output_schema: fragment.output_schema.clone(),
        functions: fragment.functions.clone(),
        allowed_effects: fragment.allowed_effects.clone(),
    };
    validate_program_with_entry_input(&program, false)
}

pub fn validate_patch(program: &Program, patch: &ProgramPatch) -> Result<Program> {
    if patch.target_program_id != program.program_id {
        return Err(anyhow!(
            "patch target_program_id {} does not match program {}",
            patch.target_program_id,
            program.program_id
        ));
    }

    let mut patched = program.clone();
    for operation in &patch.operations {
        apply_patch_op(&mut patched, operation)?;
    }
    patched.version = format!("{}+{}", program.version, patch.patch_id);
    validate_program(&patched)?;
    Ok(patched)
}

pub fn effect_allowed(allowed: &[EffectPermission], effect: &EffectCall) -> bool {
    allowed.iter().any(|permission| match (permission, effect) {
        (
            EffectPermission::ModelTask {
                strength: allowed_strength,
            },
            EffectCall::ModelTask { strength, .. },
        ) => allowed_strength == strength,
        (EffectPermission::Think, EffectCall::Think { .. }) => true,
        (
            EffectPermission::CompileProgram {
                strength: allowed_strength,
            },
            EffectCall::CompileProgram { strength, .. },
        ) => allowed_strength == strength,
        (
            EffectPermission::LocalTool {
                tool_name: allowed_tool,
            },
            EffectCall::LocalTool { tool_name, .. },
        ) => allowed_tool == tool_name,
        _ => false,
    })
}

fn validate_function(
    program: &Program,
    name: &str,
    function: &FunctionDef,
    input_is_bound: bool,
) -> Result<()> {
    if function.body.len() > MAX_INSTRUCTIONS_PER_FUNCTION {
        return Err(anyhow!(
            "function {name} has {} instructions, limit is {MAX_INSTRUCTIONS_PER_FUNCTION}",
            function.body.len()
        ));
    }

    validate_json_schema(&function.output_schema)
        .map_err(|err| anyhow!("function {name} output_schema is invalid: {err}"))?;

    if !matches!(function.body.last(), Some(Instr::Return { .. })) {
        return Err(anyhow!("function {name} must end with return"));
    }

    let mut defined = BTreeSet::new();
    for param in &function.params {
        ensure_not_reserved_input_binding(param, &format!("function {name} parameter"), None)?;
        if !defined.insert(param.clone()) {
            return Err(anyhow!("function {name} has duplicate parameter {param}"));
        }
    }
    if input_is_bound {
        defined.insert(INPUT_VAR.to_owned());
    }

    for (pc, instr) in function.body.iter().enumerate() {
        validate_instr(program, name, pc, instr, &defined)?;
        if let Some(out) = instr.output_var() {
            ensure_not_reserved_input_binding(out, "instruction output", Some(name))?;
            defined.insert(out.to_owned());
        }
        if let Some(repair_target) = guard_think_repair_target(instr) {
            defined.insert(repair_target.to_owned());
        }
    }

    Ok(())
}

fn validate_instr(
    program: &Program,
    function_name: &str,
    pc: usize,
    instr: &Instr,
    defined: &BTreeSet<String>,
) -> Result<()> {
    match instr {
        Instr::Let { expr, .. } => validate_expr(expr, defined),
        Instr::Project { from, .. } => validate_expr(from, defined),
        Instr::Perform {
            effect,
            input,
            expected_schema,
            acceptance,
            ..
        } => {
            validate_expr(input, defined)?;
            validate_json_schema(expected_schema).map_err(|err| {
                anyhow!("perform expected_schema is invalid at {function_name}:{pc}: {err}")
            })?;
            validate_supported_effect_call(effect, function_name, pc)?;
            if !effect_allowed(&program.allowed_effects, effect) {
                return Err(anyhow!(
                    "effect {} is not allowed at {function_name}:{pc}",
                    effect.kind_name()
                ));
            }
            if matches!(
                &acceptance.on_failure,
                FailureHandler::CaptureToThink { .. }
            ) {
                ensure_think_permission(
                    program,
                    function_name,
                    pc,
                    "capture_to_think failure handler",
                )?;
            }
            Ok(())
        }
        Instr::Guard { condition, on_fail } => {
            validate_guard_condition(
                condition,
                defined,
                matches!(on_fail, GuardFail::Think { .. }),
                function_name,
                pc,
            )?;
            if matches!(on_fail, GuardFail::Think { .. }) {
                ensure_think_permission(program, function_name, pc, "guard think repair")?;
                ensure_not_reserved_input_binding(
                    guard_repair_target(condition),
                    "guard think repair target",
                    Some(function_name),
                )?;
            }
            Ok(())
        }
        Instr::Branch {
            condition,
            then_pc,
            else_pc,
        } => {
            validate_guard_condition(condition, defined, false, function_name, pc)?;
            validate_forward_target(program, function_name, pc, *then_pc, "branch then_pc")?;
            validate_forward_target(program, function_name, pc, *else_pc, "branch else_pc")
        }
        Instr::Jump { pc: target_pc } => {
            validate_forward_target(program, function_name, pc, *target_pc, "jump pc")
        }
        Instr::Call { function, args, .. } => {
            let target = program.functions.get(function).ok_or_else(|| {
                anyhow!("call target function {function} does not exist at {function_name}:{pc}")
            })?;
            if target.params.len() != args.len() {
                return Err(anyhow!(
                    "call target {function} expects {} args but got {} at {function_name}:{pc}",
                    target.params.len(),
                    args.len()
                ));
            }
            validate_exprs(args, defined)
        }
        Instr::Map {
            items,
            function,
            item_var,
            ..
        } => {
            ensure_not_reserved_input_binding(item_var, "map item_var", Some(function_name))?;
            let target = program.functions.get(function).ok_or_else(|| {
                anyhow!("map target function {function} does not exist at {function_name}:{pc}")
            })?;
            if target.params.len() != 1 {
                return Err(anyhow!(
                    "map target {function} must have exactly one parameter at {function_name}:{pc}"
                ));
            }
            validate_expr(items, defined)
        }
        Instr::CallDynamic { fragment, args, .. } => {
            validate_expr(fragment, defined)?;
            validate_exprs(args, defined)
        }
        Instr::Return { value } => validate_expr(value, defined),
    }
}

fn validate_expr(expr: &JsonExpr, defined: &BTreeSet<String>) -> Result<()> {
    match expr {
        JsonExpr::Literal { .. } => Ok(()),
        JsonExpr::Var { name } => ensure_defined(name, defined),
        JsonExpr::Object { fields } => {
            for value in fields.values() {
                validate_expr(value, defined)?;
            }
            Ok(())
        }
        JsonExpr::Array { items } => validate_exprs(items, defined),
    }
}

fn validate_exprs(exprs: &[JsonExpr], defined: &BTreeSet<String>) -> Result<()> {
    for expr in exprs {
        validate_expr(expr, defined)?;
    }
    Ok(())
}

fn ensure_not_reserved_input_binding(name: &str, context: &str, owner: Option<&str>) -> Result<()> {
    if name != INPUT_VAR {
        return Ok(());
    }

    match owner {
        Some(owner) => Err(anyhow!(
            "{context} {INPUT_VAR} is reserved for runtime input in function {owner}"
        )),
        None => Err(anyhow!(
            "{context} {INPUT_VAR} is reserved for runtime input"
        )),
    }
}

fn validate_guard_condition(
    condition: &GuardExpr,
    defined: &BTreeSet<String>,
    allow_missing_var_exists: bool,
    function_name: &str,
    pc: usize,
) -> Result<()> {
    match condition {
        GuardExpr::VarExists { name } => {
            if !allow_missing_var_exists {
                ensure_defined(name, defined)?;
            }
            Ok(())
        }
        GuardExpr::JsonSchemaValid { var, schema } => {
            ensure_defined(var, defined)?;
            validate_json_schema(schema)
                .map_err(|err| anyhow!("guard schema is invalid at {function_name}:{pc}: {err}"))
        }
        GuardExpr::FieldEquals { var, .. } | GuardExpr::FieldIsTruthy { var, .. } => {
            ensure_defined(var, defined)
        }
    }
}

fn validate_forward_target(
    program: &Program,
    function_name: &str,
    pc: usize,
    target_pc: usize,
    label: &str,
) -> Result<()> {
    let body_len = program
        .functions
        .get(function_name)
        .map(|function| function.body.len())
        .ok_or_else(|| anyhow!("function {function_name} does not exist"))?;
    if target_pc >= body_len {
        return Err(anyhow!(
            "{label} {target_pc} is outside function {function_name} body"
        ));
    }
    if target_pc <= pc {
        return Err(anyhow!(
            "{label} {target_pc} must be forward-only from {function_name}:{pc}"
        ));
    }
    Ok(())
}

fn guard_repair_target(condition: &GuardExpr) -> &str {
    match condition {
        GuardExpr::VarExists { name } => name,
        GuardExpr::JsonSchemaValid { var, .. }
        | GuardExpr::FieldEquals { var, .. }
        | GuardExpr::FieldIsTruthy { var, .. } => var,
    }
}

fn guard_think_repair_target(instr: &Instr) -> Option<&str> {
    match instr {
        Instr::Guard {
            condition,
            on_fail: GuardFail::Think { .. },
        } => Some(guard_repair_target(condition)),
        _ => None,
    }
}

fn validate_supported_effect_permissions(program: &Program) -> Result<()> {
    for permission in &program.allowed_effects {
        match permission {
            EffectPermission::LocalTool { tool_name } => {
                if tool_name.trim().is_empty() {
                    return Err(anyhow!("local_tool permission has empty tool_name"));
                }
            }
            EffectPermission::CompileProgram {
                strength: ModelStrength::Weak,
            } => {
                return Err(anyhow!(
                    "weak compile_program effects are not supported until weak compile handlers are implemented"
                ));
            }
            EffectPermission::ModelTask { .. }
            | EffectPermission::Think
            | EffectPermission::CompileProgram {
                strength: ModelStrength::Strong,
            } => {}
        }
    }
    Ok(())
}

fn validate_supported_effect_call(
    effect: &EffectCall,
    function_name: &str,
    pc: usize,
) -> Result<()> {
    match effect {
        EffectCall::LocalTool { tool_name, .. } => {
            if tool_name.trim().is_empty() {
                return Err(anyhow!(
                    "local_tool effect has empty tool_name at {function_name}:{pc}"
                ));
            }
        }
        EffectCall::CompileProgram {
            strength: ModelStrength::Weak,
            ..
        } => {
            return Err(anyhow!(
                "weak compile_program effects are not supported until weak compile handlers are implemented at {function_name}:{pc}"
            ));
        }
        EffectCall::CompileProgram {
            strength: ModelStrength::Strong,
            input_schema,
            output_schema,
            ..
        } => {
            validate_json_schema(input_schema).map_err(|err| {
                anyhow!("compile_program input_schema is invalid at {function_name}:{pc}: {err}")
            })?;
            validate_json_schema(output_schema).map_err(|err| {
                anyhow!("compile_program output_schema is invalid at {function_name}:{pc}: {err}")
            })?;
        }
        EffectCall::ModelTask { .. } | EffectCall::Think { .. } => {}
    }
    Ok(())
}

fn ensure_think_permission(
    program: &Program,
    function_name: &str,
    pc: usize,
    context: &str,
) -> Result<()> {
    if program
        .allowed_effects
        .iter()
        .any(|permission| matches!(permission, EffectPermission::Think))
    {
        Ok(())
    } else {
        Err(anyhow!(
            "{context} requires think effect permission at {function_name}:{pc}"
        ))
    }
}

fn ensure_defined(name: &str, defined: &BTreeSet<String>) -> Result<()> {
    if defined.contains(name) {
        Ok(())
    } else {
        Err(anyhow!("variable {name} is used before it is defined"))
    }
}

fn validate_json_schema(schema: &Value) -> Result<()> {
    jsonschema::validator_for(schema)?;
    Ok(())
}

fn ensure_no_static_recursion(program: &Program) -> Result<()> {
    let graph = program
        .functions
        .iter()
        .map(|(name, function)| {
            (
                name.clone(),
                function
                    .body
                    .iter()
                    .filter_map(|instr| match instr {
                        Instr::Call { function, .. } | Instr::Map { function, .. } => {
                            Some(function.clone())
                        }
                        _ => None,
                    })
                    .collect::<Vec<_>>(),
            )
        })
        .collect::<BTreeMap<_, _>>();

    let mut visiting = BTreeSet::new();
    let mut visited = BTreeSet::new();
    for function in program.functions.keys() {
        dfs_no_cycle(function, &graph, &mut visiting, &mut visited)?;
    }
    Ok(())
}

fn dfs_no_cycle(
    function: &str,
    graph: &BTreeMap<String, Vec<String>>,
    visiting: &mut BTreeSet<String>,
    visited: &mut BTreeSet<String>,
) -> Result<()> {
    if visited.contains(function) {
        return Ok(());
    }
    if !visiting.insert(function.to_owned()) {
        return Err(anyhow!(
            "recursive function call involving {function} is not allowed"
        ));
    }
    for callee in graph.get(function).into_iter().flatten() {
        dfs_no_cycle(callee, graph, visiting, visited)?;
    }
    visiting.remove(function);
    visited.insert(function.to_owned());
    Ok(())
}

fn apply_patch_op(program: &mut Program, operation: &PatchOp) -> Result<()> {
    match operation {
        PatchOp::ReplaceInstruction {
            function,
            pc,
            instr,
        } => {
            let body = function_body_mut(program, function)?;
            let slot = body
                .get_mut(*pc)
                .ok_or_else(|| anyhow!("replace pc {pc} is outside function {function}"))?;
            *slot = instr.clone();
            Ok(())
        }
        PatchOp::InsertInstruction {
            function,
            pc,
            instr,
        } => {
            let body = function_body_mut(program, function)?;
            if *pc > body.len() {
                return Err(anyhow!("insert pc {pc} is outside function {function}"));
            }
            body.insert(*pc, instr.clone());
            Ok(())
        }
        PatchOp::AddFunction { name, function } => {
            if program.functions.contains_key(name) {
                return Err(anyhow!("patch cannot override existing function {name}"));
            }
            program.functions.insert(name.clone(), function.clone());
            Ok(())
        }
        PatchOp::UpdateAcceptancePolicy {
            function,
            pc,
            acceptance,
        } => {
            let body = function_body_mut(program, function)?;
            let instr = body
                .get_mut(*pc)
                .ok_or_else(|| anyhow!("acceptance pc {pc} is outside function {function}"))?;
            match instr {
                Instr::Perform {
                    acceptance: existing,
                    ..
                } => {
                    *existing = acceptance.clone();
                    Ok(())
                }
                _ => Err(anyhow!(
                    "update_acceptance_policy target {function}:{pc} is not perform"
                )),
            }
        }
    }
}

fn function_body_mut<'a>(program: &'a mut Program, function: &str) -> Result<&'a mut Vec<Instr>> {
    program
        .functions
        .get_mut(function)
        .map(|function| &mut function.body)
        .ok_or_else(|| anyhow!("function {function} does not exist"))
}
