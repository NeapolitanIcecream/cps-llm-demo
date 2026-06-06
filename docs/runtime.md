# Runtime Contract

The runtime executes a typed `Program` IR. It does not act as a model pipeline where one model freely calls another model. Rust validates the program, schedules effects, validates handler decisions, records traces, and resumes saved continuations.

## Program Shape

A `Program` has:

- `program_id` and `version` for registry tracking.
- `entry`, which names the first function.
- `input_schema` and `output_schema`.
- `functions`, each with params, an output schema, and instruction body.
- `allowed_effects`, which controls what effects the runtime may schedule.

Supported instructions include `let`, `project`, `perform`, `guard`, `branch`, `jump`, `call`, `map`, `call_dynamic`, and `return`.

## Effects

`perform` can request these effect kinds:

- `model_task`: a weak or strong model task.
- `think`: strong-model reasoning over a captured state or guard failure.
- `compile_program`: a model returns `Program` IR.
- `local_tool`: a deterministic local tool.

The built-in local tools are:

- `fast_path_apply`
- `validator_apply`
- `template_emit`

The runtime validates local-tool names before dispatch.

## Handler Decisions

Models return typed `HandlerDecision` values:

- `return_value`
- `request_effect`
- `return_program`
- `return_program_fragment`
- `return_program_patch`
- `abort`

The runtime validates each decision against the expected schema and the current effect context. A weak handler may request another effect only when the requested effect is allowed by the program and the runtime budget.

## Continuations

When the runtime cannot accept an effect result directly, it captures a continuation. A continuation is structured data, not a prose log. It includes the boundary id, program id, runtime frames, resume variable, resume program counter, expected schema, fuel, and effect depth.

The strong handler can inspect the captured state, return a value, request another allowed effect, or abort. If the returned value validates, the runtime resumes the saved stack at the recorded program counter.

## Invariants

- Models never call each other directly. They return decisions; the runtime schedules effects.
- Weak model calls happen only when an allowed weak effect is scheduled.
- Strong model calls happen only when an allowed strong effect is scheduled, a `think` effect is reached, a strong compile is requested, or a captured continuation requires strong handling.
- `Program.allowed_effects` is enforced before direct `perform` effects and nested effects requested through `request_effect`.
- Handler return values are schema-validated before use.
- Action payloads do not receive runtime provenance fields. Provenance is recorded in observations, traces, metrics, and model-call logs.
- Runtime code does not branch on notification, calendar, OTP, message, or business keywords. Domain behavior belongs in task files, program fixtures, patches, prompts, and test data.
- `run-program` and `compile-run` share the same execution path once a program has been validated.

## Trace Events

Use `--trace-json` to write trace JSONL to stderr:

```bash
cargo run -- run-program \
  --program examples/message_action.v2.program.json \
  --input examples/messages.json \
  --trace-json \
  1>out.json \
  2>trace.jsonl
```

Useful trace events include:

- `program_validated`
- `program_start`
- `enter_function`
- `exec_instr`
- `perform_effect`
- `handler_request`
- `handler_decision`
- `capture_continuation`
- `effect_accepted`
- `resume_continuation`
- `return_from_function`
- `program_finished`
- `program_aborted`

Replay checks that trace JSONL is parseable, every captured continuation is resumed or aborted, every resume references a known continuation id, and strong `return_value` entries are not schema-invalid:

```bash
cargo run -- replay --trace trace.jsonl
```

Replay is a guard over trace structure. It is not full deterministic replay.

## Patches

A `ProgramPatch` is validated before it is installed. In the notification-triage workflow, a patch can add a semantic fast path that uses weak semantic matching, local validation, and template emission before falling back to normal model handling.

Installed patches advance the workflow program version in the state directory. Metrics and comparison commands use those versions to check whether the post-patch workflow actually ran the new program.
