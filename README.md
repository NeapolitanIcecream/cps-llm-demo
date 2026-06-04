# cps-llm-demo

Rust CLI demo for a defunctionalized CPS / typed-effect LLM runtime.

This is a runtime semantics demo, not a product agent. A strong model can compile a task spec into JSON `Program` IR. The Rust runtime validates and interprets that program, executes functions and `Map`, performs typed effects, captures stuck continuations as data, schedules weak/strong handlers, validates typed handler decisions, and resumes the saved program stack.

## Core Invariants

- Models never call each other directly. They return typed `HandlerDecision` values such as `return_value`, `request_effect`, `return_program_fragment`, `return_program_patch`, or `abort`.
- Weak runs only when the runtime schedules an allowed weak effect, such as direct `Perform(ModelTask { strength: Weak })` or an allowed nested `HandlerDecision::RequestEffect` targeting a weak model task.
- Strong runs only when the runtime schedules an allowed strong effect, such as direct `Perform(ModelTask { strength: Strong })`, `Perform(Think)`, `CompileProgram { strength: Strong }`, runtime handling of a captured `Think` frame, or an allowed nested `HandlerDecision::RequestEffect`.
- `Program.allowed_effects` is enforced before scheduling direct `Perform` effects and runtime-scheduled nested effects requested via `HandlerDecision::RequestEffect`.
- The built-in `local_tool` names are `fast_path_apply` and `validator_apply`; validation rejects other local-tool names before runtime dispatch.
- `Continuation` contains a serializable stack: boundary id, program id, runtime frames, resume var, resume pc, expected schema, fuel, and effect depth.
- Rust runtime code does not branch on message/task/calendar/OTP/business keywords. Domain semantics live in task files, Program fixtures, prompts, and test data.
- Handler return values are schema-validated as returned. Action drafts are business payloads; runtime provenance is recorded in observations and trace metadata, not injected into returned JSON.
- `run-program` and `compile-run` share the same runtime path.

## Setup

```bash
cp .env.example .env
# edit .env and set OPENAI_API_KEY
```

Defaults:

```bash
OPENAI_BASE_URL=https://api.openai.com/v1
CPS_WEAK_MODEL=gpt-5.4-mini
CPS_STRONG_MODEL=gpt-5.5
```

All model IDs and the OpenAI-compatible base URL can be overridden with CLI flags or env vars.

## Commands

```bash
cargo run -- schema
cargo run -- validate-program --program examples/message_action.v2.program.json
cargo run -- run-program --program examples/message_action.v2.program.json --input examples/messages.json --trace-json
cargo run -- run-program --program examples/fractal_subprogram.program.json --input examples/messages.json --trace-json
cargo run -- compile-run --task examples/message_action.task.md --input examples/messages.json --trace-json
cargo run -- replay --trace traces/example.trace.jsonl
```

`run-program` skips compilation and runs an existing Program fixture. `compile-run` asks the strong compiler handler to return Program IR, validates it, then runs the same runtime. Both commands write final JSON output to stdout. With `--trace-json`, runtime trace JSONL is written to stderr:

```bash
cargo run -- run-program --program examples/message_action.v2.program.json --input examples/messages.json --trace-json 1>out.json 2>trace.jsonl
cargo run -- replay --trace trace.jsonl
```

## System Value Loop Fixture

The system-layer commands run a workflow across JSONL event streams, store program versions and metrics in a local state directory, install a typed patch, and compare the CPS path with a strong-direct baseline.

This transcript uses 100-event round fixtures and fixture handlers when `OPENAI_API_KEY` is unset, so it does not call external models:

```bash
STATE=/tmp/cps-llm-state

cargo run -- init-workflow \
  --workflow notification_triage \
  --program examples/notification_triage.v1.program.json \
  --state-dir $STATE

cargo run -- run-stream \
  --workflow notification_triage \
  --events examples/notification_triage.round1.jsonl \
  --state-dir $STATE \
  --trace-json

cargo run -- optimize \
  --workflow notification_triage \
  --patch examples/notification_triage.fast_path.patch.json \
  --state-dir $STATE

cargo run -- run-stream \
  --workflow notification_triage \
  --events examples/notification_triage.round2.jsonl \
  --state-dir $STATE \
  --trace-json

cargo run -- baseline-strong-direct \
  --workflow notification_triage \
  --task examples/notification_triage.task.md \
  --events examples/notification_triage.round1.jsonl \
  --state-dir $STATE

cargo run -- metrics-report \
  --workflow notification_triage \
  --state-dir $STATE

# Use the run_id values returned by run-stream and baseline-strong-direct.
cargo run -- compare-runs \
  --baseline-run $BASELINE_RUN \
  --before-run $ROUND1_RUN \
  --after-run $ROUND2_RUN \
  --state-dir $STATE
```

`compare-runs` returns the merge-readiness booleans for the fixture loop:

```json
{
  "strong_calls_per_event_decreased": true,
  "fast_path_coverage_increased": true,
  "continuation_frame_size_bounded": true,
  "program_version_advanced": true
}
```

The report includes `latest_program_version`, per-run metrics, and a summary that checks whether fast-path hit rate increased, StrongThink rate decreased, the strong-direct baseline used one strong call per event, and continuation frames stayed under the configured byte limit.

`optimize --patch` installs a fixture `ProgramPatch` after validating it and evaluating it against stored stream events. Without `--patch`, `optimize` reads profile fingerprints, asks the strong handler for a typed `ProgramPatch`, evaluates the returned patch, and installs it only if the evaluation passes.

## What To Look For

Demo A, ordinary program execution:

```text
program_validated
program_start
enter_function
exec_instr op=project
exec_instr op=return
program_finished
```

No handler request appears. That proves the runtime is not a model pipeline.

Demo B, strong-generated program with weak effects:

```text
program_validated
program_start
exec_instr function=main op=map
map_item_start index=0
enter_function function=process_message
exec_instr function=process_message pc=0 op=perform
handler_request handler=weak_model effect=model_task task=classify_intent
handler_decision handler=weak_model decision=return_value
exec_instr function=process_message pc=1 op=perform
handler_request handler=weak_model effect=model_task task=extract_action_draft_from_intent
handler_decision handler=weak_model decision=return_value schema_valid=true
capture_continuation function=process_message resume_pc=2 resume_var=draft map_index=0
handler_request handler=strong_model effect=think
handler_decision handler=strong_model decision=return_value
effect_accepted effect=model_task captured=true
resume_continuation pc=2 resume_var=draft
return_from_function function=process_message
map_item_done index=0
program_finished
```

The strong model handles a stuck continuation, not the whole task.

Demo C, fractal nested effect:

```text
handler_request handler=weak_model effect=model_task task=generate_processor_program
handler_decision handler=weak_model decision=return_program_fragment
program_fragment_validated
program_fragment_installed
enter_function function=__fragment_0__generated_processor
exec_instr function=__fragment_0__generated_processor op=perform effect=think
handler_request handler=strong_model effect=think
handler_decision handler=strong_model decision=return_value
effect_accepted effect=think captured=false
return_from_function function=__fragment_0__generated_processor
```

Weak-generated code can contain a strong `Think` effect, but the call is still scheduled by the runtime.

## Trace Replay

`replay` is a sanity checker for trace JSONL. It verifies that trace events are parseable, every `capture_continuation` has a matching `resume_continuation` or abort, every resume references a known continuation id, and strong handler `return_value` trace entries are not schema-invalid.

This is not full deterministic replay yet; it is the v1.0 guard that proves captured continuations are machine-readable trace facts rather than prose logs.

## Development

```bash
cargo fmt --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test
```
