# cps-llm-demo

Minimal Rust CLI demo for a defunctionalized CPS / typed-effect LLM runtime.

This is a runtime semantics demo, not a product agent. A strong model can compile a task spec into a small JSON `Program` IR. The Rust runtime interprets that program, calls the weak model only when it executes `Instr::WeakCall`, captures unresolved typed effects as `EffectFrame`, asks strong Think to handle the stuck continuation, validates the returned value against `continuation.expected_schema`, and resumes from `continuation.pc`.

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
cargo run -- probe-models
cargo run -- run-program --program examples/message_action.program.json --input examples/messages.json --trace-json
cargo run -- run-program --program examples/message_action.two_stage.program.json --input examples/messages.json --trace-json
cargo run -- compile-run --task examples/message_action.task.md --input examples/messages.json --trace-json
```

`run-program` skips compilation and runs an existing Program fixture. `compile-run` asks the strong model to compile the task spec into Program IR, then runs that same runtime. Both commands write final JSON output to stdout. With `--trace-json`, runtime trace JSONL is written to stderr:

```bash
cargo run -- run-program --program examples/message_action.two_stage.program.json --input examples/messages.json --trace-json 1>out.json 2>trace.jsonl
```

## What To Look For

The trace should show program execution, not a fixed weak-to-strong router:

```text
program_start
exec_instr pc=0 op=weak_call
weak_call
weak_result
exec_instr pc=1 op=weak_call
weak_result
capture_continuation pc=2 resume_var=draft
strong_think decision=request_weak_probe
weak_probe
strong_think decision=resume_with_value
resume_continuation pc=2
exec_instr pc=2 op=finish
program_finished
```

The important shape is:

```text
strong compiles program
runtime interprets program
program performs weak semantic effects
failed effect captures continuation
strong handles the stuck continuation
runtime resumes program
```

## Why This Is CPS

- `Continuation` is serializable data: `program_id`, `pc`, `resume_var`, `env`, and `expected_schema`.
- The runtime does not call weak by default. Weak runs only when the Program reaches `Instr::WeakCall`.
- The strong model is not the weak model's next step. It handles `EffectFrame` values captured from unresolved continuation frames.
- Strong Think returns a typed `ThinkDecision`, not free-form final output.
- `ResumeWithValue` is validated against `continuation.expected_schema` before execution resumes.

## API Shape

The client calls:

```text
POST {base_url}/responses
```

It uses the OpenAI-compatible Responses API with structured JSON schema output and supports trailing slashes in `--base-url`.

## Development

```bash
cargo fmt --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test
```
