# cps-llm-demo

Minimal Rust CLI demo for a defunctionalized CPS / typed-effect LLM runtime.

This is not a product agent. It is a runtime semantics demo: a weak classifier fills a typed semantic hole, and the runtime captures a stuck continuation as an `EffectFrame` when structural guard conditions require deeper handling. The Think handler receives that frame, returns a typed `ThinkDecision`, and the runtime resumes.

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
CPS_THRESHOLD=0.75
```

All model IDs and the OpenAI-compatible base URL can be overridden with CLI flags or env vars.

## Commands

```bash
cargo run -- schema
cargo run -- probe-models
cargo run -- run examples/messages.json --trace-json
cargo run -- run examples/messages.json --trace-json --capture-policy always-after-weak
```

`run` writes final `ActionDraft[]` JSON to stdout. With `--trace-json`, it writes runtime trace JSONL to stderr, so the streams can be split:

```bash
cargo run -- run examples/messages.json --trace-json --capture-policy always-after-weak 1>out.json 2>trace.jsonl
```

`--capture-policy confidence-only` is the default and captures only when the weak model returns low confidence, `need_strong_think`, or invalid structured output. `--capture-policy always-after-weak` is for demos and debugging: every non-empty message still reaches the real weak model first, then the runtime captures the typed weak result as a continuation for the strong Think handler.

## What To Look For

The trace shows the CPS path:

```text
capture_continuation:
  runtime defunctionalizes the current continuation into data.

strong_think:
  the Think handler processes only the stuck EffectFrame, not the whole task.

resume_continuation:
  runtime resumes execution with the typed ThinkDecision.
```

The runtime does not branch on business keywords such as OTPs, meetings, proposals, or unsubscribe notices. Empty or whitespace-only input is the structural deterministic path; non-empty semantic messages go to the weak model first. Use `--capture-policy always-after-weak` when you want the trace to reliably show continuation capture without adding message-text rules to the runtime.

## Why This Is CPS

- The runtime turns the workflow into a `Step::Done` / `Step::Effect` trampoline.
- `Continuation` is a Serde-serializable enum, not a closure.
- The strong model receives one `EffectFrame`, not arbitrary task history.
- The strong model returns `ThinkDecision` JSON, not a free-form final answer.

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
