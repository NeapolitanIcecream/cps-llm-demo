# cps-llm-demo

Minimal Rust CLI demo for a defunctionalized CPS / typed-effect LLM runtime.

This is not a product agent. It is a runtime semantics demo: deterministic code handles cheap cases, a small classifier fills a typed semantic hole, and the runtime captures a stuck continuation as an `EffectFrame` when guard conditions require deeper handling. The Think handler receives that frame, returns a typed `ThinkDecision`, and the runtime resumes.

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
```

`run` writes final `ActionDraft[]` JSON to stdout. With `--trace-json`, it writes runtime trace JSONL to stderr, so the streams can be split:

```bash
cargo run -- run examples/messages.json --trace-json 1>out.json 2>trace.jsonl
```

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

The deterministic guard forces messages containing `proposal`, `方向`, or `推进` through the Think effect so the demo reliably shows continuation capture. Verification-code messages are handled without any model call.

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
