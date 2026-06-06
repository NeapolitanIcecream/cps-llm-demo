# cps-llm-demo

`cps-llm-demo` is a Rust CLI for experimenting with LLM workflows represented as typed-effect `Program` IR. It validates and interprets JSON programs, schedules weak and strong model handlers, captures continuations as data, installs program patches, and evaluates a notification-triage workflow.

This repository is a runtime and experiment demo, not a product agent. The runtime owns control flow, validation, state, and traces. Domain behavior lives in task files, program fixtures, patches, prompts, and data.

## What You Can Do

- Validate and run JSON `Program` fixtures.
- Ask a strong model to compile a task spec into `Program` IR, then run it through the same Rust runtime.
- Run a stateful workflow over JSONL event streams.
- Install and evaluate typed `ProgramPatch` changes.
- Reproduce the notification-triage experiment with budgets, model-call logs, prediction files, quality reports, and trace artifacts.

## Repository Map

- `src/`: runtime, CLI, schema validation, model handlers, state stores, optimizer, experiments, and reporting.
- `examples/`: small task, input, program, and patch fixtures.
- `data/notification/`: notification-triage event streams and gold labels.
- `experiments/`: experiment YAML configs, schemas, and price catalog.
- `reports/`: generated experiment reports and the final audit bundle.
- `traces/`: trace replay fixture.
- `tests/`: CLI, runtime, schema, workflow, and experiment contract tests.
- `docs/`: detailed user-facing documentation.

## Requirements

- Rust `1.85` or newer.
- An OpenAI-compatible Responses API key for commands that call real models.

Install dependencies and run the local checks:

```bash
cargo test
cargo fmt --check
cargo clippy --all-targets --all-features -- -D warnings
```

## Configure Model Access

Copy the example environment file when you want to call real models:

```bash
cp .env.example .env
```

Edit `.env` and set:

```bash
OPENAI_API_KEY=replace_me
OPENAI_BASE_URL=https://api.openai.com/v1
CPS_WEAK_MODEL=gpt-5.4-mini
CPS_STRONG_MODEL=gpt-5.5
```

All four values can also be passed with CLI flags. Some stateful workflow commands can use fixture handlers when no API key is supplied; pass `--api-key ""` to force fixture mode even when `.env` exists.

## Quick Start

Print the schema bundle and validate a program fixture:

```bash
cargo run -- schema > /tmp/cps-schema.json
cargo run -- validate-program --program examples/message_action.v2.program.json
```

Run an existing program fixture with real model handlers:

```bash
cargo run -- run-program \
  --program examples/message_action.v2.program.json \
  --input examples/messages.json \
  --trace-json \
  1>out.json \
  2>trace.jsonl
```

Replay the emitted trace:

```bash
cargo run -- replay --trace trace.jsonl
```

`run-program` skips compilation. `compile-run` first asks the strong handler to return `Program` IR, validates that program, then executes it through the same runtime:

```bash
cargo run -- compile-run \
  --task examples/message_action.task.md \
  --input examples/messages.json \
  --trace-json
```

## Run the Local Workflow Fixture

The system-value-loop commands run a workflow over JSONL event streams, store program versions and metrics in a local state directory, install a typed patch, and compare the CPS path with a strong-direct baseline.

Use `--api-key ""` for fixture handlers and no external model calls:

```bash
STATE=/tmp/cps-llm-demo-state

cargo run -- init-workflow \
  --workflow notification_triage \
  --program examples/notification_triage.v1.program.json \
  --state-dir "$STATE"

cargo run -- run-stream \
  --workflow notification_triage \
  --events examples/notification_triage.round1.jsonl \
  --state-dir "$STATE" \
  --api-key "" \
  --trace-json

cargo run -- optimize \
  --workflow notification_triage \
  --patch examples/notification_triage.fast_path.patch.json \
  --state-dir "$STATE" \
  --api-key ""

cargo run -- run-stream \
  --workflow notification_triage \
  --events examples/notification_triage.round2.jsonl \
  --state-dir "$STATE" \
  --api-key "" \
  --trace-json

cargo run -- baseline-strong-direct \
  --workflow notification_triage \
  --task examples/notification_triage.task.md \
  --events examples/notification_triage.round1.jsonl \
  --state-dir "$STATE" \
  --api-key ""

cargo run -- metrics-report \
  --workflow notification_triage \
  --state-dir "$STATE"
```

Use the `run_id` values returned by `run-stream` and `baseline-strong-direct` with `compare-runs`:

```bash
cargo run -- compare-runs \
  --baseline-run "$BASELINE_RUN" \
  --before-run "$ROUND1_RUN" \
  --after-run "$ROUND2_RUN" \
  --state-dir "$STATE"
```

## Run Experiments

Estimate cost before a real-model run:

```bash
cargo run -- run-experiment \
  --config experiments/notification_triage.pilot.yaml \
  --dry-run-cost
```

Run the pilot experiment:

```bash
cargo run -- run-experiment \
  --config experiments/notification_triage.pilot.yaml
```

Generate a markdown report from an experiment state directory:

```bash
cargo run -- experiment-report \
  --experiment notification_triage_pilot_v1 \
  --out reports/notification_triage_pilot_v1.md
```

The final captured notification-triage result is documented in `reports/notification_triage_real_v1.md`. Its acceptance artifacts live in `reports/notification_triage_real_v1.bundle/` and `.cps-real-full-v11/`.

## Documentation

- [CLI Reference](docs/cli.md)
- [Runtime Contract](docs/runtime.md)
- [Experiment Workflow](docs/experiments.md)

## Development Notes

- Most commands write machine-readable JSON to stdout.
- `--trace-json` writes runtime trace JSONL to stderr so stdout can stay reserved for final command output.
- `.env` and local experiment state directories are ignored by git. The final `.cps-real-full-v11/` state directory is tracked because it is part of the reproducibility evidence.
- CLI help can display current environment variable values for env-backed options. Avoid pasting raw help output when your shell contains secrets.
