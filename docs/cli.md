# CLI Reference

The binary is `cps-llm-demo`. Run commands through Cargo during development:

```bash
cargo run -- <command> [options]
```

Most commands print JSON to stdout. Commands with `--trace-json` write runtime trace JSONL to stderr.

## Environment

The CLI reads `.env` at startup and also accepts explicit flags.

| Variable | Default | Used for |
|---|---|---|
| `OPENAI_API_KEY` | none | Responses API authentication |
| `OPENAI_BASE_URL` | `https://api.openai.com/v1` | OpenAI-compatible API endpoint |
| `CPS_WEAK_MODEL` | `gpt-5.4-mini` | weak model handler |
| `CPS_STRONG_MODEL` | `gpt-5.5` | strong model handler |

`compile-run`, `run-program`, and `probe-models` require a non-empty API key. The stateful workflow commands `run-stream`, `optimize`, and `baseline-strong-direct` use fixture handlers when the API key is blank:

```bash
cargo run -- run-stream ... --api-key ""
```

Experiment YAML files have their own `models.api_key_env` and `models.base_url_env` settings. The checked-in real experiment configs set `models.use_responses_api: true`, so they require the configured API key environment variable.

## Inspect Commands

```bash
cargo run -- --help
cargo run -- run-experiment --help
```

Clap may show the current value of env-backed options in help output. Do not paste raw help output into issues or docs when your environment contains real credentials.

## Schema and Validation

Print the schema bundle:

```bash
cargo run -- schema > /tmp/cps-schema.json
```

Validate a program fixture:

```bash
cargo run -- validate-program \
  --program examples/message_action.v2.program.json
```

Validate a trace capture/resume stream:

```bash
cargo run -- replay --trace traces/example.trace.jsonl
```

Probe configured weak and strong model handlers:

```bash
cargo run -- probe-models
```

## Program Execution

Run an existing `Program`:

```bash
cargo run -- run-program \
  --program examples/message_action.v2.program.json \
  --input examples/messages.json \
  --trace-json \
  1>out.json \
  2>trace.jsonl
```

Compile a task to `Program` IR with the strong handler, then execute it:

```bash
cargo run -- compile-run \
  --task examples/message_action.task.md \
  --input examples/messages.json \
  --trace-json
```

Both commands use the same runtime path after a program is available.

## Stateful Workflow Commands

Initialize a workflow registry from a checked-in program:

```bash
STATE=/tmp/cps-llm-demo-state

cargo run -- init-workflow \
  --workflow notification_triage \
  --program examples/notification_triage.v1.program.json \
  --state-dir "$STATE"
```

Compile the initial workflow program from a task instead:

```bash
cargo run -- init-workflow \
  --workflow notification_triage \
  --task examples/notification_triage.task.md \
  --state-dir "$STATE"
```

Run events through the latest workflow program:

```bash
cargo run -- run-stream \
  --workflow notification_triage \
  --events examples/notification_triage.round1.jsonl \
  --state-dir "$STATE" \
  --api-key "" \
  --trace-json
```

Install a fixture patch:

```bash
cargo run -- optimize \
  --workflow notification_triage \
  --patch examples/notification_triage.fast_path.patch.json \
  --state-dir "$STATE" \
  --api-key ""
```

Ask the strong handler to propose a patch from profile data:

```bash
cargo run -- optimize \
  --workflow notification_triage \
  --state-dir "$STATE" \
  --max-patches 3
```

Run the strong-direct baseline:

```bash
cargo run -- baseline-strong-direct \
  --workflow notification_triage \
  --task examples/notification_triage.task.md \
  --events examples/notification_triage.round1.jsonl \
  --state-dir "$STATE" \
  --api-key ""
```

Print workflow metrics:

```bash
cargo run -- metrics-report \
  --workflow notification_triage \
  --state-dir "$STATE"
```

Compare baseline, before-patch, and after-patch runs:

```bash
cargo run -- compare-runs \
  --baseline-run "$BASELINE_RUN" \
  --before-run "$ROUND1_RUN" \
  --after-run "$ROUND2_RUN" \
  --state-dir "$STATE"
```

`compare-runs` reports readiness checks such as strong-call reduction, fast-path coverage increase, bounded continuation frames, and program-version advancement.

## Budget Commands

Print the state directory budget ledger:

```bash
cargo run -- budget-report --state-dir "$STATE"
```

Reset the ledger:

```bash
cargo run -- budget-reset --state-dir "$STATE" --confirm
```

Use `budget-reset` only for disposable local state directories.

## Data and Quality Utilities

Split event and gold-label files:

```bash
cargo run -- split-events \
  --events data/notification/all_events.jsonl \
  --gold data/notification/gold_labels.jsonl \
  --out data/notification/splits \
  --strategy time-cluster \
  --profile-train 120 \
  --patch-validation 80 \
  --heldout-test 160 \
  --adversarial-test 40
```

Evaluate prediction quality:

```bash
cargo run -- evaluate-quality \
  --predictions .cps-real-full-v11/experiments/notification_triage_real_v1/predictions/cps_generalized_patch.heldout.jsonl \
  --gold data/notification/splits/heldout_test.gold.jsonl \
  --out /tmp/cps-quality.json
```

Build an exact-memo patch for an initialized workflow:

```bash
cargo run -- build-exact-memo-patch \
  --workflow notification_triage \
  --from-events data/notification/splits/profile_train.events.jsonl \
  --state-dir "$STATE"
```

## Experiment Commands

Estimate cost without calling models:

```bash
cargo run -- run-experiment \
  --config experiments/notification_triage.pilot.yaml \
  --dry-run-cost
```

Run an experiment:

```bash
cargo run -- run-experiment \
  --config experiments/notification_triage.pilot.yaml
```

Override the configured state directory:

```bash
cargo run -- run-experiment \
  --config experiments/notification_triage.pilot.yaml \
  --state-dir .cps-real-exp
```

Copy the generated report into `reports/`:

```bash
cargo run -- experiment-report \
  --experiment notification_triage_pilot_v1 \
  --state-dir .cps-real-exp \
  --out reports/notification_triage_pilot_v1.md
```

`run-experiment` writes `run_manifest.json`, `config.lock.yaml`, `price_catalog.lock.yaml`, predictions, quality reports, phase markers, model-call logs, budget records, and final reports under the selected state directory.
