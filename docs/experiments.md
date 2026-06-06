# Experiment Workflow

The experiment layer runs the notification-triage workflow across fixed data splits, records model-call attribution, enforces a budget ledger, evaluates predictions, and writes reports.

## Inputs

Key inputs are checked in:

- `experiments/notification_triage.pilot.yaml`: small real-model pilot config.
- `experiments/notification_triage.real.yaml`: full real-model experiment config.
- `experiments/notification_triage.real_readonly_replay.yaml`: replay config using read-only cache.
- `experiments/price_catalog.openai.yaml`: model price catalog used for budget estimates and spend attribution.
- `experiments/schemas/`: event, output, and gold-label schemas.
- `data/notification/all_events.jsonl`: source events.
- `data/notification/gold_labels.jsonl`: labels used for quality evaluation.
- `data/notification/splits/`: checked-in split files and leakage report.
- `data/notification/optimizer_hard_negative_evidence.jsonl`: hard negatives supplied to the optimizer.

## Phases

The real experiment config runs these phases:

1. `split_events`
2. `weak_only`
3. `strong_direct`
4. `cps_unoptimized_profile`
5. `optimize`
6. `cps_unoptimized_heldout`
7. `patch_validation`
8. `cps_exact_memo_heldout`
9. `cps_generalized_patch_heldout`
10. `cps_generalized_patch_no_semantic_weak`
11. `adversarial_audit`
12. `shadow_audit`
13. `quality_eval`
14. `report`

`run-experiment` writes a phase marker after each completed or skipped phase. If a manifest already exists and the locked config matches, a later run resumes from the next unfinished phase.

## Cost Estimate

Run a dry estimate first:

```bash
cargo run -- run-experiment \
  --config experiments/notification_triage.pilot.yaml \
  --dry-run-cost
```

The estimate uses `experiments/price_catalog.openai.yaml`, the configured split sizes, and the budget projection multiplier. A non-dry run refuses to start when the projected cost exceeds the hard cap.

## Pilot Run

Set model access in `.env`, then run:

```bash
cargo run -- run-experiment \
  --config experiments/notification_triage.pilot.yaml
```

The pilot config uses smaller split counts:

- profile train: 12
- patch validation: 8
- heldout test: 16
- adversarial test: 4

The default pilot state directory is `.cps-real-exp`.

## Full Run

The full config uses:

- profile train: 120
- patch validation: 80
- heldout test: 160
- adversarial test: 40

Run it only after checking the dry-run estimate:

```bash
cargo run -- run-experiment \
  --config experiments/notification_triage.real.yaml \
  --dry-run-cost

cargo run -- run-experiment \
  --config experiments/notification_triage.real.yaml
```

Use `--state-dir <DIR>` to keep a run separate from the default state directory.

## Reports

Copy an experiment report out of the state directory:

```bash
cargo run -- experiment-report \
  --experiment notification_triage_pilot_v1 \
  --state-dir .cps-real-exp \
  --out reports/notification_triage_pilot_v1.md
```

The final captured full-run report is:

- `reports/notification_triage_real_v1.md`
- `reports/notification_triage_real_v1.json`
- `reports/notification_triage_real_v1.cost.json`
- `reports/notification_triage_real_v1.quality.json`
- `reports/notification_triage_real_v1.bundle/`

The Chinese narrative report is `reports/notification_triage_experiment_report_zh.md`.

## Output Layout

Inside a state directory, the experiment writes:

- `experiments/<experiment_id>/run_manifest.json`
- `experiments/<experiment_id>/config.lock.yaml`
- `experiments/<experiment_id>/price_catalog.lock.yaml`
- `experiments/<experiment_id>/predictions/`
- `experiments/<experiment_id>/quality/`
- `experiments/<experiment_id>/artifacts/`
- `experiments/<experiment_id>/report.md`
- `experiments/<experiment_id>/report.json`
- `workflows/<workflow_id>/programs/`
- `workflows/<workflow_id>/patches/`
- `workflows/<workflow_id>/traces/`
- `model_calls/calls.jsonl`
- `budgets/spend.jsonl`

The tracked final state directory is `.cps-real-full-v11/`. Older local `.cps-real-*` directories are ignored unless explicitly unignored in `.gitignore`.

## Acceptance Evidence

The final accepted notification-triage run is v11. Use these paths as the authoritative evidence:

- `reports/notification_triage_real_v1.bundle/`
- `.cps-real-full-v11/`
- `reports/authoritative_audit_scope.md`
- `reports/proposal_self_check.md`

The v11 report records:

- `experiment_passed: true`
- `patch_gate_accepted: true`
- `program_version_advanced: true`
- `GeneralizedPatch` heldout quality of `0.8375`
- `GeneralizedPatch` strong calls per event of `0.0`
- fast-path hit rate of `53.125%`
- false fast-path rate of `0.0%`

Provider billing remains the authoritative financial source. The repository budget ledger records local marginal spend for reproducibility and audit attribution.
