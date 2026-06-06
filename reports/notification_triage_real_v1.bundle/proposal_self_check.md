# CPS LLM Real-LLM Proposal Self-Check

## Status

The real-LLM notification-triage experiment has been implemented, run, audited, and iterated through `.cps-real-full-v11`. The final report has `experiment_passed: true`.

Main value result: `cps_generalized_patch` installs `semantic_patch_v1` as Program `v0002`, reduces strong calls per held-out event from `1.0` to `0.0`, beats exact memoization by `53.125` fast-path percentage points, preserves quality within the non-inferiority margin, keeps critical misses at `0`, keeps adversarial false fast-path rate at `0.0`, and stays under the `$100` hard cap.

## Audit Gap Closure

- P1 optimizer provenance: closed by implementation, not by narrowing the claim. `.cps-real-full-v11/model_calls/calls.jsonl` contains a real `gpt-5.5` optimize-phase call with `task_name=optimize_semantic_patch_plan`, `success=true`, and marginal spend `$0.105235`.
- P1 raw source preservation: closed. `artifacts/optimizer/optimizer_response.json` is a `return_value` whose `optimizer_source` is `strong_model_generated_semantic_patch_plan`; `artifacts/semantic_patch_plan.json` preserves the same source and `artifacts/semantic_patch_generation.json` records `raw_optimizer_response_is_final_plan: true`.
- P2 hard-negative guard provenance: closed. The experiment YAML no longer contains hand-authored semantic negative-guard regex. Hard-negative examples live in `data/notification/optimizer_hard_negative_evidence.jsonl`; the strong optimizer generated 5 negative guards, including the deadline-completed guard added after v10 exposed an adversarial false fast-path.
- P3 old artifact ambiguity: closed by scope. `reports/authoritative_audit_scope.md` states that final acceptance evidence is limited to `reports/notification_triage_real_v1.bundle/` and `.cps-real-full-v11/`; older `.cps-real-full-v*` directories are retained only as historical spend ledgers.
- Weak observation source leak: closed. Weak handler input sanitizes observation metadata sources `weak_model` and `strong_model` to `model` while preserving user payload fields named `source`.
- Label provenance: closed. `reports/label_provenance.md` documents synthetic fixture label provenance, the default `generated_from_predictions=false` behavior for all 400 labels, and evaluator rejection of `generated_from_predictions=true`.
- Cost context: closed. `reports/notification_triage_real_v1.cost.json` includes v11 marginal spend and `reports/cumulative_experiment_spend.json` totals. Current cumulative local recorded spend across `.cps-real*` state dirs is `$13.32784850`; provider billing remains authoritative.
- Shadow disagreement definition: closed. The report explains that shadow disagreement compares generalized-patch fast-path outputs against shadow strong-direct outputs. v11 has `29/93 = 31.1828%` disagreements and `0` critical shadow disagreements.
- Adversarial fallback risk: documented. v11 has adversarial `false_fast_path_rate: 0.0`; fallback hard-negative false-action rate remains `32.5%` (`13/40`) and is explicitly reported as outside the false-fast-path gate.

## Verification

- `cargo fmt --check`
- `cargo test`
- `cargo clippy --all-targets --all-features -- -D warnings`
- Focused regression: `cargo test run_experiment_executes_variants_and_writes_predictions_and_quality`
- Full run: `CPS_EXPERIMENT_CONCURRENCY=8 cargo run -- run-experiment --config experiments/notification_triage.real.yaml --state-dir .cps-real-full-v11`
- Hardcoding check: `rg -n "unsubscribe|prototype|wording|opinion request|already|replied|product review" src/experiment/runner.rs src/models.rs` returns no matches.

## Final Results

- Split leakage:
  - `events_total`: 400
  - `exact_duplicate_cross_split`: 0
  - `near_duplicate_cross_split_rate`: 0.00075
  - `heldout_unseen_cluster_rate`: 0.28125
  - `adversarial_cases_total`: 40
- Budget:
  - hard cap: `$100.00`
  - v11 marginal spend: `$0.92373350`
  - optimize phase spend: `$0.105235`
  - cumulative local recorded spend: `$13.32784850`
  - calls: 1586
  - cache hits: 1278
  - cache misses: 308
  - unknown phase records: 0
  - event-named `by_run` files: 0
- Optimizer:
  - request has no `candidate_plan`
  - profile clusters supplied: 7
  - hard-negative evidence examples supplied: 10
  - generated rules: 6
  - generated negative guards: 5
  - installed registry source: `optimizer_strong_model`
  - final plan source: `strong_model_generated_semantic_patch_plan`
- Generalized heldout:
  - events: 160
  - schema validity: 100%
  - quality: `0.8375`
  - strong calls/event: `0.0`
  - weak calls/event: `1.46875`
  - fast-path hit rate: `53.125%`
  - false fast-path rate: `0.0`
  - p95 continuation frame bytes: `2339`
- Exact memo heldout:
  - quality: `0.71875`
  - fast-path hit rate: `0.0`
  - strong calls/event: `1.0`
- Strong direct heldout:
  - quality: `0.725`
  - strong calls/event: `1.0`
- Adversarial generalized:
  - events: 40
  - quality: `0.675`
  - fast-path hit rate: `20.0%`
  - false fast-path rate: `0.0`
  - fallback hard-negative false-action rate: `32.5%` (`13/40`)
- Shadow audit:
  - fast-path hits checked: 93
  - semantic disagreements: 29
  - disagreement rate: `31.1828%`
  - critical disagreements: 0

## Final Pass/Fail

```json
{
  "quality_noninferior": true,
  "critical_miss_not_worse": true,
  "strong_calls_reduced_by_50_percent": true,
  "generalized_patch_beats_exact_memo": true,
  "false_fast_path_under_1_percent": true,
  "adversarial_false_fast_path_under_2_percent": true,
  "continuation_frame_p95_under_limit": true,
  "program_version_advanced": true,
  "multi_cluster_improvement": true,
  "patch_gate_accepted": true,
  "within_100_usd_budget": true,
  "experiment_passed": true
}
```

## Artifacts

- Final report: `reports/notification_triage_real_v1.md`
- Final JSON report: `reports/notification_triage_real_v1.json`
- Cost report: `reports/notification_triage_real_v1.cost.json`
- Quality report: `reports/notification_triage_real_v1.quality.json`
- Cumulative spend report: `reports/cumulative_experiment_spend.json`
- Label provenance: `reports/label_provenance.md`
- Audit scope: `reports/authoritative_audit_scope.md`
- Reproducibility bundle: `reports/notification_triage_real_v1.bundle/`
- Final state dir: `.cps-real-full-v11/`

## Scope Note

The optimizer path includes a real strong-model generation phase. Rust provides profile evidence, failure-cluster evidence, validation constraints, and schema; the strong model returns the semantic plan, including rules and negative guards. Rust then performs schema/safety/gate validation and renders the typed ProgramPatch without replacing the model-generated rules or negative guards.
