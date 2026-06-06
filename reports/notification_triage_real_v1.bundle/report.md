# Notification Triage Real-LLM Experiment

Budget cap: $100.00
Report-run marginal API spend: $0.9237
Cache hit rate: 80.6%

The spend above is the marginal spend for this captured report run. A high cache-hit rate means this was a cached rerun, not a cold-start full-experiment cost.

The cumulative local recorded spend across `.cps-real*` experiment state dirs is $13.3278, including the v11 rerun's $0.9237 marginal spend. Provider billing/invoice data remains authoritative.

The authoritative audit artifacts for acceptance are `reports/notification_triage_real_v1.bundle/` and `.cps-real-full-v11/`. Older `.cps-real-full-v*` directories are retained only as historical ledgers for cumulative spend and should not be used as final acceptance evidence.

| Variant | Quality | Critical miss | Strong calls / event | Weak calls / event | Fast path hit | API spend |
|---|---:|---:|---:|---:|---:|---:|
| cps_exact_memo | 0.719 | 0.000% | 1.000 | 0.000 | 0.0% | $0.0000 |
| cps_generalized_patch | 0.838 | 0.000% | 0.000 | 1.469 | 53.1% | $0.4190 |
| cps_generalized_patch.adversarial | 0.675 | 0.000% | 0.000 | 1.800 | 20.0% | $0.1085 |
| cps_generalized_patch_no_semantic_weak | 0.725 | 0.000% | 1.000 | 0.000 | 0.0% | $0.0000 |
| cps_unoptimized | 0.775 | 0.000% | 0.000 | 1.000 | 0.0% | $0.0210 |
| strong_direct | 0.725 | 0.000% | 1.000 | 0.000 | 0.0% | $0.0000 |
| weak_only | 0.806 | 33.333% | 0.000 | 1.000 | 0.0% | $0.0000 |

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

## Generalization Breakdown

```json
[
  {
    "events": 23,
    "exact_memo_hits": 0,
    "false_fast_path_hits": 0,
    "generalized_patch_hits": 2,
    "semantic_cluster": "ambiguous_requires_strong_think"
  },
  {
    "events": 11,
    "exact_memo_hits": 0,
    "false_fast_path_hits": 0,
    "generalized_patch_hits": 11,
    "semantic_cluster": "deadline_request_without_document"
  },
  {
    "events": 11,
    "exact_memo_hits": 0,
    "false_fast_path_hits": 0,
    "generalized_patch_hits": 10,
    "semantic_cluster": "fyi_no_action"
  },
  {
    "events": 45,
    "exact_memo_hits": 0,
    "false_fast_path_hits": 0,
    "generalized_patch_hits": 5,
    "semantic_cluster": "low_value_system_code"
  },
  {
    "events": 12,
    "exact_memo_hits": 0,
    "false_fast_path_hits": 0,
    "generalized_patch_hits": 11,
    "semantic_cluster": "meeting_invitation"
  },
  {
    "events": 24,
    "exact_memo_hits": 0,
    "false_fast_path_hits": 0,
    "generalized_patch_hits": 22,
    "semantic_cluster": "reply_needed_opinion_request"
  },
  {
    "events": 11,
    "exact_memo_hits": 0,
    "false_fast_path_hits": 0,
    "generalized_patch_hits": 11,
    "semantic_cluster": "request_send_document_before_deadline"
  },
  {
    "events": 23,
    "exact_memo_hits": 0,
    "false_fast_path_hits": 0,
    "generalized_patch_hits": 13,
    "semantic_cluster": "shipping_or_booking_update"
  }
]
```

## Patch

```json
{
  "installed": true,
  "installed_program_version": "v0002",
  "negative_guards": 5,
  "optimizer_source": "strong_model_generated_semantic_patch_plan",
  "patch_id": "semantic_patch_v1",
  "positive_clusters": [
    "deadline_request_without_document",
    "fyi_no_action",
    "meeting_invitation",
    "reply_needed_opinion_request",
    "request_send_document_before_deadline",
    "shipping_or_booking_update"
  ],
  "raw_optimizer_response_is_final_plan": true,
  "source": "optimizer_strong_model"
}
```

Patch provenance: `optimizer_source` is `strong_model_generated_semantic_patch_plan`. The raw optimize-phase strong-model response is the final semantic plan source; Rust validates the plan and renders the ProgramPatch without replacing rules or negative guards.

## Continuation Frames

```json
{
  "frames_total": 1,
  "limit_bytes": 16384,
  "max_bytes": 2339,
  "over_limit": 0,
  "p50_bytes": 2339,
  "p95_bytes": 2339
}
```

## Adversarial Residual Risk

Adversarial false fast-path rate is 0.0%, so the fast path satisfies the proposal gate. The adversarial hard-negative false-action rate is 32.5% (13/40 events); those false actions came from fallback behavior, not fast-path hits.

## Shadow Audit

Shadow disagreement is measured on generalized-patch fast-path hits by comparing the fast-path output with a shadow strong-direct output. The current disagreement rate is 31.2%; critical miss rate remains 0.0%, so these disagreements are non-critical output differences and remain a quality-monitoring limitation rather than a gate failure.
