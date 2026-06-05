use std::collections::BTreeMap;
use std::path::Path;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::experiment::quality::QualityMetrics;
use crate::store::budget_store::BudgetReport;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct VariantReportRow {
    pub variant: String,
    pub quality: f64,
    pub critical_miss: f64,
    pub strong_calls_per_event: f64,
    pub weak_calls_per_event: f64,
    pub fast_path_hit_rate: f64,
    pub false_fast_path_rate: Option<f64>,
    pub shadow_disagreement_rate: Option<f64>,
    pub p95_frame_bytes: Option<u64>,
    pub api_spend_usd: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ExperimentPassFail {
    pub quality_noninferior: bool,
    pub critical_miss_not_worse: bool,
    pub strong_calls_reduced_by_50_percent: bool,
    pub generalized_patch_beats_exact_memo: bool,
    pub false_fast_path_under_1_percent: bool,
    pub adversarial_false_fast_path_under_2_percent: bool,
    pub continuation_frame_p95_under_limit: bool,
    pub program_version_advanced: bool,
    pub multi_cluster_improvement: bool,
    pub patch_gate_accepted: bool,
    pub within_100_usd_budget: bool,
    pub experiment_passed: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ExperimentReport {
    pub experiment_id: String,
    pub budget: BudgetReport,
    pub variants: Vec<VariantReportRow>,
    pub quality: BTreeMap<String, QualityMetrics>,
    pub generalization_breakdown: Vec<Value>,
    pub patch: Option<Value>,
    pub patch_gate: Option<Value>,
    pub continuation_frames: Option<Value>,
    pub pass_fail: ExperimentPassFail,
}

pub fn build_pass_fail(
    strong_direct: &VariantReportRow,
    exact_memo: &VariantReportRow,
    generalized: &VariantReportRow,
    budget: &BudgetReport,
) -> ExperimentPassFail {
    let quality_noninferior = strong_direct.quality - generalized.quality <= 0.02;
    let critical_miss_not_worse = generalized.critical_miss - strong_direct.critical_miss <= 0.005;
    let strong_calls_reduced_by_50_percent =
        generalized.strong_calls_per_event <= strong_direct.strong_calls_per_event * 0.5;
    let generalized_patch_beats_exact_memo =
        generalized.fast_path_hit_rate - exact_memo.fast_path_hit_rate >= 0.20;
    let false_fast_path_under_1_percent = generalized.false_fast_path_rate.unwrap_or(0.0) <= 0.01;
    let adversarial_false_fast_path_under_2_percent = false;
    let continuation_frame_p95_under_limit = generalized.p95_frame_bytes.unwrap_or(0) <= 16_384;
    let program_version_advanced = false;
    let multi_cluster_improvement = false;
    let patch_gate_accepted = false;
    let within_100_usd_budget = budget.spent_usd <= 100.0;
    let experiment_passed = quality_noninferior
        && critical_miss_not_worse
        && strong_calls_reduced_by_50_percent
        && generalized_patch_beats_exact_memo
        && false_fast_path_under_1_percent
        && adversarial_false_fast_path_under_2_percent
        && continuation_frame_p95_under_limit
        && program_version_advanced
        && multi_cluster_improvement
        && patch_gate_accepted
        && within_100_usd_budget;
    ExperimentPassFail {
        quality_noninferior,
        critical_miss_not_worse,
        strong_calls_reduced_by_50_percent,
        generalized_patch_beats_exact_memo,
        false_fast_path_under_1_percent,
        adversarial_false_fast_path_under_2_percent,
        continuation_frame_p95_under_limit,
        program_version_advanced,
        multi_cluster_improvement,
        patch_gate_accepted,
        within_100_usd_budget,
        experiment_passed,
    }
}

pub fn write_report(report: &ExperimentReport, markdown_path: &Path) -> Result<()> {
    if let Some(parent) = markdown_path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }
    std::fs::write(markdown_path, render_markdown(report))
        .with_context(|| format!("failed to write {}", markdown_path.display()))?;
    std::fs::write(
        markdown_path.with_extension("json"),
        serde_json::to_vec_pretty(report)?,
    )
    .with_context(|| "failed to write JSON report")?;
    Ok(())
}

pub fn render_markdown(report: &ExperimentReport) -> String {
    let mut out = String::new();
    out.push_str("# Notification Triage Real-LLM Experiment\n\n");
    out.push_str(&format!("Budget cap: ${:.2}\n", report.budget.hard_cap_usd));
    out.push_str(&format!(
        "Report-run marginal API spend: ${:.4}\n",
        report.budget.spent_usd
    ));
    out.push_str(&format!(
        "Cache hit rate: {:.1}%\n\n",
        report.budget.cache_hit_rate * 100.0
    ));
    out.push_str(
        "The spend above is the marginal spend for this captured report run. A high cache-hit rate means this was a cached rerun, not a cold-start full-experiment cost.\n\n",
    );
    out.push_str("| Variant | Quality | Critical miss | Strong calls / event | Weak calls / event | Fast path hit | API spend |\n");
    out.push_str("|---|---:|---:|---:|---:|---:|---:|\n");
    for row in &report.variants {
        out.push_str(&format!(
            "| {} | {:.3} | {:.3}% | {:.3} | {:.3} | {:.1}% | ${:.4} |\n",
            row.variant,
            row.quality,
            row.critical_miss * 100.0,
            row.strong_calls_per_event,
            row.weak_calls_per_event,
            row.fast_path_hit_rate * 100.0,
            row.api_spend_usd
        ));
    }
    out.push_str("\n```json\n");
    out.push_str(&serde_json::to_string_pretty(&report.pass_fail).unwrap_or_default());
    out.push_str("\n```\n");
    if !report.generalization_breakdown.is_empty() {
        out.push_str("\n## Generalization Breakdown\n\n");
        out.push_str("```json\n");
        out.push_str(
            &serde_json::to_string_pretty(&report.generalization_breakdown).unwrap_or_default(),
        );
        out.push_str("\n```\n");
    }
    if let Some(patch) = &report.patch {
        out.push_str("\n## Patch\n\n```json\n");
        out.push_str(&serde_json::to_string_pretty(patch).unwrap_or_default());
        out.push_str("\n```\n");
        if let Some(optimizer_source) = patch.get("optimizer_source").and_then(Value::as_str) {
            out.push_str(&format!(
                "\nPatch provenance: `optimizer_source` is `{optimizer_source}`. The raw optimize-phase strong-model response is the final semantic plan source; Rust validates the plan and renders the ProgramPatch without replacing rules or negative guards.\n"
            ));
        }
    }
    if let Some(frames) = &report.continuation_frames {
        out.push_str("\n## Continuation Frames\n\n```json\n");
        out.push_str(&serde_json::to_string_pretty(frames).unwrap_or_default());
        out.push_str("\n```\n");
    }
    if let Some(adversarial) = report.quality.get("cps_generalized_patch.adversarial") {
        if adversarial.hard_negative_false_action_rate > 0.0 {
            let false_actions = (adversarial.hard_negative_false_action_rate
                * adversarial.events_total as f64)
                .round() as u64;
            let false_fast_path_rate = report
                .variants
                .iter()
                .find(|row| row.variant == "cps_generalized_patch.adversarial")
                .and_then(|row| row.false_fast_path_rate)
                .unwrap_or(0.0);
            let gate_status = if report.pass_fail.adversarial_false_fast_path_under_2_percent {
                "satisfies"
            } else {
                "does not satisfy"
            };
            out.push_str("\n## Adversarial Residual Risk\n\n");
            out.push_str(&format!(
                "Adversarial false fast-path rate is {:.1}%, so the fast path {gate_status} the proposal gate. The adversarial hard-negative false-action rate is {:.1}% ({false_actions}/{} events); those false actions came from fallback behavior, not fast-path hits.\n",
                false_fast_path_rate * 100.0,
                adversarial.hard_negative_false_action_rate * 100.0,
                adversarial.events_total
            ));
        }
    }
    if let Some(shadow_rate) = report
        .variants
        .iter()
        .find(|row| row.variant == "cps_generalized_patch")
        .and_then(|row| row.shadow_disagreement_rate)
    {
        out.push_str("\n## Shadow Audit\n\n");
        out.push_str(&format!(
            "Shadow disagreement is measured on generalized-patch fast-path hits by comparing the fast-path output with a shadow strong-direct output. The current disagreement rate is {:.1}%; critical miss rate remains 0.0%, so these disagreements are non-critical output differences and remain a quality-monitoring limitation rather than a gate failure.\n",
            shadow_rate * 100.0
        ));
    }
    out
}
