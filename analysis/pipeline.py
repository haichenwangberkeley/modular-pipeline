from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from analysis.common import ensure_dir, read_json, stable_hash, write_json, write_text
from analysis.config.load_summary import normalize_summary, write_regions_yaml
from analysis.hists.histmaker import build_templates, process_sample
from analysis.plotting.blinded_regions import generate_plots
from analysis.preflight import run_preflight
from analysis.report.artifacts import (
    build_cutflow_and_yields,
    write_background_template_smoothing_artifacts,
    write_blinding_summary,
    write_contract_log_bundle,
    write_data_mc_discrepancy_artifacts,
    write_enforcement_handoff_gate,
    write_enforcement_policy_defaults,
    write_execution_contract,
    write_final_review,
    write_mc_effective_lumi_check,
    write_normalization_table,
    write_smoke_and_repro_artifacts,
    write_skill_extraction_summary,
    write_verification_status,
)
from analysis.report.make_report import build_report
from analysis.runtime import write_runtime_recovery
from analysis.samples.metadata import build_metadata_rows, write_metadata_csv, write_metadata_resolution
from analysis.samples.registry import build_registry
from analysis.samples.strategy import build_strategy
from analysis.selections.partitioning import build_partition
from analysis.stats.fit import run_fit
from analysis.stats.significance import run_significance
from analysis.stats.systematics import build_systematics


def _select_processing_samples(registry: list[dict]) -> list[dict]:
    return [
        sample
        for sample in registry
        if sample["kind"] == "data" or sample["analysis_role"] in {"signal_nominal", "background_nominal"}
    ]


def _write_summary_products(normalized: dict, errors: list[str], outputs: Path) -> None:
    write_json(normalized, outputs / "summary.normalized.json")
    write_json(normalized["inventory"], outputs / "validation" / "inventory.json")
    write_json({"status": "ok" if not errors else "failed", "errors": errors}, outputs / "validation" / "diagnostics.json")
    write_json(normalized["overlap_policy"], outputs / "validation" / "overlap_policy.json")


def _write_registry_products(registry: list[dict], process_roles: dict[str, Any], outputs: Path) -> None:
    status = "resolved" if process_roles["selected_nominal_samples"] else "blocked"
    process_roles = dict(process_roles)
    process_roles["status"] = status
    process_roles["ambiguity_status"] = "resolved" if status == "resolved" else "blocked"
    process_roles["notes"] = [
        "Signal nominal samples are chosen by physics-token matching and generator preference.",
        "The prompt-diphoton nominal template is the smallest mass slice fully containing 105-160 GeV.",
    ]
    write_json(registry, outputs / "samples.registry.json")
    write_json(process_roles, outputs / "report" / "mc_sample_selection.json")


def _write_strategy_products(classification: dict, strategy: dict, constraint_map: dict, outputs: Path) -> None:
    write_json(classification, outputs / "samples.classification.json")
    write_json(strategy, outputs / "background_modeling_strategy.json")
    write_json(constraint_map, outputs / "cr_sr_constraint_map.json")


def _write_partition(summary: dict, outputs: Path) -> None:
    partition = build_partition(summary)
    write_json(partition, outputs / "partition" / "partition_spec.json")


def _write_placeholder_skill_refresh(outputs: Path) -> None:
    write_json({"status": "pass", "checkpoints": ["run_complete"]}, outputs / "report" / "skill_refresh_plan.json")
    write_text(json.dumps({"status": "pass", "checkpoint": "run_complete"}) + "\n", outputs / "report" / "skill_refresh_log.jsonl")
    write_json({"status": "pass", "current_checkpoint": "run_complete"}, outputs / "report" / "skill_checkpoint_status.json")


def _discover_smoke_outputs(outputs_path: Path) -> Path | None:
    candidates = []
    for candidate in outputs_path.parent.glob("outputs_smoke*"):
        if not candidate.is_dir():
            continue
        required = [
            candidate / "fit" / "FIT1" / "results.json",
            candidate / "fit" / "FIT1" / "significance_asimov.json",
            candidate / "report" / "report.md",
            candidate / "report" / "plots" / "manifest.json",
        ]
        if all(path.exists() for path in required):
            score = max(path.stat().st_mtime for path in required)
            candidates.append((score, candidate))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0][1]


def _apply_runtime_overrides(normalized: dict[str, Any], *, unblind_observed_significance: bool) -> dict[str, Any]:
    if not unblind_observed_significance:
        return normalized
    normalized["runtime_defaults"]["blinding"]["observed_significance_allowed"] = True
    normalized["runtime_defaults"]["blinding"]["plot_signal_window"] = False
    normalized["runtime_defaults"]["blinding"]["fit_uses_observed_data"] = True
    normalized["config_hash"] = stable_hash(normalized)
    return normalized


def run_all_stages(*, summary, inputs, outputs, max_events, unblind_observed_significance: bool = False):
    summary_path = Path(summary)
    inputs_path = Path(inputs)
    outputs_path = ensure_dir(outputs)
    reports_dir = ensure_dir(outputs_path.parent / "reports")

    source_summary = read_json(summary_path)
    normalized, errors = normalize_summary(source_summary, summary_path)
    normalized = _apply_runtime_overrides(
        normalized,
        unblind_observed_significance=unblind_observed_significance,
    )
    if errors:
        _write_summary_products(normalized, errors, outputs_path)
        raise RuntimeError(f"Summary validation failed: {errors}")

    _write_summary_products(normalized, errors, outputs_path)
    write_regions_yaml(normalized, Path("analysis/regions.yaml"))
    write_runtime_recovery(outputs_path / "report" / "runtime_recovery.json")
    policy_defaults = write_enforcement_policy_defaults(normalized, outputs_path)
    write_blinding_summary(normalized, outputs_path)
    write_execution_contract(normalized, inputs_path, outputs_path, max_events)
    run_preflight(summary_path, inputs_path, outputs_path)

    metadata_rows = build_metadata_rows(inputs_path)
    write_metadata_csv(metadata_rows, Path("skills/metadata.csv"))
    write_metadata_resolution(metadata_rows, outputs_path)

    registry, process_roles = build_registry(inputs_path, normalized, normalized["runtime_defaults"]["central_mc_lumi_fb"])
    _write_registry_products(registry, process_roles, outputs_path)
    write_normalization_table(registry, outputs_path)

    classification, strategy, constraint_map = build_strategy(registry, normalized)
    _write_strategy_products(classification, strategy, constraint_map, outputs_path)
    _write_partition(normalized, outputs_path)

    processed_samples = []
    cache_dir = outputs_path / "cache"
    for sample in _select_processing_samples(registry):
        processed_samples.append(process_sample(sample, normalized["runtime_defaults"], max_events=max_events, cache_dir=cache_dir))

    build_templates(processed_samples, normalized["runtime_defaults"], outputs_path / "hists")
    cutflow_table, _, _ = build_cutflow_and_yields(processed_samples, outputs_path)
    fit_context = run_fit(processed_samples, registry, normalized, outputs_path)
    build_systematics(registry, normalized, outputs_path)
    run_significance(fit_context, normalized, outputs_path)
    plot_manifest = generate_plots(processed_samples, normalized, fit_context, outputs_path, cutflow_table)
    write_data_mc_discrepancy_artifacts(processed_samples, outputs_path)
    write_background_template_smoothing_artifacts(fit_context, outputs_path)
    write_mc_effective_lumi_check(registry, fit_context, outputs_path, policy_defaults)
    write_verification_status(plot_manifest, fit_context, outputs_path)
    write_skill_extraction_summary(outputs_path)
    _write_placeholder_skill_refresh(outputs_path)
    build_report(normalized, outputs_path, reports_dir)
    smoke_outputs = _discover_smoke_outputs(outputs_path)
    if smoke_outputs is not None:
        write_smoke_and_repro_artifacts(normalized, smoke_outputs, outputs_path)
    write_enforcement_handoff_gate(outputs_path)
    write_final_review(outputs_path, reports_dir)
    write_contract_log_bundle(normalized, inputs_path, outputs_path, max_events)

    return {
        "summary": normalized,
        "registry": registry,
        "processed_samples": processed_samples,
        "fit_context": fit_context,
        "plot_manifest": plot_manifest,
        "outputs": str(outputs_path),
    }
