from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from analysis.common import ensure_dir, read_json, stable_hash, write_json, write_text
from analysis.config.load_summary import normalize_summary, write_regions_yaml
from analysis.hists.histmaker import build_templates, process_sample
from analysis.pipeline import (
    _apply_runtime_overrides,
    _discover_smoke_outputs,
    _select_processing_samples,
    _write_partition,
    _write_placeholder_skill_refresh,
    _write_registry_products,
    _write_strategy_products,
    _write_summary_products,
)
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
from analysis.stats.fit import run_fit
from analysis.stats.significance import run_significance
from analysis.stats.systematics import build_systematics
from modular_pipeline.tracking import write_state


Context = dict[str, Any]
ComponentFunc = Callable[[Context], None]


@dataclass(frozen=True)
class Component:
    name: str
    description: str
    requires: frozenset[str]
    provides: frozenset[str]
    run: ComponentFunc
    groups: frozenset[str] = field(default_factory=frozenset)


def parse_mask(raw: str | None) -> set[str]:
    if not raw:
        return set()
    return {item.strip() for item in raw.split(",") if item.strip()}


def _init_context(
    *,
    summary: Path,
    inputs: Path,
    outputs: Path,
    max_events: int | None,
    unblind_observed_significance: bool,
) -> Context:
    outputs_path = ensure_dir(outputs)
    return {
        "summary_path": Path(summary),
        "inputs_path": Path(inputs),
        "outputs_path": outputs_path,
        "reports_dir": ensure_dir(outputs_path.parent / "reports"),
        "max_events": max_events,
        "unblind_observed_significance": unblind_observed_significance,
    }


def _component_names_and_groups(components: list[Component]) -> set[str]:
    names = {component.name for component in components}
    for component in components:
        names.update(component.groups)
    return names


def _is_masked(component: Component, mask: set[str]) -> bool:
    return component.name in mask or bool(component.groups & mask)


def _record_manifest(ctx: Context, records: list[dict[str, Any]], mask: set[str]) -> None:
    payload = {
        "status": "ok",
        "pipeline": "modular_pipeline",
        "mask": sorted(mask),
        "components": records,
        "outputs": str(ctx["outputs_path"]),
    }
    write_json(payload, ctx["outputs_path"] / "modular_pipeline_manifest.json")
    write_state(ctx["outputs_path"], COMPONENTS, records=records, mask=mask)


def _summary(ctx: Context) -> None:
    source_summary = read_json(ctx["summary_path"])
    normalized, errors = normalize_summary(source_summary, ctx["summary_path"])
    normalized = _apply_runtime_overrides(
        normalized,
        unblind_observed_significance=bool(ctx["unblind_observed_significance"]),
    )
    ctx["summary"] = normalized
    ctx["summary_errors"] = errors
    _write_summary_products(normalized, errors, ctx["outputs_path"])
    if errors:
        raise RuntimeError(f"Summary validation failed: {errors}")


def _runtime_contract(ctx: Context) -> None:
    summary = ctx["summary"]
    outputs_path = ctx["outputs_path"]
    write_regions_yaml(summary, Path("analysis/regions.yaml"))
    write_runtime_recovery(outputs_path / "report" / "runtime_recovery.json")
    ctx["policy_defaults"] = write_enforcement_policy_defaults(summary, outputs_path)
    write_blinding_summary(summary, outputs_path)
    write_execution_contract(summary, ctx["inputs_path"], outputs_path, ctx["max_events"])


def _preflight(ctx: Context) -> None:
    run_preflight(ctx["summary_path"], ctx["inputs_path"], ctx["outputs_path"])


def _metadata(ctx: Context) -> None:
    metadata_rows = build_metadata_rows(ctx["inputs_path"])
    ctx["metadata_rows"] = metadata_rows
    write_metadata_csv(metadata_rows, Path("skills/metadata.csv"))
    write_metadata_resolution(metadata_rows, ctx["outputs_path"])


def _registry(ctx: Context) -> None:
    registry, process_roles = build_registry(
        ctx["inputs_path"],
        ctx["summary"],
        ctx["summary"]["runtime_defaults"]["central_mc_lumi_fb"],
    )
    ctx["registry"] = registry
    ctx["process_roles"] = process_roles
    _write_registry_products(registry, process_roles, ctx["outputs_path"])
    write_normalization_table(registry, ctx["outputs_path"])


def _strategy(ctx: Context) -> None:
    classification, strategy, constraint_map = build_strategy(ctx["registry"], ctx["summary"])
    ctx["classification"] = classification
    ctx["strategy"] = strategy
    ctx["constraint_map"] = constraint_map
    _write_strategy_products(classification, strategy, constraint_map, ctx["outputs_path"])


def _partition(ctx: Context) -> None:
    _write_partition(ctx["summary"], ctx["outputs_path"])


def _samples(ctx: Context) -> None:
    processed_samples = []
    cache_dir = ctx["outputs_path"] / "cache"
    for sample in _select_processing_samples(ctx["registry"]):
        processed_samples.append(
            process_sample(
                sample,
                ctx["summary"]["runtime_defaults"],
                max_events=ctx["max_events"],
                cache_dir=cache_dir,
            )
        )
    ctx["processed_samples"] = processed_samples


def _templates(ctx: Context) -> None:
    build_templates(
        ctx["processed_samples"],
        ctx["summary"]["runtime_defaults"],
        ctx["outputs_path"] / "hists",
    )


def _cutflow(ctx: Context) -> None:
    cutflow_table, _, _ = build_cutflow_and_yields(ctx["processed_samples"], ctx["outputs_path"])
    ctx["cutflow_table"] = cutflow_table


def _fit(ctx: Context) -> None:
    ctx["fit_context"] = run_fit(
        ctx["processed_samples"],
        ctx["registry"],
        ctx["summary"],
        ctx["outputs_path"],
    )


def _systematics(ctx: Context) -> None:
    build_systematics(ctx["registry"], ctx["summary"], ctx["outputs_path"])


def _significance(ctx: Context) -> None:
    run_significance(ctx["fit_context"], ctx["summary"], ctx["outputs_path"])


def _plots(ctx: Context) -> None:
    ctx["plot_manifest"] = generate_plots(
        ctx["processed_samples"],
        ctx["summary"],
        ctx["fit_context"],
        ctx["outputs_path"],
        ctx["cutflow_table"],
    )


def _review_artifacts(ctx: Context) -> None:
    write_data_mc_discrepancy_artifacts(ctx["processed_samples"], ctx["outputs_path"])
    write_background_template_smoothing_artifacts(ctx["fit_context"], ctx["outputs_path"])
    write_mc_effective_lumi_check(
        ctx["registry"],
        ctx["fit_context"],
        ctx["outputs_path"],
        ctx["policy_defaults"],
    )
    write_verification_status(ctx["plot_manifest"], ctx["fit_context"], ctx["outputs_path"])
    write_skill_extraction_summary(ctx["outputs_path"])
    _write_placeholder_skill_refresh(ctx["outputs_path"])


def _report(ctx: Context) -> None:
    build_report(ctx["summary"], ctx["outputs_path"], ctx["reports_dir"])
    smoke_outputs = _discover_smoke_outputs(ctx["outputs_path"])
    if smoke_outputs is not None:
        write_smoke_and_repro_artifacts(ctx["summary"], smoke_outputs, ctx["outputs_path"])
    write_enforcement_handoff_gate(ctx["outputs_path"])
    write_final_review(ctx["outputs_path"], ctx["reports_dir"])
    write_contract_log_bundle(ctx["summary"], ctx["inputs_path"], ctx["outputs_path"], ctx["max_events"])


COMPONENTS: list[Component] = [
    Component("summary", "Normalize and validate the analysis summary.", frozenset(), frozenset({"summary"}), _summary, frozenset({"config"})),
    Component("runtime_contract", "Write runtime policy, regions, blinding, and execution-contract artifacts.", frozenset({"summary"}), frozenset({"policy_defaults"}), _runtime_contract, frozenset({"config", "reporting"})),
    Component("preflight", "Run repository preflight checks.", frozenset({"summary"}), frozenset(), _preflight, frozenset({"validation"})),
    Component("metadata", "Build metadata rows and resolution artifacts.", frozenset(), frozenset({"metadata_rows"}), _metadata, frozenset({"samples"})),
    Component("registry", "Build sample registry and normalization table.", frozenset({"summary"}), frozenset({"registry", "process_roles"}), _registry, frozenset({"samples"})),
    Component("strategy", "Build background-model strategy and CR/SR constraint map.", frozenset({"summary", "registry"}), frozenset({"classification", "strategy", "constraint_map"}), _strategy, frozenset({"modeling"})),
    Component("partition", "Write analysis partition specification.", frozenset({"summary"}), frozenset(), _partition, frozenset({"selections"})),
    Component("samples", "Process selected data, signal, and background samples.", frozenset({"summary", "registry"}), frozenset({"processed_samples"}), _samples, frozenset({"samples"})),
    Component("templates", "Build histogram/template artifacts.", frozenset({"summary", "processed_samples"}), frozenset(), _templates, frozenset({"hists", "modeling"})),
    Component("cutflow", "Build cutflow and yield artifacts.", frozenset({"processed_samples"}), frozenset({"cutflow_table"}), _cutflow, frozenset({"reporting"})),
    Component("fit", "Run local RooFit model construction and measurement fit.", frozenset({"summary", "registry", "processed_samples"}), frozenset({"fit_context"}), _fit, frozenset({"stats"})),
    Component("systematics", "Build systematics artifacts.", frozenset({"summary", "registry"}), frozenset(), _systematics, frozenset({"stats"})),
    Component("significance", "Run expected/observed significance stage.", frozenset({"summary", "fit_context"}), frozenset(), _significance, frozenset({"stats"})),
    Component("plots", "Generate blinded analysis plots.", frozenset({"summary", "processed_samples", "fit_context", "cutflow_table"}), frozenset({"plot_manifest"}), _plots, frozenset({"plotting"})),
    Component("review_artifacts", "Write downstream reviewer/check artifacts.", frozenset({"registry", "processed_samples", "fit_context", "plot_manifest", "policy_defaults"}), frozenset(), _review_artifacts, frozenset({"reporting", "validation"})),
    Component("report", "Build final report and handoff artifacts.", frozenset({"summary"}), frozenset(), _report, frozenset({"reporting"})),
]


def available_components() -> list[dict[str, Any]]:
    return [
        {
            "name": component.name,
            "groups": sorted(component.groups),
            "requires": sorted(component.requires),
            "provides": sorted(component.provides),
            "description": component.description,
        }
        for component in COMPONENTS
    ]


def run_modular_pipeline(
    *,
    summary: Path,
    inputs: Path,
    outputs: Path,
    max_events: int | None = None,
    unblind_observed_significance: bool = False,
    mask: set[str] | None = None,
    strict_mask: bool = False,
) -> Context:
    mask = set(mask or set())
    unknown = mask - _component_names_and_groups(COMPONENTS)
    if unknown:
        raise ValueError(f"Unknown component/group in mask: {sorted(unknown)}")

    ctx = _init_context(
        summary=summary,
        inputs=inputs,
        outputs=outputs,
        max_events=max_events,
        unblind_observed_significance=unblind_observed_significance,
    )
    records: list[dict[str, Any]] = []
    write_state(ctx["outputs_path"], COMPONENTS, records=records, mask=mask)
    for component in COMPONENTS:
        if _is_masked(component, mask):
            records.append({"name": component.name, "status": "masked", "reason": "explicit mask"})
            write_state(ctx["outputs_path"], COMPONENTS, records=records, mask=mask)
            continue
        missing = sorted(key for key in component.requires if key not in ctx)
        if missing:
            reason = f"missing required context: {', '.join(missing)}"
            if strict_mask:
                _record_manifest(ctx, records, mask)
                raise RuntimeError(f"Cannot run component {component.name}: {reason}")
            records.append({"name": component.name, "status": "masked", "reason": reason})
            write_state(ctx["outputs_path"], COMPONENTS, records=records, mask=mask)
            continue
        component.run(ctx)
        records.append({"name": component.name, "status": "ran", "reason": ""})
        write_state(ctx["outputs_path"], COMPONENTS, records=records, mask=mask)

    _record_manifest(ctx, records, mask)
    ctx["component_records"] = records
    return ctx
