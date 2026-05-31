from __future__ import annotations

from pathlib import Path
from typing import Any

from analysis.common import utcnow_iso, write_json


ARTIFACT_CONTRACTS: dict[str, dict[str, list[str]]] = {
    "summary": {
        "required": [
            "summary.normalized.json",
            "validation/inventory.json",
            "validation/diagnostics.json",
            "validation/overlap_policy.json",
        ],
        "optional": [],
    },
    "runtime_contract": {
        "required": [
            "report/runtime_recovery.json",
            "report/enforcement_policy_defaults.json",
            "report/blinding_summary.json",
            "report/execution_contract.json",
        ],
        "optional": [],
    },
    "preflight": {
        "required": ["report/preflight_fact_check.json"],
        "optional": [],
    },
    "metadata": {
        "required": ["normalization/metadata_resolution.json"],
        "optional": ["skills/metadata.csv"],
    },
    "registry": {
        "required": [
            "samples.registry.json",
            "report/mc_sample_selection.json",
            "normalization/norm_table.json",
        ],
        "optional": [],
    },
    "strategy": {
        "required": [
            "samples.classification.json",
            "background_modeling_strategy.json",
            "cr_sr_constraint_map.json",
        ],
        "optional": [],
    },
    "partition": {
        "required": ["partition/partition_spec.json"],
        "optional": [],
    },
    "samples": {
        "required": [
            "cache/*.npz",
            "hists/processed_samples.json",
        ],
        "optional": [],
    },
    "templates": {
        "required": [
            "hists/templates.json",
            "hists/processed_samples.json",
        ],
        "optional": [],
    },
    "cutflow": {
        "required": [
            "report/cutflow_table.json",
            "report/yields_by_category.json",
            "hists/processed_samples.json",
        ],
        "optional": [],
    },
    "fit": {
        "required": [
            "fit/FIT1/results.json",
            "fit/FIT1/workspace.root",
            "fit/FIT1/fit_provenance.json",
            "fit/FIT1/background_pdf_choice.json",
            "fit/FIT1/signal_pdf.json",
            "fit/FIT1/measurement_dataset.json",
            "fit/workspace.json",
        ],
        "optional": [
            "fit/FIT1/hhxyy_workspace/manifest.json",
        ],
    },
    "systematics": {
        "required": [
            "systematics.json",
            "systematics_provenance.json",
            "systematics_sample_mapping.json",
        ],
        "optional": [],
    },
    "significance": {
        "required": [
            "fit/FIT1/significance.json",
            "fit/FIT1/significance_asimov.json",
            "fit/FIT1/significance_asimov_plot_payload.json",
            "fit/FIT1/significance_parameter_policy.json",
        ],
        "optional": [
            "fit/FIT1/hhxyy_workspace/fitting/fit/bestfit_mu_asimovData_1.root",
            "fit/FIT1/hhxyy_workspace/fitting/fit/bestfit_mu0_asimovData_1.root",
        ],
    },
    "plots": {
        "required": ["report/plots/manifest.json"],
        "optional": ["report/plots/**/*"],
    },
    "review_artifacts": {
        "required": [
            "report/data_mc_check_log.json",
            "report/data_mc_discrepancy_audit.json",
            "report/background_template_smoothing_check.json",
            "report/background_template_smoothing_provenance.json",
            "report/mc_effective_lumi_check.json",
            "report/verification_status.json",
        ],
        "optional": [
            "report/skill_extraction_summary.json",
            "report/skill_refresh_plan.json",
            "report/skill_refresh_log.jsonl",
            "report/skill_checkpoint_status.json",
        ],
    },
    "report": {
        "required": [
            "report/report.md",
            "report/enforcement_handoff_gate.json",
            "report/final_report_review.json",
            "report/final_handoff_state.json",
            "report/run_manifest.json",
        ],
        "optional": [
            "report/artifact_link_inventory.json",
            "report/completion_status.json",
            "report/reviewer_outcomes.json",
            "report/stage_execution_log.json",
        ],
    },
}


CONTEXT_HYDRATION: dict[str, dict[str, Any]] = {
    "summary": {
        "status": "supported",
        "source": "summary.normalized.json",
        "notes": "Plain JSON summary can be loaded directly.",
    },
    "registry": {
        "status": "supported",
        "source": "samples.registry.json",
        "notes": "Plain JSON registry can be loaded directly.",
    },
    "processed_samples": {
        "status": "supported",
        "source": "hists/processed_samples.json plus cache/*.npz",
        "notes": "Use the existing processed-sample cache hydration pattern from analysis.plotting.hhxyy_fit_plots.",
    },
    "cutflow_table": {
        "status": "supported",
        "source": "report/cutflow_table.json",
        "notes": "Plain JSON cutflow table can be loaded directly.",
    },
    "plot_manifest": {
        "status": "supported",
        "source": "report/plots/manifest.json",
        "notes": "Plain JSON plot manifest can be loaded directly.",
    },
    "policy_defaults": {
        "status": "supported",
        "source": "report/enforcement_policy_defaults.json",
        "notes": "Plain JSON policy defaults can be loaded directly.",
    },
    "fit_context": {
        "status": "not_supported",
        "source": "fit/FIT1/*.json and workspace.root",
        "notes": "A live RooFit context is not exactly recoverable from current artifacts; rerun fit to recreate it.",
    },
    "metadata_rows": {
        "status": "partial",
        "source": "normalization/metadata_resolution.json or skills/metadata.csv",
        "notes": "Useful for inspection; not currently wired as a resumable context loader.",
    },
    "process_roles": {
        "status": "partial",
        "source": "report/mc_sample_selection.json",
        "notes": "Useful for inspection; registry is the main downstream input.",
    },
    "classification": {
        "status": "partial",
        "source": "samples.classification.json",
        "notes": "Useful for inspection; not currently required by downstream components.",
    },
    "strategy": {
        "status": "partial",
        "source": "background_modeling_strategy.json",
        "notes": "Useful for inspection; not currently required by downstream components.",
    },
    "constraint_map": {
        "status": "partial",
        "source": "cr_sr_constraint_map.json",
        "notes": "Useful for inspection; not currently required by downstream components.",
    },
}


def _matches(outputs: Path, pattern: str) -> list[str]:
    return sorted(str(path.relative_to(outputs)) for path in outputs.glob(pattern) if path.exists())


def _artifact_check(outputs: Path, pattern: str) -> dict[str, Any]:
    matches = _matches(outputs, pattern)
    return {
        "pattern": pattern,
        "exists": bool(matches),
        "matches": matches,
    }


def _component_record_map(records: list[dict[str, Any]] | None) -> dict[str, dict[str, Any]]:
    return {record["name"]: record for record in records or [] if "name" in record}


def _artifact_status(required: list[dict[str, Any]], optional: list[dict[str, Any]], run_status: str | None) -> str:
    if required and all(item["exists"] for item in required):
        return "complete"
    if any(item["exists"] for item in required + optional):
        return "partial"
    if run_status == "ran" and not required:
        return "complete"
    if run_status == "masked":
        return "masked"
    return "missing"


def inspect_outputs(
    outputs: Path,
    components: list[Any],
    *,
    records: list[dict[str, Any]] | None = None,
    mask: set[str] | None = None,
) -> dict[str, Any]:
    outputs = Path(outputs)
    record_map = _component_record_map(records)
    provider_for_context: dict[str, str] = {}
    component_payloads = []
    component_status: dict[str, str] = {}

    for component in components:
        for provided in component.provides:
            provider_for_context[provided] = component.name

    for component in components:
        contract = ARTIFACT_CONTRACTS.get(component.name, {"required": [], "optional": []})
        required = [_artifact_check(outputs, pattern) for pattern in contract.get("required", [])]
        optional = [_artifact_check(outputs, pattern) for pattern in contract.get("optional", [])]
        record = record_map.get(component.name, {})
        status = _artifact_status(required, optional, record.get("status"))
        component_status[component.name] = status
        component_payloads.append(
            {
                "name": component.name,
                "run_status": record.get("status", "unknown"),
                "run_reason": record.get("reason", ""),
                "artifact_status": status,
                "requires_context": sorted(component.requires),
                "provides_context": sorted(component.provides),
                "groups": sorted(component.groups),
                "required_artifacts": required,
                "optional_artifacts": optional,
            }
        )

    context_payloads = {}
    for key, provider in sorted(provider_for_context.items()):
        provider_status = component_status.get(provider, "missing")
        hydration = CONTEXT_HYDRATION.get(
            key,
            {
                "status": "not_supported",
                "source": None,
                "notes": "No hydration contract is documented for this context key.",
            },
        )
        context_payloads[key] = {
            "provider_component": provider,
            "provider_artifact_status": provider_status,
            "artifact_available": provider_status == "complete",
            "hydration": hydration,
        }

    entrypoints = []
    for component in components:
        requirements = []
        missing_artifacts = []
        unsupported = []
        partial = []
        for key in sorted(component.requires):
            context_status = context_payloads.get(key)
            if context_status is None:
                missing_artifacts.append(key)
                requirements.append({"context": key, "status": "unknown_provider"})
                continue
            hydration_status = context_status["hydration"]["status"]
            artifact_available = bool(context_status["artifact_available"])
            if not artifact_available:
                missing_artifacts.append(key)
            if hydration_status == "not_supported":
                unsupported.append(key)
            if hydration_status == "partial":
                partial.append(key)
            requirements.append(
                {
                    "context": key,
                    "provider_component": context_status["provider_component"],
                    "provider_artifact_status": context_status["provider_artifact_status"],
                    "artifact_available": artifact_available,
                    "hydration_status": hydration_status,
                    "hydration_source": context_status["hydration"].get("source"),
                    "notes": context_status["hydration"].get("notes"),
                }
            )

        if not component.requires:
            readiness = "ready_without_prior_artifacts"
        elif missing_artifacts:
            readiness = "blocked_missing_artifacts"
        elif unsupported:
            readiness = "blocked_requires_live_context"
        elif partial:
            readiness = "artifact_present_hydration_partial"
        else:
            readiness = "ready_from_artifacts"

        entrypoints.append(
            {
                "component": component.name,
                "readiness": readiness,
                "cli_resume_supported": False,
                "requirements": requirements,
                "notes": (
                    "The current CLI reports readiness but does not yet implement --start-at hydration."
                    if readiness in {"ready_from_artifacts", "artifact_present_hydration_partial"}
                    else ""
                ),
            }
        )

    ready_entrypoints = [
        item["component"]
        for item in entrypoints
        if item["readiness"] in {"ready_without_prior_artifacts", "ready_from_artifacts"}
    ]
    return {
        "status": "ok",
        "pipeline": "modular_pipeline",
        "timestamp_utc": utcnow_iso(),
        "outputs": str(outputs),
        "mask": sorted(mask or set()),
        "components": component_payloads,
        "contexts": context_payloads,
        "entrypoints": entrypoints,
        "ready_entrypoints_from_artifacts": ready_entrypoints,
        "limitations": [
            "This artifact reports entrypoint readiness from disk artifacts.",
            "The CLI does not yet implement --start-at hydration; use this ledger to decide what a future hydration/resume command may safely load.",
            "fit_context is a live RooFit object graph and should be recreated by rerunning fit rather than guessed from JSON.",
        ],
    }


def write_state(
    outputs: Path,
    components: list[Any],
    *,
    records: list[dict[str, Any]] | None = None,
    mask: set[str] | None = None,
) -> dict[str, Any]:
    state = inspect_outputs(outputs, components, records=records, mask=mask)
    write_json(state, Path(outputs) / "modular_pipeline_state.json")
    return state
