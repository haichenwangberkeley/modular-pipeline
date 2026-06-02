from __future__ import annotations

import argparse
import itertools
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from optimization_infra.plan_candidate_run import (
    load_artifact_manifests,
    load_structured,
    plan_candidate_run,
    set_path,
    stable_hash,
    write_execution_plan,
    write_yaml,
)
from optimization_infra.version_round import git_state, make_version_name


SUPPORTED_RUN_TYPES = {
    "baseline",
    "single_candidate",
    "parameter_scan",
    "structured_scan",
    "qualitative_strategy_branch",
    "validation_only",
    "service_extension_validation",
}


def validate_scan_spec(spec: dict[str, Any]) -> None:
    required = {"scan_id", "parent_run_id", "strategy_id", "objective_metric", "parameters", "execution", "reporting"}
    missing = sorted(required - set(spec))
    if missing:
        raise ValueError(f"Scan spec missing required fields: {', '.join(missing)}")
    for parameter in spec["parameters"]:
        if "config_path" not in parameter or "values" not in parameter:
            raise ValueError("Each scan parameter requires config_path and values")


def candidate_points(parameters: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keys = [item["config_path"] for item in parameters]
    values = [item["values"] for item in parameters]
    return [dict(zip(keys, combo, strict=True)) for combo in itertools.product(*values)]


def append_run_registry(registry_path: str | Path, entry: dict[str, Any]) -> None:
    if entry.get("run_type") not in SUPPORTED_RUN_TYPES:
        raise ValueError(f"Unsupported run_type: {entry.get('run_type')}")
    path = Path(registry_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(entry, sort_keys=True) + "\n")


def make_registry_entry(
    *,
    run_id: str,
    scan_id: str,
    spec: dict[str, Any],
    config_path: Path,
    plan: dict[str, Any],
) -> dict[str, Any]:
    state = git_state(Path.cwd())
    descriptor = ",".join(change["config_path"] for change in plan.get("changed_parameters", [])) or "no-config-change"
    version_name = make_version_name(
        round_id=run_id,
        strategy_id=spec["strategy_id"],
        objective=spec["objective_metric"],
        descriptor=descriptor,
    )
    return {
        "run_id": run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "parent_run_id": spec["parent_run_id"],
        "branch_id": spec.get("branch_id", spec["strategy_id"]),
        "strategy_id": spec["strategy_id"],
        "scan_id": scan_id,
        "run_type": "parameter_scan",
        "objective": spec.get("objective", f"Scan {spec['objective_metric']}"),
        "configuration_snapshot_path": str(config_path),
        "changed_parameters": plan.get("changed_parameters", []),
        "reused_artifacts": plan.get("reused_artifacts", []),
        "invalidated_artifacts": plan.get("invalidated_artifacts", []),
        "stages_executed": [s["stage_id"] for s in plan.get("stages", []) if s["decision"] == "rerun"],
        "stages_skipped": [s["stage_id"] for s in plan.get("stages", []) if s["decision"] == "reuse"],
        "verifier_status": "not_run",
        "cut_flow_path": None,
        "yield_table_path": None,
        "fit_output_path": None,
        "metric_values": {},
        "expected_significance": None,
        "observed_significance": None,
        "runtime": None,
        "warnings": plan.get("invariant_violations", []),
        "failure_reason": "blocked_by_invariant" if plan.get("status") == "blocked" else None,
        "human_or_agent_note": "Infrastructure dry run; no scientific workflow launched.",
        "git_state": state,
        "service_versions": {},
        "version_name": version_name,
        "version_ref": state["commit"],
    }


def run_scan(
    *,
    scan_spec: dict[str, Any],
    runs_dir: str | Path = "runs",
    registry_path: str | Path = "optimization_infra/runs.jsonl",
) -> dict[str, Any]:
    validate_scan_spec(scan_spec)
    runs_dir = Path(runs_dir)
    scan_id = scan_spec["scan_id"]
    base_config = load_structured(scan_spec["base_config"])
    parent_config = load_structured(scan_spec.get("parent_config", scan_spec["base_config"]))
    workflow_graph = load_structured(scan_spec.get("workflow_graph", "optimization_infra/workflow_graph.yaml"))
    artifact_manifests = load_artifact_manifests(scan_spec.get("artifact_manifests", []))
    ledger = load_structured(scan_spec.get("ledger", "ledger/EXECUTABLE_SERVICES.yaml"))
    invariants = load_structured(scan_spec.get("global_invariants", "ledger/GLOBAL_INVARIANTS.yaml"))

    scan_dir = runs_dir / scan_id
    (scan_dir / "config_snapshot").mkdir(parents=True, exist_ok=True)
    candidates = []
    for index, point in enumerate(candidate_points(scan_spec["parameters"]), start=1):
        candidate_config = base_config
        for path, value in point.items():
            candidate_config = set_path(candidate_config, path, value)
        run_id = f"{scan_id}_candidate_{index:03d}_{stable_hash(point)[:8]}"
        run_dir = runs_dir / run_id
        snapshot_path = run_dir / "config_snapshot" / "candidate.yaml"
        snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        write_yaml(snapshot_path, candidate_config)

        plan = plan_candidate_run(
            run_id=run_id,
            parent_run_id=scan_spec["parent_run_id"],
            parent_config=parent_config,
            candidate_config=candidate_config,
            workflow_graph=workflow_graph,
            artifact_manifests=artifact_manifests,
            executable_services_ledger=ledger,
            global_invariants=invariants,
        )
        write_execution_plan(run_dir, plan)
        append_run_registry(
            registry_path,
            make_registry_entry(
                run_id=run_id,
                scan_id=scan_id,
                spec=scan_spec,
                config_path=snapshot_path,
                plan=plan,
            ),
        )
        candidates.append({"run_id": run_id, "point": point, "plan_status": plan["status"], "earliest_stage": plan["earliest_stage_to_rerun"]})

    summary = {
        "scan_id": scan_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dry_run_only": bool(scan_spec.get("execution", {}).get("dry_run_only", True)),
        "candidate_count": len(candidates),
        "candidates": candidates,
    }
    write_yaml(scan_dir / "scan_dry_run_summary.yaml", summary)
    (scan_dir / "SCAN_DRY_RUN_SUMMARY.md").write_text(render_scan_dry_run_markdown(summary))
    return summary


def render_scan_dry_run_markdown(summary: dict[str, Any]) -> str:
    lines = [
        f"# Scan Dry-Run Summary: {summary['scan_id']}",
        "",
        f"Candidates planned: {summary['candidate_count']}",
        "",
    ]
    for candidate in summary["candidates"]:
        lines.append(f"- `{candidate['run_id']}`: status `{candidate['plan_status']}`, earliest stage `{candidate['earliest_stage']}`")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create dry-run plans for a configuration-driven scan.")
    parser.add_argument("--scan-spec", required=True)
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--registry", default="optimization_infra/runs.jsonl")
    args = parser.parse_args()
    summary = run_scan(scan_spec=load_structured(args.scan_spec), runs_dir=args.runs_dir, registry_path=args.registry)
    print(yaml.safe_dump(summary, sort_keys=False))


if __name__ == "__main__":
    main()
