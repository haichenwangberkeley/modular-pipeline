from __future__ import annotations

import argparse
import copy
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


REQUIRED_MANIFEST_FIELDS = {
    "artifact_id",
    "artifact_type",
    "producing_run_id",
    "producing_stage_id",
    "created_at",
    "path",
    "content_hash",
    "schema_version",
    "service_id",
    "service_version",
    "git",
    "config_subset",
    "config_subset_hash",
    "upstream_artifact_hashes",
    "verifier_status",
    "reusable",
}


class PlanningError(ValueError):
    """Raised when inputs cannot be planned safely."""


def load_structured(path: str | Path) -> Any:
    path = Path(path)
    text = path.read_text()
    if path.suffix in {".yaml", ".yml"}:
        return yaml.safe_load(text) or {}
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    return json.loads(text)


def write_yaml(path: str | Path, data: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False))


def stable_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def get_path(data: dict[str, Any], dotted_path: str) -> Any:
    current: Any = data
    for part in dotted_path.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def set_path(data: dict[str, Any], dotted_path: str, value: Any) -> dict[str, Any]:
    result = copy.deepcopy(data)
    current = result
    parts = dotted_path.split(".")
    for part in parts[:-1]:
        current = current.setdefault(part, {})
        if not isinstance(current, dict):
            raise PlanningError(f"Cannot set nested config path through non-mapping: {dotted_path}")
    current[parts[-1]] = value
    return result


def config_subset(config: dict[str, Any], dependencies: list[str]) -> dict[str, Any]:
    return {path: get_path(config, path) for path in dependencies}


def changed_parameters(parent_config: dict[str, Any], candidate_config: dict[str, Any]) -> list[dict[str, Any]]:
    paths = sorted(set(_flatten_paths(parent_config)) | set(_flatten_paths(candidate_config)))
    changes = []
    for path in paths:
        before = get_path(parent_config, path)
        after = get_path(candidate_config, path)
        if before != after:
            changes.append({"config_path": path, "parent": before, "candidate": after})
    return changes


def _flatten_paths(value: Any, prefix: str = "") -> list[str]:
    if not isinstance(value, dict):
        return [prefix] if prefix else []
    paths: list[str] = []
    for key, child in value.items():
        child_prefix = f"{prefix}.{key}" if prefix else str(key)
        paths.extend(_flatten_paths(child, child_prefix))
    return paths


def load_artifact_manifests(paths: list[str | Path]) -> list[dict[str, Any]]:
    artifacts: list[dict[str, Any]] = []
    for path in paths:
        loaded = load_structured(path)
        if isinstance(loaded, dict) and "artifacts" in loaded:
            artifacts.extend(loaded["artifacts"])
        elif isinstance(loaded, list):
            artifacts.extend(loaded)
        elif isinstance(loaded, dict):
            artifacts.append(loaded)
        else:
            raise PlanningError(f"Unsupported artifact manifest shape in {path}")
    return artifacts


def service_versions_from_ledger(ledger: dict[str, Any]) -> dict[str, str]:
    return {item["service_id"]: str(item.get("version", "")) for item in ledger.get("services", [])}


def invariant_violations(candidate_config: dict[str, Any], global_invariants: dict[str, Any]) -> list[str]:
    violations = []
    observed_allowed = (
        get_path(candidate_config, "runtime_defaults.blinding.observed_significance_allowed")
        or get_path(candidate_config, "blinding.observed_significance_allowed")
    )
    approval = get_path(candidate_config, "approvals.observed_significance_unblinding")
    if observed_allowed and not approval:
        violations.append("Observed significance unblinding requested without recorded approval.")

    changed_policy = get_path(candidate_config, "physics_policy_changed")
    if changed_policy:
        violations.append("Candidate declares a physics-policy change; ordinary optimization planning is blocked.")

    if not isinstance(global_invariants, dict):
        violations.append("Global invariants could not be loaded as a mapping.")
    return violations


def find_stage_artifacts(
    artifacts: list[dict[str, Any]],
    *,
    parent_run_id: str | None,
    stage: dict[str, Any],
) -> list[dict[str, Any]]:
    stage_outputs = {item["artifact_type"]: str(item.get("schema_version")) for item in stage.get("outputs", [])}
    matches = []
    for artifact in artifacts:
        if parent_run_id and artifact.get("producing_run_id") != parent_run_id:
            continue
        if artifact.get("producing_stage_id") != stage["stage_id"]:
            continue
        if artifact.get("artifact_type") in stage_outputs:
            matches.append(artifact)
    return matches


def artifact_reuse_problems(
    artifact: dict[str, Any],
    *,
    stage: dict[str, Any],
    expected_config_hash: str,
    service_versions: dict[str, str],
) -> list[str]:
    problems = []
    missing = sorted(REQUIRED_MANIFEST_FIELDS - set(artifact))
    if missing:
        problems.append(f"incomplete provenance: missing {', '.join(missing)}")
        return problems

    if artifact.get("verifier_status") != "pass":
        problems.append(f"verifier status is {artifact.get('verifier_status')!r}, not 'pass'")
    if artifact.get("reusable") is not True:
        problems.append("artifact is not marked reusable")

    expected_schema = {
        item["artifact_type"]: str(item.get("schema_version"))
        for item in stage.get("outputs", [])
    }.get(artifact.get("artifact_type"))
    if expected_schema is not None and str(artifact.get("schema_version")) != expected_schema:
        problems.append(
            f"schema mismatch for {artifact.get('artifact_type')}: "
            f"{artifact.get('schema_version')} != {expected_schema}"
        )

    service_id = artifact.get("service_id")
    expected_version = service_versions.get(service_id)
    if expected_version and str(artifact.get("service_version")) != expected_version:
        problems.append(
            f"service version mismatch for {service_id}: "
            f"{artifact.get('service_version')} != {expected_version}"
        )

    if artifact.get("config_subset_hash") != expected_config_hash:
        problems.append("relevant configuration subset changed")
    return problems


def plan_candidate_run(
    *,
    run_id: str,
    parent_run_id: str | None,
    parent_config: dict[str, Any],
    candidate_config: dict[str, Any],
    workflow_graph: dict[str, Any],
    artifact_manifests: list[dict[str, Any]],
    executable_services_ledger: dict[str, Any],
    global_invariants: dict[str, Any],
) -> dict[str, Any]:
    service_versions = service_versions_from_ledger(executable_services_ledger)
    violations = invariant_violations(candidate_config, global_invariants)
    config_changes = changed_parameters(parent_config, candidate_config)

    plan: dict[str, Any] = {
        "run_id": run_id,
        "parent_run_id": parent_run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "workflow_id": workflow_graph.get("workflow_id"),
        "workflow_version": workflow_graph.get("version"),
        "status": "blocked" if violations else "planned",
        "changed_parameters": config_changes,
        "invariant_violations": violations,
        "earliest_stage_to_rerun": None,
        "stages": [],
        "reused_artifacts": [],
        "invalidated_artifacts": [],
        "dry_run": True,
    }
    if violations:
        return plan

    stage_status: dict[str, str] = {}
    downstream_invalidated = False
    for stage in workflow_graph.get("stages", []):
        stage_id = stage["stage_id"]
        deps = stage.get("depends_on", [])
        dep_invalid = any(stage_status.get(dep) == "rerun" for dep in deps)
        subset = config_subset(candidate_config, stage.get("config_dependencies", []))
        subset_hash = stable_hash(subset)
        parent_subset = config_subset(parent_config, stage.get("config_dependencies", []))
        local_config_changed = stable_hash(parent_subset) != subset_hash
        artifacts = find_stage_artifacts(artifact_manifests, parent_run_id=parent_run_id, stage=stage)
        reasons: list[str] = []
        reusable_artifacts: list[str] = []

        if dep_invalid or downstream_invalidated:
            reasons.append("upstream stage invalidated")
        if local_config_changed:
            reasons.append("stage-relevant configuration changed")
        if not artifacts:
            reasons.append("no parent artifact manifest available for stage outputs")
        else:
            for artifact in artifacts:
                problems = artifact_reuse_problems(
                    artifact,
                    stage=stage,
                    expected_config_hash=subset_hash,
                    service_versions=service_versions,
                )
                if problems:
                    reasons.extend(f"{artifact.get('artifact_id', artifact.get('artifact_type'))}: {p}" for p in problems)
                else:
                    reusable_artifacts.append(artifact["artifact_id"])

        should_rerun = bool(reasons)
        status = "rerun" if should_rerun else "reuse"
        if should_rerun:
            downstream_invalidated = True
            if plan["earliest_stage_to_rerun"] is None:
                plan["earliest_stage_to_rerun"] = stage_id
            plan["invalidated_artifacts"].extend([a.get("artifact_id", a.get("artifact_type")) for a in artifacts])
        else:
            plan["reused_artifacts"].extend(reusable_artifacts)

        stage_status[stage_id] = status
        plan["stages"].append(
            {
                "stage_id": stage_id,
                "name": stage.get("name", stage_id),
                "decision": status,
                "reasons": reasons or ["all reuse checks passed"],
                "reused_artifacts": reusable_artifacts,
                "expected_runtime_class": stage.get("expected_runtime_class"),
                "failure_handling": stage.get("failure_handling"),
            }
        )

    if plan["earliest_stage_to_rerun"] is None:
        plan["earliest_stage_to_rerun"] = "none"
    return plan


def render_execution_plan_markdown(plan: dict[str, Any]) -> str:
    lines = [
        f"# Execution Plan: {plan['run_id']}",
        "",
        f"Parent run: `{plan.get('parent_run_id')}`",
        f"Status: `{plan.get('status')}`",
        f"Earliest stage to rerun: `{plan.get('earliest_stage_to_rerun')}`",
        "",
        "## Changed Parameters",
    ]
    if plan.get("changed_parameters"):
        for change in plan["changed_parameters"]:
            lines.append(f"- `{change['config_path']}`: `{change['parent']}` -> `{change['candidate']}`")
    else:
        lines.append("- None")
    if plan.get("invariant_violations"):
        lines.extend(["", "## Blocking Invariant Violations"])
        lines.extend(f"- {item}" for item in plan["invariant_violations"])
    lines.extend(["", "## Stage Decisions"])
    for stage in plan.get("stages", []):
        lines.append(f"- `{stage['stage_id']}`: **{stage['decision']}**")
        for reason in stage.get("reasons", []):
            lines.append(f"  - {reason}")
    lines.append("")
    return "\n".join(lines)


def write_execution_plan(run_dir: str | Path, plan: dict[str, Any]) -> None:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    write_yaml(run_dir / "execution_plan.yaml", plan)
    (run_dir / "EXECUTION_PLAN.md").write_text(render_execution_plan_markdown(plan))


def main() -> None:
    parser = argparse.ArgumentParser(description="Plan a resumable candidate run without launching scientific jobs.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--parent-run-id")
    parser.add_argument("--parent-config", required=True)
    parser.add_argument("--candidate-config", required=True)
    parser.add_argument("--workflow-graph", default="optimization_infra/workflow_graph.yaml")
    parser.add_argument("--artifact-manifest", action="append", default=[])
    parser.add_argument("--ledger", default="ledger/EXECUTABLE_SERVICES.yaml")
    parser.add_argument("--global-invariants", default="ledger/GLOBAL_INVARIANTS.yaml")
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    plan = plan_candidate_run(
        run_id=args.run_id,
        parent_run_id=args.parent_run_id,
        parent_config=load_structured(args.parent_config),
        candidate_config=load_structured(args.candidate_config),
        workflow_graph=load_structured(args.workflow_graph),
        artifact_manifests=load_artifact_manifests(args.artifact_manifest),
        executable_services_ledger=load_structured(args.ledger),
        global_invariants=load_structured(args.global_invariants),
    )
    run_dir = Path(args.runs_dir) / args.run_id
    write_execution_plan(run_dir, plan)
    print(render_execution_plan_markdown(plan))


if __name__ == "__main__":
    main()
