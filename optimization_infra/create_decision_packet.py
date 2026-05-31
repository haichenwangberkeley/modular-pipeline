from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

from optimization_infra.plan_candidate_run import load_structured, write_yaml


def create_decision_packet(
    *,
    decision_id: str,
    objective: dict[str, Any],
    lineage: dict[str, Any],
    evidence_paths: list[str],
    runs_dir: str | Path = "runs",
) -> dict[str, Any]:
    evidence = []
    key_observations = []
    for path in evidence_paths:
        loaded = load_structured(path)
        evidence.append({"path": path, "kind": _kind_from_path(path)})
        if isinstance(loaded, dict):
            if "observations" in loaded:
                key_observations.extend(loaded.get("observations", []))
            elif "interpretation" in loaded:
                key_observations.append({"run_id": loaded.get("run_id"), "status": loaded["interpretation"].get("status")})

    packet = {
        "decision_id": decision_id,
        "current_objective": objective,
        "analysis_lineage": lineage,
        "key_observations": key_observations,
        "evidence": evidence,
        "current_interpretation": {
            "direct_observations": key_observations,
            "plausible_interpretation": "Decision packet assembled from structured observations; human reasoning required.",
            "unresolved_uncertainty": [],
            "possible_implementation_issue": [],
        },
        "suggested_decision_types": [
            "continue local scanning",
            "narrow the scan around a promising region",
            "broaden the scan",
            "scan a correlated parameter family",
            "inspect an anomalous category or region",
            "add a new category using existing services",
            "test a new derived variable",
            "propose a new executable-service extension",
            "repair a possible implementation problem",
            "stop because improvements have saturated",
            "request human clarification",
        ],
        "requested_response_format": {
            "decision_id": None,
            "selected_next_step_type": None,
            "rationale": None,
            "proposed_parent_run_id": None,
            "strategy_id": None,
            "parameters_or_qualitative_changes_to_test": None,
            "expected_mechanism_of_improvement": None,
            "stages_likely_invalidated": None,
            "required_verification": None,
            "service_changes_required": None,
            "human_approval_required": None,
        },
    }
    decision_dir = Path(runs_dir) / decision_id
    write_yaml(decision_dir / "decision_packet.yaml", packet)
    (decision_dir / "DECISION_PACKET.md").write_text(render_decision_packet_markdown(packet))
    return packet


def _kind_from_path(path: str) -> str:
    name = Path(path).name
    if "scan_summary" in name:
        return "scan_summary"
    if "observations" in name:
        return "observation"
    return "evidence"


def render_decision_packet_markdown(packet: dict[str, Any]) -> str:
    lines = [
        f"# Decision Packet: {packet['decision_id']}",
        "",
        "## Current Objective",
    ]
    for key, value in packet["current_objective"].items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Evidence"])
    for item in packet["evidence"]:
        lines.append(f"- `{item['kind']}`: `{item['path']}`")
    lines.extend(["", "## Suggested Decision Types"])
    for item in packet["suggested_decision_types"]:
        lines.append(f"- {item}")
    lines.extend(["", "## Requested Response Format", "Return the fields listed in `decision_packet.yaml`.", ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a human/reasoning-agent decision packet.")
    parser.add_argument("--decision-id", required=True)
    parser.add_argument("--objective", required=True, help="YAML/JSON objective mapping")
    parser.add_argument("--lineage", required=True, help="YAML/JSON lineage mapping")
    parser.add_argument("--evidence", action="append", required=True)
    parser.add_argument("--runs-dir", default="runs")
    args = parser.parse_args()
    packet = create_decision_packet(
        decision_id=args.decision_id,
        objective=load_structured(args.objective),
        lineage=load_structured(args.lineage),
        evidence_paths=args.evidence,
        runs_dir=args.runs_dir,
    )
    print(yaml.safe_dump(packet, sort_keys=False))


if __name__ == "__main__":
    main()
