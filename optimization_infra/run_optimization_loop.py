from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from optimization_infra.create_decision_packet import create_decision_packet
from optimization_infra.plan_candidate_run import load_structured, write_yaml
from optimization_infra.run_scan import run_scan
from optimization_infra.version_round import create_round_version_record


def validate_loop_spec(spec: dict[str, Any]) -> None:
    required = {"loop_id", "objective_metric", "max_rounds", "rounds"}
    missing = sorted(required - set(spec))
    if missing:
        raise ValueError(f"Loop spec missing required fields: {', '.join(missing)}")
    if int(spec["max_rounds"]) < 1:
        raise ValueError("max_rounds must be at least 1")
    for item in spec["rounds"]:
        if "round_id" not in item or "descriptor" not in item or "scan_spec" not in item:
            raise ValueError("Each loop round requires round_id, descriptor, and scan_spec")


def run_optimization_loop(
    *,
    loop_spec: dict[str, Any],
    runs_dir: str | Path = "runs",
    registry_path: str | Path = "optimization_infra/runs.jsonl",
    repo: str | Path = ".",
) -> dict[str, Any]:
    validate_loop_spec(loop_spec)
    runs_dir = Path(runs_dir)
    loop_id = loop_spec["loop_id"]
    loop_dir = runs_dir / loop_id
    loop_dir.mkdir(parents=True, exist_ok=True)

    max_rounds = int(loop_spec["max_rounds"])
    executed_rounds = []
    stopped_reason = None
    for index, round_spec in enumerate(loop_spec["rounds"][:max_rounds], start=1):
        round_id = round_spec["round_id"]
        round_dir = loop_dir / round_id
        scan_spec = load_structured(round_spec["scan_spec"])
        scan_summary = run_scan(
            scan_spec=scan_spec,
            runs_dir=runs_dir,
            registry_path=registry_path,
        )
        version = create_round_version_record(
            run_dir=round_dir,
            round_id=round_id,
            strategy_id=scan_spec.get("strategy_id", loop_spec.get("strategy_id", "unspecified_strategy")),
            objective=loop_spec["objective_metric"],
            descriptor=round_spec["descriptor"],
            repo=repo,
            create_git_tag=bool(round_spec.get("create_git_tag", False)),
        )
        blocked = any(candidate.get("plan_status") == "blocked" for candidate in scan_summary.get("candidates", []))
        round_report = {
            "loop_id": loop_id,
            "round_id": round_id,
            "round_index": index,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "descriptor": round_spec["descriptor"],
            "scan_id": scan_summary["scan_id"],
            "candidate_count": scan_summary["candidate_count"],
            "blocked": blocked,
            "version_name": version["version_name"],
            "scan_summary_path": str(Path(runs_dir) / scan_summary["scan_id"] / "scan_dry_run_summary.yaml"),
            "decision": "stop_or_escalate" if blocked else "continue_or_review",
        }
        write_yaml(round_dir / "ROUND_REPORT.yaml", round_report)
        (round_dir / "ROUND_REPORT.md").write_text(render_round_report_markdown(round_report))
        packet = create_decision_packet(
            decision_id=f"{loop_id}_{round_id}_decision",
            objective={
                "scientific_goal": loop_spec.get("scientific_goal", "not_specified"),
                "optimization_metric": loop_spec["objective_metric"],
                "round_descriptor": round_spec["descriptor"],
            },
            lineage={
                "loop_id": loop_id,
                "round_id": round_id,
                "scan_id": scan_summary["scan_id"],
                "strategy_id": scan_spec.get("strategy_id"),
                "parent_run_id": scan_spec.get("parent_run_id"),
            },
            evidence_paths=[str(Path(runs_dir) / scan_summary["scan_id"] / "scan_dry_run_summary.yaml")],
            runs_dir=runs_dir,
        )
        round_report["decision_packet_path"] = str(Path(runs_dir) / packet["decision_id"] / "decision_packet.yaml")
        write_yaml(round_dir / "ROUND_REPORT.yaml", round_report)
        (round_dir / "ROUND_REPORT.md").write_text(render_round_report_markdown(round_report))
        executed_rounds.append(round_report)
        if blocked and loop_spec.get("stop_on_blocked_plan", True):
            stopped_reason = "blocked_candidate_plan"
            break

    if stopped_reason is None and len(executed_rounds) >= max_rounds:
        stopped_reason = "max_rounds_reached"
    loop_summary = {
        "loop_id": loop_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "objective_metric": loop_spec["objective_metric"],
        "max_rounds": max_rounds,
        "rounds_completed": len(executed_rounds),
        "stopped_reason": stopped_reason,
        "rounds": executed_rounds,
    }
    write_yaml(loop_dir / "LOOP_SUMMARY.yaml", loop_summary)
    (loop_dir / "LOOP_SUMMARY.md").write_text(render_loop_summary_markdown(loop_summary))
    return loop_summary


def render_round_report_markdown(report: dict[str, Any]) -> str:
    lines = [
        f"# Round Report: {report['round_id']}",
        "",
        f"Loop: `{report['loop_id']}`",
        f"Descriptor: {report['descriptor']}",
        f"Scan: `{report['scan_id']}`",
        f"Candidates planned: `{report['candidate_count']}`",
        f"Version name: `{report['version_name']}`",
        f"Decision: `{report['decision']}`",
    ]
    if report.get("decision_packet_path"):
        lines.append(f"Decision packet: `{report['decision_packet_path']}`")
    lines.append("")
    return "\n".join(lines)


def render_loop_summary_markdown(summary: dict[str, Any]) -> str:
    lines = [
        f"# Optimization Loop Summary: {summary['loop_id']}",
        "",
        f"Objective metric: `{summary['objective_metric']}`",
        f"Rounds completed: `{summary['rounds_completed']}` / `{summary['max_rounds']}`",
        f"Stopped reason: `{summary['stopped_reason']}`",
        "",
        "## Rounds",
    ]
    for item in summary["rounds"]:
        lines.append(f"- `{item['round_id']}`: `{item['decision']}` using version `{item['version_name']}`")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a bounded, report-driven optimization loop in dry-run mode.")
    parser.add_argument("--loop-spec", required=True)
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--registry", default="optimization_infra/runs.jsonl")
    parser.add_argument("--repo", default=".")
    args = parser.parse_args()
    summary = run_optimization_loop(
        loop_spec=load_structured(args.loop_spec),
        runs_dir=args.runs_dir,
        registry_path=args.registry,
        repo=args.repo,
    )
    print(yaml.safe_dump(summary, sort_keys=False))


if __name__ == "__main__":
    main()
