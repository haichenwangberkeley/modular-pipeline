from __future__ import annotations

import argparse
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from optimization_infra.plan_candidate_run import write_yaml


def slugify(value: str, *, max_length: int = 80) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip().lower())
    slug = re.sub(r"-+", "-", slug).strip("-")
    return (slug or "unnamed")[:max_length].rstrip("-")


def git_state(repo: str | Path = ".") -> dict[str, Any]:
    repo = Path(repo)
    commit = _git(repo, "rev-parse", "HEAD")
    branch = _git(repo, "branch", "--show-current")
    status = _git(repo, "status", "--short")
    return {
        "commit": commit,
        "branch": branch or None,
        "dirty": bool(status),
        "status_short": status.splitlines() if status else [],
    }


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout.strip()


def make_version_name(
    *,
    round_id: str,
    strategy_id: str,
    objective: str,
    descriptor: str,
) -> str:
    parts = ["opt", round_id, strategy_id, objective, descriptor]
    return slugify("-".join(str(part) for part in parts if part))


def create_round_version_record(
    *,
    run_dir: str | Path,
    round_id: str,
    strategy_id: str,
    objective: str,
    descriptor: str,
    repo: str | Path = ".",
    create_git_tag: bool = False,
) -> dict[str, Any]:
    run_dir = Path(run_dir)
    state = git_state(repo)
    version_name = make_version_name(
        round_id=round_id,
        strategy_id=strategy_id,
        objective=objective,
        descriptor=descriptor,
    )
    tag_name = f"opt/{version_name}"
    record = {
        "round_id": round_id,
        "strategy_id": strategy_id,
        "objective": objective,
        "descriptor": descriptor,
        "version_name": version_name,
        "git_tag": tag_name if create_git_tag else None,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_state": state,
        "notes": [
            "This record is a lightweight version anchor for one optimization round.",
            "Large scientific outputs should be referenced by path/hash, not copied into this file.",
        ],
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    write_yaml(run_dir / "VERSION.yaml", record)
    (run_dir / "VERSION.md").write_text(render_version_markdown(record))
    if create_git_tag:
        subprocess.run(
            ["git", "tag", "-a", tag_name, state["commit"], "-m", f"Optimization round {round_id}: {descriptor}"],
            cwd=repo,
            check=True,
        )
    return record


def render_version_markdown(record: dict[str, Any]) -> str:
    lines = [
        f"# Version Anchor: {record['version_name']}",
        "",
        f"Round: `{record['round_id']}`",
        f"Strategy: `{record['strategy_id']}`",
        f"Objective: `{record['objective']}`",
        f"Descriptor: `{record['descriptor']}`",
        f"Git commit: `{record['git_state']['commit']}`",
        f"Git branch: `{record['git_state'].get('branch')}`",
        f"Dirty tree: `{record['git_state']['dirty']}`",
    ]
    if record.get("git_tag"):
        lines.append(f"Git tag: `{record['git_tag']}`")
    if record["git_state"].get("status_short"):
        lines.extend(["", "## Dirty Files"])
        lines.extend(f"- `{item}`" for item in record["git_state"]["status_short"])
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a descriptive version anchor for an optimization round.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--round-id", required=True)
    parser.add_argument("--strategy-id", required=True)
    parser.add_argument("--objective", required=True)
    parser.add_argument("--descriptor", required=True)
    parser.add_argument("--repo", default=".")
    parser.add_argument("--create-git-tag", action="store_true")
    args = parser.parse_args()
    record = create_round_version_record(
        run_dir=args.run_dir,
        round_id=args.round_id,
        strategy_id=args.strategy_id,
        objective=args.objective,
        descriptor=args.descriptor,
        repo=args.repo,
        create_git_tag=args.create_git_tag,
    )
    print(yaml.safe_dump(record, sort_keys=False))


if __name__ == "__main__":
    main()
