from __future__ import annotations

import argparse
from pathlib import Path

from analysis.common import write_json
from analysis.config.load_summary import normalize_summary
from analysis.preflight import run_preflight
from analysis.runtime import write_runtime_recovery


def bootstrap(summary: Path, outputs: Path) -> None:
    normalized, errors = normalize_summary(__import__("analysis.common").common.read_json(summary), summary)
    write_json(normalized, outputs / "summary.normalized.json")
    write_runtime_recovery(outputs / "report" / "runtime_recovery.json")
    if errors:
        raise SystemExit(1)


def run_pipeline(
    summary: Path,
    inputs: Path,
    outputs: Path,
    max_events: int | None,
    unblind_observed_significance: bool = False,
) -> None:
    from analysis.pipeline import run_all_stages

    run_all_stages(
        summary=summary,
        inputs=inputs,
        outputs=outputs,
        max_events=max_events,
        unblind_observed_significance=unblind_observed_significance,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    bootstrap_parser = subparsers.add_parser("bootstrap")
    bootstrap_parser.add_argument("--summary", required=True)
    bootstrap_parser.add_argument("--outputs", required=True)

    preflight_parser = subparsers.add_parser("preflight")
    preflight_parser.add_argument("--summary", required=True)
    preflight_parser.add_argument("--inputs", required=True)
    preflight_parser.add_argument("--outputs", required=True)

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--summary", required=True)
    run_parser.add_argument("--inputs", required=True)
    run_parser.add_argument("--outputs", required=True)
    run_parser.add_argument("--max-events", type=int)
    run_parser.add_argument("--unblind-observed-significance", action="store_true")

    args = parser.parse_args()
    if args.command == "bootstrap":
        bootstrap(Path(args.summary), Path(args.outputs))
    elif args.command == "preflight":
        run_preflight(Path(args.summary), Path(args.inputs), Path(args.outputs))
    else:
        run_pipeline(
            Path(args.summary),
            Path(args.inputs),
            Path(args.outputs),
            args.max_events,
            args.unblind_observed_significance,
        )


if __name__ == "__main__":
    main()
