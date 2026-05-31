from __future__ import annotations

import argparse
import json
from pathlib import Path

from modular_pipeline.components import COMPONENTS, available_components, parse_mask, run_modular_pipeline
from modular_pipeline.tracking import inspect_outputs, write_state


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the analysis through a maskable modular pipeline.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list-components", help="Print component names and groups.")
    list_parser.add_argument("--verbose", action="store_true")

    inspect_parser = subparsers.add_parser("inspect", help="Inspect an existing output directory for resumable artifacts.")
    inspect_parser.add_argument("--outputs", required=True)
    inspect_parser.add_argument("--write-state", action="store_true", help="Write modular_pipeline_state.json into the output directory.")
    inspect_parser.add_argument("--json", action="store_true", help="Print the full JSON state artifact.")

    run_parser = subparsers.add_parser("run", help="Run the modular pipeline.")
    run_parser.add_argument("--summary", required=True)
    run_parser.add_argument("--inputs", required=True)
    run_parser.add_argument("--outputs", required=True)
    run_parser.add_argument("--max-events", type=int)
    run_parser.add_argument("--unblind-observed-significance", action="store_true")
    run_parser.add_argument(
        "--mask",
        default="",
        help="Comma-separated component or group names to skip, e.g. plots,report or stats.",
    )
    run_parser.add_argument(
        "--strict-mask",
        action="store_true",
        help="Fail if an unmasked component is missing context from an earlier masked component.",
    )

    args = parser.parse_args()
    if args.command == "list-components":
        for component in available_components():
            groups = ",".join(component["groups"])
            if args.verbose:
                print(
                    f"{component['name']}: groups=[{groups}] "
                    f"requires={component['requires']} provides={component['provides']} "
                    f"- {component['description']}"
                )
            else:
                print(f"{component['name']}\t{groups}")
        return

    if args.command == "inspect":
        outputs = Path(args.outputs)
        state = write_state(outputs, COMPONENTS) if args.write_state else inspect_outputs(outputs, COMPONENTS)
        if args.json:
            print(json.dumps(state, indent=2, sort_keys=True))
            return
        print(f"outputs: {state['outputs']}")
        print("ready entrypoints from artifacts:")
        for name in state["ready_entrypoints_from_artifacts"]:
            print(f"  {name}")
        print("component artifact status:")
        for component in state["components"]:
            print(f"  {component['name']}: {component['artifact_status']}")
        return

    run_modular_pipeline(
        summary=Path(args.summary),
        inputs=Path(args.inputs),
        outputs=Path(args.outputs),
        max_events=args.max_events,
        unblind_observed_significance=args.unblind_observed_significance,
        mask=parse_mask(args.mask),
        strict_mask=args.strict_mask,
    )


if __name__ == "__main__":
    main()
