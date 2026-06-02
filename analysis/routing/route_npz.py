from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from analysis.common import ensure_dir
from analysis.routing.config import load_routing_config
from analysis.routing.router import route_categories


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {name: np.asarray(payload[name]) for name in payload.files}


def main() -> None:
    parser = argparse.ArgumentParser(description="Reroute materialized observables using a routing config.")
    parser.add_argument("--events", required=True, type=Path, help="Input NPZ observable table.")
    parser.add_argument("--routing-config", required=True, type=Path, help="Repo-relative or absolute routing YAML.")
    parser.add_argument("--outputs", required=True, type=Path, help="Output directory for routed assignments.")
    args = parser.parse_args()

    observables = _load_npz(args.events)
    result = route_categories(observables, load_routing_config(args.routing_config))
    outputs = ensure_dir(args.outputs)
    np.savez_compressed(
        outputs / "routed_categories.npz",
        assigned_category=result.assigned_category,
        assignment_reason=result.assignment_reason,
        assignment_blocked=result.assignment_blocked.astype(int),
        category_label=result.category_label,
    )
    print(outputs / "routed_categories.npz")


if __name__ == "__main__":
    main()
