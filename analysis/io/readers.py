from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterator

import awkward as ak
import uproot

from analysis.common import read_json, write_json
from analysis.runtime import check_pyroot


REQUIRED_BRANCHES = [
    "eventNumber",
    "runNumber",
    "mcWeight",
    "ScaleFactor_PILEUP",
    "ScaleFactor_PHOTON",
    "ScaleFactor_JVT",
    "photon_n",
    "photon_pt",
    "photon_eta",
    "photon_phi",
    "photon_e",
    "photon_isTightID",
    "photon_isTightIso",
    "jet_n",
    "jet_pt",
    "jet_eta",
    "jet_phi",
    "jet_e",
]


def iterate_events(files: list[str] | list[Path], tree_name: str, branches: list[str], max_events: int | None = None, step_size: str = "100 MB") -> Iterator[ak.Array]:
    seen = 0
    for file_path in files:
        with uproot.open(file_path) as handle:
            tree = handle[tree_name]
            for batch in tree.iterate(branches, step_size=step_size, library="ak"):
                if max_events is None:
                    yield batch
                    continue
                remaining = max_events - seen
                if remaining <= 0:
                    return
                if len(batch[branches[0]]) > remaining:
                    yield batch[:remaining]
                    return
                yield batch
                seen += len(batch[branches[0]])


def io_diagnostics(files: list[str] | list[Path], tree_name: str = "analysis") -> dict:
    field_inventory: set[str] = set()
    total_entries = 0
    for file_path in files:
        with uproot.open(file_path) as handle:
            tree = handle[tree_name]
            total_entries += int(tree.num_entries)
            field_inventory.update(tree.keys())
    return {
        "status": "ok",
        "tree_name": tree_name,
        "file_count": len(files),
        "event_count": total_entries,
        "fields": sorted(field_inventory),
        "pyroot": check_pyroot(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", required=True)
    parser.add_argument("--sample", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    registry = read_json(args.registry)
    sample = next(item for item in registry if item["sample_id"] == args.sample)
    diagnostics = io_diagnostics(sample["files"])
    write_json(diagnostics, args.out)
    print(diagnostics["event_count"])


if __name__ == "__main__":
    main()
