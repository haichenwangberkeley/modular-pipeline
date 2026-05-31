from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Any

import uproot

from analysis.common import ensure_dir, list_root_files, write_json


META_BRANCHES = [
    "num_events",
    "sum_of_weights",
    "sum_of_weights_squared",
    "xsec",
    "filteff",
    "kfac",
    "channelNumber",
]


def dsid_from_name(path: Path) -> str | None:
    match = re.search(r"_mc_(\d+)\.", path.name)
    return match.group(1) if match else None


def descriptor_from_name(path: Path) -> str:
    name = path.name.removesuffix(".GamGam.root")
    return name.split("_mc_", 1)[1].split(".", 1)[1]


def generator_from_descriptor(descriptor: str) -> tuple[str, str]:
    tokens = descriptor.split("_")
    generator = tokens[0]
    simulation = "_".join(tokens[1:]) if len(tokens) > 1 else descriptor
    return generator, simulation


def read_root_metadata(path: Path) -> dict[str, Any]:
    with uproot.open(path) as handle:
        tree = handle["analysis"]
        arrays = tree.arrays(META_BRANCHES, entry_stop=1, library="np")
        values = {branch: float(arrays[branch][0]) for branch in META_BRANCHES if branch in arrays}
        values["entries"] = int(tree.num_entries)
    return values


def build_metadata_rows(inputs: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in list_root_files(inputs / "MC"):
        dsid = dsid_from_name(path)
        descriptor = descriptor_from_name(path)
        generator, simulation = generator_from_descriptor(descriptor)
        meta = read_root_metadata(path)
        denom = meta["xsec"] * meta["kfac"] * meta["filteff"] * 1000.0
        effective_lumi_fb = meta["sum_of_weights"] / denom if denom > 0 else None
        rows.append(
            {
                "sample_id": dsid,
                "dsid": dsid,
                "file": str(path),
                "filename": path.name,
                "descriptor": descriptor,
                "generator": generator,
                "simulation_config": simulation,
                "xsec_pb": meta["xsec"],
                "k_factor": meta["kfac"],
                "filter_eff": meta["filteff"],
                "sumw": meta["sum_of_weights"],
                "sumw2": meta["sum_of_weights_squared"],
                "num_events": int(meta["num_events"]),
                "entries": meta["entries"],
                "effective_lumi_fb": effective_lumi_fb,
            }
        )
    return rows


def write_metadata_csv(rows: list[dict[str, Any]], path: Path) -> Path:
    ensure_dir(path.parent)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def write_metadata_resolution(rows: list[dict[str, Any]], outputs: Path) -> None:
    write_json(
        {
            "status": "ok",
            "column_mapping": {
                "xsec_pb": "xsec",
                "k_factor": "kfac",
                "filter_eff": "filteff",
                "sumw": "sum_of_weights",
            },
            "row_count": len(rows),
            "source": "ROOT branch metadata reconstructed in-task because official metadata.csv was not provided",
        },
        outputs / "normalization" / "metadata_resolution.json",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", required=True)
    parser.add_argument("--csv-out", default="skills/metadata.csv")
    parser.add_argument("--resolution-out", default="outputs/normalization/metadata_resolution.json")
    args = parser.parse_args()
    rows = build_metadata_rows(Path(args.inputs))
    write_metadata_csv(rows, Path(args.csv_out))
    write_metadata_resolution(rows, Path(args.resolution_out).parents[1])
    print(f"metadata rows: {len(rows)}")


if __name__ == "__main__":
    main()
