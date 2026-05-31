from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from analysis.common import read_json, write_json


DEFAULT_CORRELATION_GROUPS = {
    "luminosity": "global_lumi",
    "photon_energy_scale": "global_photon_energy_scale",
    "photon_identification_efficiency": "global_photon_id",
    "background_modeling": "per_category_background_modeling",
    "mc_statistical": "per_sample_stat",
}


def build_systematics(registry: list[dict], summary: dict, outputs: Path) -> dict[str, Any]:
    nominal_samples = [sample for sample in registry if sample.get("is_nominal")]
    variation_map = {}
    for sample in nominal_samples:
        if sample["kind"] == "data":
            continue
        variation_map[sample["sample_id"]] = [
            candidate["sample_id"]
            for candidate in registry
            if candidate["process_key"] == sample["process_key"] and not candidate["is_nominal"]
        ]

    nuisances = [
        {
            "name": "luminosity",
            "type": "norm",
            "affected_processes": [sample["sample_id"] for sample in nominal_samples if sample["kind"] != "data"],
            "affected_regions": summary["inventory"]["fit_ids"],
            "correlation_group": DEFAULT_CORRELATION_GROUPS["luminosity"],
            "magnitude": 0.02,
        },
        {
            "name": "photon_energy_scale",
            "type": "norm",
            "affected_processes": [sample["sample_id"] for sample in nominal_samples if sample["kind"] == "signal"],
            "affected_regions": summary["inventory"]["fit_ids"],
            "correlation_group": DEFAULT_CORRELATION_GROUPS["photon_energy_scale"],
            "mode": "placeholder",
        },
        {
            "name": "photon_identification_efficiency",
            "type": "norm",
            "affected_processes": [sample["sample_id"] for sample in nominal_samples if sample["kind"] == "signal"],
            "affected_regions": summary["inventory"]["fit_ids"],
            "correlation_group": DEFAULT_CORRELATION_GROUPS["photon_identification_efficiency"],
            "mode": "placeholder",
        },
        {
            "name": "background_modeling",
            "type": "shape",
            "affected_processes": ["continuum_background"],
            "affected_regions": summary["inventory"]["fit_ids"],
            "correlation_group": DEFAULT_CORRELATION_GROUPS["background_modeling"],
            "source": "spurious_signal_and_background_choice",
        },
        {
            "name": "mc_statistical",
            "type": "stat",
            "affected_processes": [sample["sample_id"] for sample in nominal_samples if sample["kind"] != "data"],
            "affected_regions": summary["inventory"]["fit_ids"],
            "correlation_group": DEFAULT_CORRELATION_GROUPS["mc_statistical"],
            "mode": "per_bin_effective_statistical_uncertainty",
        },
    ]

    systematics = {
        "status": "ok",
        "mode": "stat_only_plus_placeholder_norms",
        "stat_only_fallback": True,
        "nuisances": nuisances,
        "notes": [
            "Only statistical uncertainties and placeholder normalization nuisances are implemented because full variation templates are not provided in the open-data inputs.",
            "Alternative generator/background samples are mapped as variation inputs and are excluded from central yields.",
        ],
    }
    provenance = {
        "status": "ok",
        "assumptions": [
            "No explicit up/down variation templates were supplied in the input directories.",
            "Continuum-background systematics are represented through the spurious-signal/model-choice artifacts rather than a full template morphing model.",
        ],
        "missing_inputs": ["shape variation templates", "full experimental response nuisance payloads"],
        "nominal_central_yield_policy": "only nominal samples contribute to central yields",
    }
    mapping = {
        "status": "ok",
        "nominal_to_variations": variation_map,
        "nominal_sample_ids": [sample["sample_id"] for sample in nominal_samples if sample["kind"] != "data"],
    }

    write_json(systematics, outputs / "systematics.json")
    write_json(provenance, outputs / "systematics_provenance.json")
    write_json(mapping, outputs / "systematics_sample_mapping.json")
    return {
        "systematics": systematics,
        "provenance": provenance,
        "mapping": mapping,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--outputs", default="outputs")
    args = parser.parse_args()
    build_systematics(read_json(args.registry), read_json(args.summary), Path(args.outputs))
    print("ok")


if __name__ == "__main__":
    main()
