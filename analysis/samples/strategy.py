from __future__ import annotations

import argparse
from pathlib import Path

from analysis.common import read_json, write_json


def build_strategy(registry: list[dict], summary: dict) -> tuple[dict, dict, dict]:
    nominal_signal = sorted(sample["sample_id"] for sample in registry if sample["analysis_role"] == "signal_nominal")
    nominal_bkg = sorted(sample["sample_id"] for sample in registry if sample["analysis_role"] == "background_nominal")
    classification = {
        "status": "ok",
        "data_samples": [sample["sample_id"] for sample in registry if sample["kind"] == "data"],
        "signal_samples": nominal_signal,
        "background_samples": nominal_bkg,
        "data_driven_backgrounds": [bkg["process_name"] for bkg in summary.get("background_processes", [])],
    }
    strategy = {
        "status": "ok",
        "analysis_target": summary["analysis_objectives"][0]["target_process"],
        "fit_background_model": "data_sideband_analytic",
        "spurious_signal_template_source": nominal_bkg,
        "notes": [
            "Continuum backgrounds are modeled with analytic functions fitted to the diphoton mass spectrum in data.",
            "Prompt-diphoton MC is retained as the nominal template source for smoothing and spurious-signal checks.",
        ],
    }
    constraint_map = {
        "status": "ok",
        "constraints": [],
        "overlap_policy": {
            "control_region": "CR_BKG_VALIDATION",
            "allow_overlap_with_signal_regions": True,
            "reason": "control region is implemented as sideband validation within the same selected event categories",
        },
    }
    return classification, strategy, constraint_map


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--classification-out", default="outputs/samples.classification.json")
    parser.add_argument("--strategy-out", default="outputs/background_modeling_strategy.json")
    parser.add_argument("--constraint-out", default="outputs/cr_sr_constraint_map.json")
    args = parser.parse_args()
    registry = read_json(args.registry)
    summary = read_json(args.summary)
    classification, strategy, constraint_map = build_strategy(registry, summary)
    write_json(classification, args.classification_out)
    write_json(strategy, args.strategy_out)
    write_json(constraint_map, args.constraint_out)
    print("ok")


if __name__ == "__main__":
    main()
