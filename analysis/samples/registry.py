from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

from analysis.common import list_root_files, read_json, write_json
from analysis.samples.metadata import build_metadata_rows, dsid_from_name, generator_from_descriptor, descriptor_from_name


NEGATIVE_SIGNAL_TOKENS = [
    "tautau",
    "mumu",
    "zz",
    "ww",
    "bb",
    "hzzinv",
    "hinv",
    "gamstargam",
    "zgam",
    "h_incl",
]


SIGNAL_PROCESS_PATTERNS = [
    ("ggh", ["ggh125"]),
    ("vbf", ["vbfh125", "vbf125_gammagamma", "vbf_hc_hyy"]),
    ("wmh", ["wmh125j", "wmh_fxfx_hyy"]),
    ("wph", ["wph125j", "wph_fxfx_hyy"]),
    ("zh", ["zh125j_hyy", "zh_fxfx_hyy", "zh_had_fxfx_hyy"]),
    ("ggzh", ["ggzh125_hgamgam", "ggzh_hyy", "ggzh_zhad_hyy"]),
    ("tth", ["tth125_gamgam", "tth_gamgam"]),
    ("thj", ["thjb125_4fl_gamgam"]),
    ("twh", ["twh125_yy"]),
]


BACKGROUND_PROCESS_PATTERNS = [
    ("prompt_diphoton", ["diphoton_myy"]),
    ("gamma_jet", ["gammajet"]),
    ("jet_jet", ["jetjet"]),
    ("ttgamma", ["ttgamma"]),
    ("z_to_ee_fake", ["eegammagamma", "eegamma"]),
    ("electroweak_yy", ["mumugammagamma", "tautaugammagamma", "nunugammagamma", "enugammagamma", "munugammagamma", "taunugammagamma"]),
]


def _has_target_decay(descriptor: str) -> bool:
    lowered = descriptor.lower()
    if any(token in lowered for token in NEGATIVE_SIGNAL_TOKENS):
        return False
    return any(token in lowered for token in ["gamgam", "gammagamma", "hyy", "hgamgam", "_yy"])


def classify_mc_descriptor(descriptor: str) -> tuple[str, str, str]:
    lowered = descriptor.lower()
    if _has_target_decay(lowered):
        for process_key, tokens in SIGNAL_PROCESS_PATTERNS:
            if any(token in lowered for token in tokens):
                return "signal", process_key, f"pp -> {process_key} -> H -> gamma gamma"
    for process_key, tokens in BACKGROUND_PROCESS_PATTERNS:
        if any(token in lowered for token in tokens):
            return "background", process_key, process_key.replace("_", " ")
    return "background", "other_background", descriptor


def analysis_role(kind: str, process_key: str) -> str:
    if kind == "data":
        return "data"
    if kind == "signal":
        return f"{kind}_nominal"
    if process_key == "prompt_diphoton":
        return "background_nominal"
    return "background_alternative"


def score_nominal_candidate(descriptor: str) -> tuple[int, int, int]:
    lowered = descriptor.lower()
    generator_priority = 0
    if "powheg" in lowered:
        generator_priority += 30
    if "pythia" in lowered or "py8" in lowered:
        generator_priority += 20
    if "amc" in lowered:
        generator_priority += 10
    if "herwig" in lowered or "h7" in lowered or "shower" in lowered:
        generator_priority -= 20
    decay_priority = 10 if any(token in lowered for token in ["hyy", "gamgam", "gammagamma"]) else 0
    aux_penalty = -50 if any(token in lowered for token in ["shw", "fix", "alternative"]) else 0
    return generator_priority, decay_priority, aux_penalty


def parse_mass_window(descriptor: str) -> tuple[float, float] | None:
    match = re.search(r"myy_(\d+)_([\dinf]+)", descriptor.lower())
    if not match:
        return None
    lo = float(match.group(1))
    hi = 1e9 if match.group(2) == "e" else float("inf") if match.group(2) == "inf" else float(match.group(2))
    return lo, hi


def choose_nominal_samples(samples: list[dict[str, Any]]) -> tuple[dict[str, list[str]], dict[str, list[str]], dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for sample in samples:
        if sample["kind"] == "data":
            continue
        grouped.setdefault(sample["process_key"], []).append(sample)

    selected: dict[str, list[str]] = {}
    alternatives: dict[str, list[str]] = {}
    diphoton_policy: dict[str, Any] = {
        "mass_range_relevance_status": "not_applicable",
        "nominal_background_template_policy": "not_applicable",
        "selected_nominal_background_template_sample": None,
        "selected_nominal_background_template_mass_window": None,
    }
    for process_key, candidates in grouped.items():
        if process_key == "prompt_diphoton":
            valid = []
            for candidate in candidates:
                window = parse_mass_window(candidate["descriptor"])
                if window and window[0] <= 105.0 and window[1] >= 160.0:
                    width = window[1] - window[0]
                    valid.append((width, candidate, window))
            valid.sort(key=lambda item: item[0])
            nominal = valid[0][1] if valid else max(candidates, key=lambda sample: score_nominal_candidate(sample["descriptor"]))
            selected[process_key] = [nominal["sample_id"]]
            alternatives[process_key] = [sample["sample_id"] for sample in candidates if sample["sample_id"] != nominal["sample_id"]]
            chosen_window = parse_mass_window(nominal["descriptor"])
            diphoton_policy = {
                "mass_range_relevance_status": "verified",
                "nominal_background_template_policy": "default_diphoton_minimum_window",
                "selected_nominal_background_template_sample": nominal["sample_id"],
                "selected_nominal_background_template_mass_window": list(chosen_window) if chosen_window else None,
            }
            continue
        if candidates[0]["kind"] == "background":
            selected[process_key] = []
            alternatives[process_key] = [sample["sample_id"] for sample in candidates]
            continue
        ranked = sorted(candidates, key=lambda sample: score_nominal_candidate(sample["descriptor"]), reverse=True)
        nominal = ranked[0]
        selected[process_key] = [nominal["sample_id"]]
        alternatives[process_key] = [sample["sample_id"] for sample in ranked[1:]]
    return selected, alternatives, diphoton_policy


def build_registry(inputs: Path, summary: dict[str, Any], target_lumi_fb: float) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    metadata_rows = {row["sample_id"]: row for row in build_metadata_rows(inputs)}
    registry: list[dict[str, Any]] = []
    for data_path in list_root_files(inputs / "data"):
        sample_id = data_path.name.removesuffix(".GamGam.root").replace("ODEO_FEB2025_v0_GamGam_", "")
        registry.append(
            {
                "sample_id": sample_id,
                "process_name": "observed data",
                "process_key": "data",
                "kind": "data",
                "analysis_role": "data",
                "is_nominal": True,
                "nominal_process_key": "data",
                "files": [str(data_path)],
                "generator": None,
                "simulation_config": None,
                "xsec_pb": None,
                "k_factor": None,
                "filter_eff": None,
                "sumw": None,
                "lumi_fb": None,
                "weight_expr": "1.0",
            }
        )
    for mc_path in list_root_files(inputs / "MC"):
        sample_id = dsid_from_name(mc_path)
        descriptor = descriptor_from_name(mc_path)
        generator, simulation = generator_from_descriptor(descriptor)
        kind, process_key, process_name = classify_mc_descriptor(descriptor)
        meta = metadata_rows[sample_id]
        registry.append(
            {
                "sample_id": sample_id,
                "process_name": process_name,
                "process_key": process_key,
                "kind": kind,
                "analysis_role": analysis_role(kind, process_key),
                "is_nominal": False,
                "nominal_process_key": process_key,
                "files": [str(mc_path)],
                "generator": generator,
                "simulation_config": simulation,
                "descriptor": descriptor,
                "xsec_pb": meta["xsec_pb"],
                "k_factor": meta["k_factor"],
                "filter_eff": meta["filter_eff"],
                "sumw": meta["sumw"],
                "lumi_fb": target_lumi_fb,
                "effective_lumi_fb": meta["effective_lumi_fb"],
                "weight_expr": "w_norm * mcWeight * ScaleFactor_PILEUP * ScaleFactor_PHOTON * ScaleFactor_JVT",
            }
        )
    selected, alternatives, diphoton_policy = choose_nominal_samples(registry)
    for sample in registry:
        sample["is_nominal"] = sample["kind"] == "data" or sample["sample_id"] in selected.get(sample["process_key"], [])
        if sample["kind"] == "signal":
            sample["analysis_role"] = "signal_nominal" if sample["is_nominal"] else "signal_alternative"
        elif sample["kind"] == "background" and sample["process_key"] == "prompt_diphoton":
            sample["analysis_role"] = "background_nominal" if sample["is_nominal"] else "background_alternative"
    process_roles = {
        "signal_processes": sorted({sample["process_key"] for sample in registry if sample["kind"] == "signal"}),
        "background_processes": sorted({sample["process_key"] for sample in registry if sample["kind"] == "background"}),
        "data_samples": [sample["sample_id"] for sample in registry if sample["kind"] == "data"],
        "selected_nominal_samples": selected,
        "alternative_samples": alternatives,
        "diphoton_template_policy": diphoton_policy,
        "analysis_objective": summary["analysis_objectives"][0]["target_process"],
    }
    return registry, process_roles


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--roles-out", default="outputs/report/mc_sample_selection.json")
    parser.add_argument("--target-lumi-fb", type=float, default=36.1)
    args = parser.parse_args()
    summary = read_json(args.summary)
    registry, process_roles = build_registry(Path(args.inputs), summary, args.target_lumi_fb)
    write_json(registry, args.out)
    status = "resolved" if process_roles["selected_nominal_samples"] else "blocked"
    process_roles["status"] = status
    process_roles["ambiguity_status"] = "resolved" if status == "resolved" else "blocked"
    process_roles["notes"] = [
        "Signal nominal samples are chosen by physics-token matching and generator preference.",
        "The prompt-diphoton nominal template is the smallest mass slice fully containing 105-160 GeV.",
    ]
    write_json(process_roles, args.roles_out)
    print(f"registry rows: {len(registry)}")


if __name__ == "__main__":
    main()
