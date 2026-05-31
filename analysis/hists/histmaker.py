from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any

import awkward as ak
import numpy as np

from analysis.common import ensure_dir, read_json, write_json
from analysis.io.readers import REQUIRED_BRANCHES, iterate_events, io_diagnostics
from analysis.objects.jets import build_jets
from analysis.objects.photons import build_photons
from analysis.selections.engine import CATEGORY_ORDER, assign_categories, sideband_mask, signal_window_mask


CUT_STEPS = ["all_events", "two_photons", "pt_fraction", "mass_window", "categorized"]


def event_weights(batch: ak.Array, sample: dict) -> np.ndarray:
    size = len(batch["eventNumber"])
    if sample["kind"] == "data":
        return np.ones(size, dtype=float)
    norm = sample.get("norm_factor", 1.0)
    weights = norm * np.asarray(batch["mcWeight"], dtype=float)
    for branch in ("ScaleFactor_PILEUP", "ScaleFactor_PHOTON", "ScaleFactor_JVT"):
        if branch in batch.fields:
            weights *= np.asarray(batch[branch], dtype=float)
    return weights


def compute_norm_factor(sample: dict) -> float:
    denom = sample["sumw"]
    if not denom:
        return 1.0
    return sample["xsec_pb"] * sample["k_factor"] * sample["filter_eff"] * sample["lumi_fb"] * 1000.0 / denom


def process_sample(sample: dict, cfg: dict, max_events: int | None = None, cache_dir: Path | None = None) -> dict:
    sample = dict(sample)
    if sample["kind"] != "data":
        sample["norm_factor"] = compute_norm_factor(sample)
    output = {
        "sample_id": sample["sample_id"],
        "process_key": sample["process_key"],
        "kind": sample["kind"],
        "analysis_role": sample["analysis_role"],
        "cutflow": {step: {"weighted": 0.0, "unweighted": 0} for step in CUT_STEPS},
        "events": defaultdict(list),
        "io_diagnostics": io_diagnostics(sample["files"], tree_name=cfg["tree_name"]),
    }

    for batch in iterate_events(sample["files"], cfg["tree_name"], REQUIRED_BRANCHES, max_events=max_events):
        weights_all = event_weights(batch, sample)
        output["cutflow"]["all_events"]["weighted"] += float(np.sum(weights_all))
        output["cutflow"]["all_events"]["unweighted"] += int(len(weights_all))

        photons = build_photons(batch, cfg["photon_selection"])
        mask_two = photons["mask_has_two"]
        if np.any(mask_two):
            output["cutflow"]["two_photons"]["weighted"] += float(np.sum(weights_all[mask_two]))
            output["cutflow"]["two_photons"]["unweighted"] += int(np.sum(mask_two))
        selected_weights = weights_all[mask_two]
        selected_event_numbers = np.asarray(batch["eventNumber"], dtype=np.int64)[mask_two]
        selected_run_numbers = np.asarray(batch["runNumber"], dtype=np.int64)[mask_two]

        pt_mask = photons["pt_fraction_mask"]
        if np.any(pt_mask):
            output["cutflow"]["pt_fraction"]["weighted"] += float(np.sum(selected_weights[pt_mask]))
            output["cutflow"]["pt_fraction"]["unweighted"] += int(np.sum(pt_mask))

        mass = photons["diphoton_mass"]
        mass_mask = pt_mask & (mass >= cfg["fit_mass_range_gev"][0]) & (mass <= cfg["fit_mass_range_gev"][1])
        if np.any(mass_mask):
            output["cutflow"]["mass_window"]["weighted"] += float(np.sum(selected_weights[mass_mask]))
            output["cutflow"]["mass_window"]["unweighted"] += int(np.sum(mass_mask))

        jets = build_jets(batch, cfg["jet_selection"], mask_two)
        features = {
            "diphoton_mass": mass[mass_mask],
            "ptt": photons["ptt"][mass_mask],
            "delta_r": photons["delta_r"][mass_mask],
            "lead_pt": photons["lead_pt"][mass_mask],
            "sublead_pt": photons["sublead_pt"][mass_mask],
            "lead_eta": photons["lead_eta"][mass_mask],
            "sublead_eta": photons["sublead_eta"][mass_mask],
            "photon_multiplicity": photons["photon_multiplicity"][mass_mask],
            "n_jets": jets["n_jets"][mass_mask],
            "mjj": jets["mjj"][mass_mask],
            "delta_eta_jj": jets["delta_eta_jj"][mass_mask],
        }
        categories = assign_categories(features)
        categorized_mask = categories != "unassigned"
        if np.any(categorized_mask):
            output["cutflow"]["categorized"]["weighted"] += float(np.sum(selected_weights[mass_mask][categorized_mask]))
            output["cutflow"]["categorized"]["unweighted"] += int(np.sum(categorized_mask))

        for category in CATEGORY_ORDER:
            category_mask = categorized_mask & (categories == category)
            if not np.any(category_mask):
                continue
            output["events"]["category"].append(np.repeat(category, np.sum(category_mask)))
            output["events"]["mgg"].append(features["diphoton_mass"][category_mask])
            output["events"]["ptt"].append(features["ptt"][category_mask])
            output["events"]["delta_r"].append(features["delta_r"][category_mask])
            output["events"]["lead_pt"].append(features["lead_pt"][category_mask])
            output["events"]["sublead_pt"].append(features["sublead_pt"][category_mask])
            output["events"]["weight"].append(selected_weights[mass_mask][category_mask])
            output["events"]["is_sideband"].append(sideband_mask(features["diphoton_mass"][category_mask]))
            output["events"]["is_signal_window"].append(signal_window_mask(features["diphoton_mass"][category_mask]))
            output["events"]["event_number"].append(selected_event_numbers[mass_mask][category_mask])
            output["events"]["run_number"].append(selected_run_numbers[mass_mask][category_mask])
            output["events"]["lead_eta"].append(features["lead_eta"][category_mask])
            output["events"]["sublead_eta"].append(features["sublead_eta"][category_mask])
            output["events"]["photon_multiplicity"].append(features["photon_multiplicity"][category_mask])
            output["events"]["n_jets"].append(features["n_jets"][category_mask])
            output["events"]["mjj"].append(features["mjj"][category_mask])
            output["events"]["delta_eta_jj"].append(features["delta_eta_jj"][category_mask])

    flat_events = {}
    for key, chunks in output["events"].items():
        if not chunks:
            flat_events[key] = np.array([])
            continue
        if key == "category":
            flat_events[key] = np.concatenate(chunks).astype(str)
        else:
            flat_events[key] = np.concatenate(chunks)
    output["events"] = flat_events
    output["object_summary"] = {
        "selected_entries": int(len(flat_events.get("mgg", []))),
        "avg_good_jets": float(np.mean(flat_events["n_jets"])) if len(flat_events.get("n_jets", [])) else 0.0,
        "avg_photon_multiplicity": float(np.mean(flat_events["photon_multiplicity"])) if len(flat_events.get("photon_multiplicity", [])) else 0.0,
    }
    if cache_dir is not None:
        ensure_dir(cache_dir)
        cache_path = cache_dir / f"{sample['sample_id']}.npz"
        np.savez_compressed(cache_path, **flat_events)
        output["cache_path"] = str(cache_path)
    return output


def build_templates(processed_samples: list[dict], cfg: dict, out_dir: Path) -> dict:
    edges = np.arange(
        cfg["fit_mass_range_gev"][0],
        cfg["fit_mass_range_gev"][1] + cfg["histogramming"]["mass_bin_width_gev"],
        cfg["histogramming"]["mass_bin_width_gev"],
    )
    templates: dict[str, Any] = {"edges": edges.tolist(), "samples": {}}
    for sample in processed_samples:
        sample_templates = {}
        if len(sample["events"].get("mgg", [])) == 0:
            templates["samples"][sample["sample_id"]] = sample_templates
            continue
        for category in CATEGORY_ORDER:
            mask = sample["events"]["category"] == category
            masses = sample["events"]["mgg"][mask]
            weights = sample["events"]["weight"][mask]
            counts, _ = np.histogram(masses, bins=edges, weights=weights)
            variances, _ = np.histogram(masses, bins=edges, weights=weights**2)
            sample_templates[category] = {
                "counts": counts.tolist(),
                "sumw2": variances.tolist(),
                "yield": float(np.sum(weights)),
                "entries": int(np.sum(mask)),
            }
        templates["samples"][sample["sample_id"]] = sample_templates
    write_json(templates, out_dir / "templates.json")
    return templates


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--sample", required=True)
    parser.add_argument("--out-dir", default="outputs/hists")
    parser.add_argument("--max-events", type=int)
    args = parser.parse_args()
    registry = read_json(args.registry)
    summary = read_json(args.summary)
    cfg = summary["runtime_defaults"]
    sample = next(item for item in registry if item["sample_id"] == args.sample)
    processed = process_sample(sample, cfg, args.max_events, Path("outputs/cache"))
    build_templates([processed], cfg, Path(args.out_dir))
    print(processed["cutflow"]["categorized"]["unweighted"])


if __name__ == "__main__":
    main()
