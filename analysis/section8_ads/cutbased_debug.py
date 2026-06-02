from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analysis.common import ensure_dir, read_json, stable_hash, utcnow_iso, write_json, write_text
from analysis.config.load_summary import normalize_summary
from analysis.config.versions import apply_analysis_version
from analysis.samples.registry import build_registry
from analysis.section8_ads.categories import BDT_DEPENDENT_CATEGORIES, NON_BDT_CATEGORIES, assign_categories
from analysis.section8_ads.classifiers import score_samples
from analysis.section8_ads.pipeline import FIT_RANGE, _process_sample
from analysis.selections.engine import section8_category_id


ASSIGNMENT_FULL31 = "current_full31_assignment"
ASSIGNMENT_CUTBASED = "cutbased_no_bdt_assignment"
SIGNAL_WINDOW = (120.0, 130.0)
MASS_BINS = np.linspace(FIT_RANGE[0], FIT_RANGE[1], int(FIT_RANGE[1] - FIT_RANGE[0]) + 1)
SUMMARY_VARIABLES = [
    "m_gammagamma",
    "lead_pt",
    "sublead_pt",
    "lead_pt_over_mgg",
    "sublead_pt_over_mgg",
    "lead_eta",
    "sublead_eta",
    "pT_gammagamma",
    "pTt_gammagamma",
    "N_jets_25",
    "N_jets_30",
    "N_jets_25_jvt_diagnostic",
    "N_jets_30_jvt_diagnostic",
    "N_central_jets_25",
    "N_forward_jets_25",
    "N_btag_25",
    "N_lep",
    "m_ll",
    "pT_lepton_plus_MET",
    "MET",
    "MET_significance",
    "leading_jet_pT_30",
    "m_jj_30",
    "abs_delta_eta_jj_30",
    "VBF_centrality",
]
FOCUS_CATEGORIES = [
    "ggH_1J_Med",
    "ggH_2J_Med",
    "ggH_0J_Cen",
    "ggH_0J_Fwd",
    "jet_BSM",
    "VH_MET_Low",
    "VH_MET_High",
]


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> Path:
    ensure_dir(path.parent)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})
    return path


def _event_modulo_selector(modulo: int, remainder: int):
    def select(batch: Any) -> np.ndarray:
        event_number = np.asarray(batch["eventNumber"], dtype=np.int64)
        return np.remainder(event_number, modulo) == remainder

    return select


def _sample_group(sample: dict[str, Any]) -> str:
    if sample["kind"] == "data":
        return "data"
    if sample["kind"] == "signal":
        return str(sample["process_key"])
    if sample["process_key"] == "prompt_diphoton":
        return "prompt_diphoton"
    return str(sample["process_key"])


def _processing_samples(registry: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected = []
    for sample in registry:
        if sample["kind"] == "data":
            selected.append(sample)
            continue
        if sample["analysis_role"] == "signal_nominal":
            selected.append(sample)
            continue
        if sample["analysis_role"] == "background_nominal" and sample["process_key"] == "prompt_diphoton":
            selected.append(sample)
    return selected


def _load_boundaries(runtime: dict[str, Any]) -> dict[str, list[float]] | None:
    section8 = runtime.get("section8_ads", {})
    boundary_file = section8.get("boundary_file")
    if not boundary_file:
        return None
    path = Path(boundary_file)
    if not path.exists():
        return None
    payload = read_json(path)
    return payload.get("selected_boundaries")


def _training_report_path(runtime: dict[str, Any]) -> Path | None:
    artifact_dir = runtime.get("section8_ads", {}).get("bdt_artifacts_dir")
    if not artifact_dir:
        return None
    path = Path(artifact_dir) / "classifier_training_report.json"
    return path if path.exists() else None


def _ensure_bdt_arrays(arrays: dict[str, np.ndarray], fill: float = np.nan) -> dict[str, np.ndarray]:
    n_events = len(arrays.get("event_number", []))
    for name in ("BDT_ttH", "BDT_VH", "BDT_VBF"):
        if name not in arrays:
            arrays[name] = np.full(n_events, fill, dtype=float)
    return arrays


def _category_ids(labels: np.ndarray) -> np.ndarray:
    ids = []
    known_labels = set(NON_BDT_CATEGORIES) | set(BDT_DEPENDENT_CATEGORIES)
    for label in labels.astype(str):
        ids.append(section8_category_id(label) if label in known_labels else label)
    return np.asarray(ids, dtype=object)


def _cutbased_assignment(arrays: dict[str, np.ndarray], boundaries: dict[str, list[float]] | None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cut_arrays = dict(arrays)
    n_events = len(cut_arrays.get("event_number", []))
    for score_name in ("BDT_ttH", "BDT_VH", "BDT_VBF"):
        cut_arrays[score_name] = np.full(n_events, -1.0e9, dtype=float)
    assigned, reasons, blocked = assign_categories(cut_arrays, boundaries)
    return _category_ids(assigned), reasons, blocked


def _full31_assignment(arrays: dict[str, np.ndarray], boundaries: dict[str, list[float]] | None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    assigned, reasons, blocked = assign_categories(arrays, boundaries)
    return _category_ids(assigned), reasons, blocked


def _finite_summary(values: np.ndarray, weights: np.ndarray | None = None) -> dict[str, Any]:
    arr = np.asarray(values, dtype=float)
    mask = np.isfinite(arr)
    arr = arr[mask]
    if weights is not None:
        w = np.asarray(weights, dtype=float)[mask]
    else:
        w = np.ones(len(arr), dtype=float)
    quantile_weights = np.clip(w, 0.0, None)
    if len(arr) == 0:
        return {
            "n": 0,
            "weighted_yield": 0.0,
            "min": "",
            "max": "",
            "mean": "",
            "p10": "",
            "p50": "",
            "p90": "",
        }
    return {
        "n": int(len(arr)),
        "weighted_yield": float(np.sum(w)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "mean": float(np.average(arr, weights=w)) if np.sum(w) else float(np.mean(arr)),
        "p10": float(_weighted_quantile(arr, quantile_weights, 0.10)),
        "p50": float(_weighted_quantile(arr, quantile_weights, 0.50)),
        "p90": float(_weighted_quantile(arr, quantile_weights, 0.90)),
    }


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    if len(values) == 0:
        return float("nan")
    weights = np.asarray(weights, dtype=float)
    if not np.any(weights > 0):
        return float(np.quantile(values, quantile))
    order = np.argsort(values)
    sorted_values = values[order]
    sorted_weights = weights[order]
    cumulative = np.cumsum(sorted_weights)
    threshold = quantile * cumulative[-1]
    return float(sorted_values[np.searchsorted(cumulative, threshold, side="left")])


def _aggregate_cutflows(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    aggregate: dict[str, dict[str, dict[str, float]]] = defaultdict(lambda: defaultdict(lambda: {"unweighted": 0, "weighted": 0.0}))
    sample_rows = []
    for record in records:
        sample = record["sample"]
        group = _sample_group(sample)
        for step, payload in record["processed"]["cutflow"].items():
            unweighted = int(payload.get("after", 0))
            weighted = float(payload.get("weighted_after", unweighted))
            aggregate[group][step]["unweighted"] += unweighted
            aggregate[group][step]["weighted"] += weighted
            sample_rows.append(
                {
                    "sample_id": sample["sample_id"],
                    "process": group,
                    "kind": sample["kind"],
                    "analysis_role": sample["analysis_role"],
                    "step": step,
                    "unweighted_events": unweighted,
                    "normalized_yield_36fb": weighted,
                    "status": payload.get("status", ""),
                    "notes": payload.get("notes", ""),
                }
            )
    aggregate_rows = []
    for group, steps in sorted(aggregate.items()):
        for step, payload in steps.items():
            aggregate_rows.append(
                {
                    "process": group,
                    "step": step,
                    "unweighted_events": int(payload["unweighted"]),
                    "normalized_yield_36fb": float(payload["weighted"]),
                }
            )
    return aggregate_rows, sample_rows


def _category_yield_rows(records: list[dict[str, Any]], assignment_mode: str, process_filter: str) -> list[dict[str, Any]]:
    rows = []
    cut_ids = [section8_category_id(category) for category in NON_BDT_CATEGORIES]
    grouped: dict[tuple[str, str], dict[str, list[np.ndarray] | float | int]] = {}
    for record in records:
        sample = record["sample"]
        include = False
        if process_filter == "data":
            include = sample["kind"] == "data"
        elif process_filter == "signal":
            include = sample["kind"] == "signal" and sample["analysis_role"] == "signal_nominal"
        elif process_filter == "prompt_diphoton":
            include = sample["process_key"] == "prompt_diphoton" and sample["analysis_role"] == "background_nominal"
        if not include:
            continue
        arrays = record["processed"]["arrays"]
        assignments = record["assignments"][assignment_mode]
        weights = np.asarray(arrays.get("weight", []), dtype=float)
        masses = np.asarray(arrays.get("m_gammagamma", []), dtype=float)
        process = _sample_group(sample)
        for category in cut_ids:
            mask = assignments == category
            if not np.any(mask):
                continue
            key = (process, category)
            entry = grouped.setdefault(key, {"masses": [], "weights": [], "n": 0, "yield": 0.0})
            entry["masses"].append(masses[mask])
            entry["weights"].append(weights[mask])
            entry["n"] += int(np.sum(mask))
            entry["yield"] += float(np.sum(weights[mask]))
    for (process, category), entry in sorted(grouped.items()):
        masses = np.concatenate(entry["masses"]) if entry["masses"] else np.array([])
        weights = np.concatenate(entry["weights"]) if entry["weights"] else np.array([])
        summary = _finite_summary(masses, weights)
        rows.append(
            {
                "assignment_mode": assignment_mode,
                "process": process,
                "category": category,
                "unweighted_events": int(entry["n"]),
                "normalized_yield_36fb": float(entry["yield"]),
                "mgg_mean": summary["mean"],
                "mgg_p10": summary["p10"],
                "mgg_p50": summary["p50"],
                "mgg_p90": summary["p90"],
            }
        )
    return rows


def _assignment_audit_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for record in records:
        sample = record["sample"]
        arrays = record["processed"]["arrays"]
        weights = np.asarray(arrays.get("weight", []), dtype=float)
        process = _sample_group(sample)
        for mode, assignments in record["assignments"].items():
            for category in sorted(set(assignments.astype(str).tolist())):
                mask = assignments == category
                rows.append(
                    {
                        "row_type": "assignment_yield",
                        "assignment_mode": mode,
                        "sample_id": sample["sample_id"],
                        "process": process,
                        "kind": sample["kind"],
                        "from_category": "",
                        "to_category": category,
                        "unweighted_events": int(np.sum(mask)),
                        "normalized_yield_36fb": float(np.sum(weights[mask])),
                    }
                )
        full = record["assignments"][ASSIGNMENT_FULL31]
        cut = record["assignments"][ASSIGNMENT_CUTBASED]
        for from_category in sorted(set(full.astype(str).tolist())):
            from_mask = full == from_category
            for to_category in sorted(set(cut[from_mask].astype(str).tolist())):
                mask = from_mask & (cut == to_category)
                if not np.any(mask):
                    continue
                rows.append(
                    {
                        "row_type": "migration",
                        "assignment_mode": f"{ASSIGNMENT_FULL31}_to_{ASSIGNMENT_CUTBASED}",
                        "sample_id": sample["sample_id"],
                        "process": process,
                        "kind": sample["kind"],
                        "from_category": from_category,
                        "to_category": to_category,
                        "unweighted_events": int(np.sum(mask)),
                        "normalized_yield_36fb": float(np.sum(weights[mask])),
                    }
                )
    return rows


def _variable_summary_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    cut_ids = [section8_category_id(category) for category in NON_BDT_CATEGORIES]
    grouped: dict[tuple[str, str, str, str], dict[str, list[np.ndarray]]] = defaultdict(lambda: {"values": [], "weights": []})
    for record in records:
        sample = record["sample"]
        process = _sample_group(sample)
        arrays = record["processed"]["arrays"]
        weights = np.asarray(arrays.get("weight", []), dtype=float)
        for mode in (ASSIGNMENT_FULL31, ASSIGNMENT_CUTBASED):
            assignments = record["assignments"][mode]
            for category in cut_ids:
                category_mask = assignments == category
                if not np.any(category_mask):
                    continue
                for variable in SUMMARY_VARIABLES:
                    if variable not in arrays:
                        continue
                    grouped[(mode, process, category, variable)]["values"].append(np.asarray(arrays[variable])[category_mask])
                    grouped[(mode, process, category, variable)]["weights"].append(weights[category_mask])
    for (mode, process, category, variable), payload in sorted(grouped.items()):
        values = np.concatenate(payload["values"]) if payload["values"] else np.array([])
        weights = np.concatenate(payload["weights"]) if payload["weights"] else np.array([])
        summary = _finite_summary(values, weights)
        rows.append(
            {
                "assignment_mode": mode,
                "process": process,
                "category": category,
                "variable": variable,
                "finite_entries": summary["n"],
                "weighted_yield": summary["weighted_yield"],
                "min": summary["min"],
                "max": summary["max"],
                "mean": summary["mean"],
                "p10": summary["p10"],
                "p50": summary["p50"],
                "p90": summary["p90"],
            }
        )
    return rows


def _plot_category_myy(records: list[dict[str, Any]], category: str, outputs: Path) -> dict[str, str]:
    plot_dir = ensure_dir(outputs / "plots" / "m_yy_by_category")
    data_values: list[np.ndarray] = []
    prompt_values: list[np.ndarray] = []
    prompt_weights: list[np.ndarray] = []
    signal_values: list[np.ndarray] = []
    signal_weights: list[np.ndarray] = []
    for record in records:
        sample = record["sample"]
        arrays = record["processed"]["arrays"]
        assignments = record["assignments"][ASSIGNMENT_CUTBASED]
        mask = assignments == category
        if not np.any(mask):
            continue
        masses = np.asarray(arrays["m_gammagamma"], dtype=float)[mask]
        weights = np.asarray(arrays["weight"], dtype=float)[mask]
        if sample["kind"] == "data":
            data_values.append(masses)
        elif sample["analysis_role"] == "background_nominal" and sample["process_key"] == "prompt_diphoton":
            prompt_values.append(masses)
            prompt_weights.append(weights)
        elif sample["analysis_role"] == "signal_nominal":
            signal_values.append(masses)
            signal_weights.append(weights)
    data = np.concatenate(data_values) if data_values else np.array([])
    prompt = np.concatenate(prompt_values) if prompt_values else np.array([])
    prompt_w = np.concatenate(prompt_weights) if prompt_weights else np.array([])
    signal = np.concatenate(signal_values) if signal_values else np.array([])
    signal_w = np.concatenate(signal_weights) if signal_weights else np.array([])

    data_counts, _ = np.histogram(data, bins=MASS_BINS)
    prompt_counts, _ = np.histogram(prompt, bins=MASS_BINS, weights=prompt_w)
    prompt_sumw2, _ = np.histogram(prompt, bins=MASS_BINS, weights=prompt_w**2)
    signal_counts, _ = np.histogram(signal, bins=MASS_BINS, weights=signal_w)
    centers = 0.5 * (MASS_BINS[:-1] + MASS_BINS[1:])
    widths = np.diff(MASS_BINS)
    prompt_unc = np.sqrt(prompt_sumw2)

    fig, (ax, ratio_ax) = plt.subplots(
        2,
        1,
        figsize=(7.5, 6.5),
        gridspec_kw={"height_ratios": [3.0, 1.0], "hspace": 0.08},
        sharex=True,
        layout="constrained",
    )
    ax.errorbar(centers, data_counts, yerr=np.sqrt(data_counts), fmt="o", color="black", label="Data", markersize=3)
    ax.step(MASS_BINS[:-1], prompt_counts, where="post", color="#276FBF", label="Prompt diphoton MC")
    ax.fill_between(
        centers,
        prompt_counts - prompt_unc,
        prompt_counts + prompt_unc,
        step="mid",
        color="#276FBF",
        alpha=0.20,
        label="Prompt MC stat. unc.",
    )
    if np.any(signal_counts):
        ax.step(MASS_BINS[:-1], signal_counts, where="post", color="#D1495B", label="Total Higgs signal MC")
    ax.axvspan(SIGNAL_WINDOW[0], SIGNAL_WINDOW[1], color="gray", alpha=0.12, label="Signal window")
    ax.set_ylabel("Events / 1 GeV")
    ax.set_title(f"{category}: cut-based no-BDT assignment")
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.25)

    ratio = np.full_like(prompt_counts, np.nan, dtype=float)
    ratio_err = np.full_like(prompt_counts, np.nan, dtype=float)
    mc_rel_unc = np.full_like(prompt_counts, np.nan, dtype=float)
    good = prompt_counts > 0
    ratio[good] = data_counts[good] / prompt_counts[good]
    ratio_err[good] = np.sqrt(data_counts[good]) / prompt_counts[good]
    mc_rel_unc[good] = prompt_unc[good] / prompt_counts[good]
    ratio_ax.axhline(1.0, color="black", linewidth=1)
    ratio_ax.fill_between(
        centers[good],
        1.0 - mc_rel_unc[good],
        1.0 + mc_rel_unc[good],
        step="mid",
        color="#276FBF",
        alpha=0.20,
    )
    ratio_ax.errorbar(centers[good], ratio[good], yerr=ratio_err[good], fmt="o", color="black", markersize=3)
    ratio_ax.set_ylim(0.0, 2.5)
    ratio_ax.set_ylabel("Data / MC")
    ratio_ax.set_xlabel(r"$m_{\gamma\gamma}$ [GeV]")
    ratio_ax.grid(alpha=0.25)

    png_path = plot_dir / f"{category}_myy.png"
    pdf_path = plot_dir / f"{category}_myy.pdf"
    fig.savefig(png_path, dpi=150)
    fig.savefig(pdf_path)
    plt.close(fig)
    return {"png": str(png_path), "pdf": str(pdf_path)}


def _plot_focus_before_after(records: list[dict[str, Any]], category: str, outputs: Path) -> dict[str, str]:
    plot_dir = ensure_dir(outputs / "plots" / "suspicious_cut_stages")
    inclusive: list[np.ndarray] = []
    selected: list[np.ndarray] = []
    for record in records:
        sample = record["sample"]
        if sample["kind"] != "data":
            continue
        arrays = record["processed"]["arrays"]
        masses = np.asarray(arrays.get("m_gammagamma", []), dtype=float)
        inclusive.append(masses)
        assignments = record["assignments"][ASSIGNMENT_CUTBASED]
        selected.append(masses[assignments == category])
    inc = np.concatenate(inclusive) if inclusive else np.array([])
    cat = np.concatenate(selected) if selected else np.array([])
    inc_counts, _ = np.histogram(inc, bins=MASS_BINS)
    cat_counts, _ = np.histogram(cat, bins=MASS_BINS)
    centers = 0.5 * (MASS_BINS[:-1] + MASS_BINS[1:])

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.step(centers, inc_counts, where="mid", color="#444444", label="Inclusive baseline data")
    ax.step(centers, cat_counts, where="mid", color="#D1495B", label=f"{category} data")
    ax.axvspan(SIGNAL_WINDOW[0], SIGNAL_WINDOW[1], color="gray", alpha=0.12, label="Signal window")
    ax.set_xlabel(r"$m_{\gamma\gamma}$ [GeV]")
    ax.set_ylabel("Events / 1 GeV")
    ax.set_title(f"{category}: inclusive baseline vs assigned category")
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    png_path = plot_dir / f"{category}_inclusive_vs_category_myy.png"
    pdf_path = plot_dir / f"{category}_inclusive_vs_category_myy.pdf"
    fig.savefig(png_path, dpi=150)
    fig.savefig(pdf_path)
    plt.close(fig)
    return {"png": str(png_path), "pdf": str(pdf_path)}


def _shape_audit_markdown(
    data_rows: list[dict[str, Any]],
    prompt_rows: list[dict[str, Any]],
    audit_rows: list[dict[str, Any]],
    plot_manifest: dict[str, Any],
    run_metadata: dict[str, Any],
) -> str:
    data_by_category = {row["category"]: row for row in data_rows if row["assignment_mode"] == ASSIGNMENT_CUTBASED}
    prompt_by_category = {row["category"]: row for row in prompt_rows if row["assignment_mode"] == ASSIGNMENT_CUTBASED}
    lines = [
        "# Phase 2 m_yy Shape Audit",
        "",
        "This is a diagnostic-only cut-based validation artifact. It uses the configured trigger policy and does not apply any additional hidden selection changes.",
        "",
        "## Run Metadata",
        "",
        f"- Created UTC: `{run_metadata['created_utc']}`",
        f"- Summary path: `{run_metadata['summary_path']}`",
        f"- Inputs path: `{run_metadata['inputs_path']}`",
        f"- Subset policy: `{run_metadata['subset_policy']}`",
        f"- Trigger policy: `{run_metadata['trigger_policy']}`",
        f"- Event selector: `eventNumber % {run_metadata['sample_modulo']} == {run_metadata['sample_remainder']}`",
        f"- Max selected input rows per sample: `{run_metadata['max_selected_events_per_sample']}`",
        f"- Plot binning: `105-160 GeV, 1 GeV bins`",
        f"- Plot normalization: `data unweighted counts; MC normalized to target luminosity via existing weights`",
        f"- Ratio uncertainty: `data Poisson errors and prompt-MC sumw2 band`",
        "",
        "## Focus Category Summary",
        "",
        "| Category | Data events | Data p50 | Prompt events | Prompt yield | Prompt p50 | Flag |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for category in FOCUS_CATEGORIES:
        data = data_by_category.get(category, {})
        prompt = prompt_by_category.get(category, {})
        data_p50 = data.get("mgg_p50", "")
        prompt_p50 = prompt.get("mgg_p50", "")
        flags = []
        if isinstance(data_p50, float) and (data_p50 < 120.0 or data_p50 > 140.0):
            flags.append("data_p50_outside_120_140")
        if isinstance(prompt_p50, float) and (prompt_p50 < 120.0 or prompt_p50 > 140.0):
            flags.append("prompt_p50_outside_120_140")
        if int(data.get("unweighted_events", 0) or 0) < 10:
            flags.append("low_data_stat")
        lines.append(
            f"| `{category}` | {data.get('unweighted_events', 0)} | {data_p50} | "
            f"{prompt.get('unweighted_events', 0)} | {prompt.get('normalized_yield_36fb', 0.0)} | {prompt_p50} | "
            f"{', '.join(flags) if flags else 'none'} |"
        )

    migration_totals: dict[tuple[str, str], int] = defaultdict(int)
    for row in audit_rows:
        if row["row_type"] != "migration" or row["from_category"] == row["to_category"] or row["process"] != "data":
            continue
        migration_totals[(row["from_category"], row["to_category"])] += int(row["unweighted_events"])
    migrations = sorted(migration_totals.items(), key=lambda item: item[1], reverse=True)[:15]
    lines.extend(
        [
            "",
            "## Largest Data Migrations",
            "",
            "| From full31 | To cutbased no-BDT | Events |",
            "|---|---|---:|",
        ]
    )
    for (from_category, to_category), events in migrations:
        lines.append(f"| `{from_category}` | `{to_category}` | {events} |")
    lines.extend(
        [
            "",
            "## Plot Manifest",
            "",
            f"- Per-category plots: `{len(plot_manifest['m_yy_by_category'])}` categories.",
            f"- Suspicious-cut stage plots: `{len(plot_manifest['suspicious_cut_stages'])}` categories.",
            "",
            "## Interpretation Guardrails",
            "",
            "- This subset is for debugging selection logic and shape pathologies, not final yields.",
            "- BDT-dependent categories are deliberately skipped in the cut-based assignment mode.",
            "- Any fix must be justified by these diagnostics and recorded in `DEBUG_REPORT.md`.",
        ]
    )
    return "\n".join(lines) + "\n"


def run_cutbased_debug(
    *,
    summary: Path,
    inputs: Path,
    outputs: Path,
    analysis_version: str | None,
    section8_bdt_artifacts: Path | None,
    sample_modulo: int,
    sample_remainder: int,
    max_selected_events: int | None,
    trigger_policy: str,
) -> dict[str, Any]:
    outputs = ensure_dir(outputs)
    source_summary = read_json(summary)
    normalized, errors = normalize_summary(source_summary, summary)
    if errors:
        raise RuntimeError(f"Summary validation failed: {errors}")
    runtime = apply_analysis_version(
        normalized["runtime_defaults"],
        version_name=analysis_version,
        section8_bdt_artifacts=section8_bdt_artifacts,
    )
    normalized["runtime_defaults"] = runtime
    normalized["config_hash"] = stable_hash(normalized)
    registry, process_roles = build_registry(inputs, normalized, runtime["central_mc_lumi_fb"])
    selected_samples = _processing_samples(registry)
    selector = _event_modulo_selector(sample_modulo, sample_remainder)

    processed_records = []
    for sample in selected_samples:
        processed = _process_sample(
            sample,
            max_events=max_selected_events,
            event_selector=selector,
            trigger_policy=trigger_policy,
        )
        processed_records.append({"sample": sample, "processed": processed})

    training_report_path = _training_report_path(runtime)
    scoring_status = "not_available"
    if training_report_path is not None:
        try:
            score_samples([record["processed"] for record in processed_records], read_json(training_report_path))
            scoring_status = "scored_from_configured_artifacts"
        except Exception as exc:  # Diagnostic mode should still expose cut-only behavior.
            scoring_status = f"failed:{type(exc).__name__}:{exc}"

    boundaries = _load_boundaries(runtime)
    for record in processed_records:
        arrays = _ensure_bdt_arrays(record["processed"]["arrays"])
        full_assigned, full_reasons, full_blocked = _full31_assignment(arrays, boundaries)
        cut_assigned, cut_reasons, cut_blocked = _cutbased_assignment(arrays, boundaries)
        record["assignments"] = {
            ASSIGNMENT_FULL31: full_assigned,
            ASSIGNMENT_CUTBASED: cut_assigned,
        }
        record["assignment_reasons"] = {
            ASSIGNMENT_FULL31: full_reasons,
            ASSIGNMENT_CUTBASED: cut_reasons,
        }
        record["assignment_blocked"] = {
            ASSIGNMENT_FULL31: full_blocked,
            ASSIGNMENT_CUTBASED: cut_blocked,
        }

    run_metadata = {
        "created_utc": utcnow_iso(),
        "summary_path": str(summary),
        "inputs_path": str(inputs),
        "outputs_path": str(outputs),
        "analysis_version": analysis_version,
        "config_hash": normalized["config_hash"],
        "sample_modulo": sample_modulo,
        "sample_remainder": sample_remainder,
        "max_selected_events_per_sample": max_selected_events,
        "trigger_policy": trigger_policy,
        "subset_policy": "deterministic_event_number_modulo_with_per_sample_cap",
        "selected_sample_count": len(selected_samples),
        "selected_samples": [sample["sample_id"] for sample in selected_samples],
        "training_report_path": str(training_report_path) if training_report_path else None,
        "bdt_scoring_status": scoring_status,
        "cutbased_categories": [section8_category_id(category) for category in NON_BDT_CATEGORIES],
        "excluded_bdt_categories": [section8_category_id(category) for category in BDT_DEPENDENT_CATEGORIES],
        "process_roles": process_roles,
    }
    write_json(run_metadata, outputs / "run_metadata.json")

    cutflow_rows, sample_cutflow_rows = _aggregate_cutflows(processed_records)
    data_cutflow_rows = [row for row in cutflow_rows if row["process"] == "data"]
    _write_csv(
        outputs / "inclusive_data_cutflow.csv",
        data_cutflow_rows,
        ["process", "step", "unweighted_events", "normalized_yield_36fb"],
    )
    _write_csv(
        outputs / "inclusive_process_cutflow.csv",
        cutflow_rows,
        ["process", "step", "unweighted_events", "normalized_yield_36fb"],
    )
    _write_csv(
        outputs / "sample_cutflow_debug.csv",
        sample_cutflow_rows,
        ["sample_id", "process", "kind", "analysis_role", "step", "unweighted_events", "normalized_yield_36fb", "status", "notes"],
    )

    data_yields = _category_yield_rows(processed_records, ASSIGNMENT_CUTBASED, "data")
    signal_yields = _category_yield_rows(processed_records, ASSIGNMENT_CUTBASED, "signal")
    prompt_yields = _category_yield_rows(processed_records, ASSIGNMENT_CUTBASED, "prompt_diphoton")
    yield_fields = [
        "assignment_mode",
        "process",
        "category",
        "unweighted_events",
        "normalized_yield_36fb",
        "mgg_mean",
        "mgg_p10",
        "mgg_p50",
        "mgg_p90",
    ]
    _write_csv(outputs / "category_data_yields.csv", data_yields, yield_fields)
    _write_csv(outputs / "category_mc_signal_yields.csv", signal_yields, yield_fields)
    _write_csv(outputs / "category_prompt_diphoton_yields.csv", prompt_yields, yield_fields)

    audit_rows = _assignment_audit_rows(processed_records)
    _write_csv(
        outputs / "category_assignment_audit.csv",
        audit_rows,
        [
            "row_type",
            "assignment_mode",
            "sample_id",
            "process",
            "kind",
            "from_category",
            "to_category",
            "unweighted_events",
            "normalized_yield_36fb",
        ],
    )

    variable_rows = _variable_summary_rows(processed_records)
    _write_csv(
        outputs / "variable_summaries.csv",
        variable_rows,
        [
            "assignment_mode",
            "process",
            "category",
            "variable",
            "finite_entries",
            "weighted_yield",
            "min",
            "max",
            "mean",
            "p10",
            "p50",
            "p90",
        ],
    )

    plot_manifest = {"m_yy_by_category": {}, "suspicious_cut_stages": {}}
    for category in [section8_category_id(item) for item in NON_BDT_CATEGORIES]:
        plot_manifest["m_yy_by_category"][category] = _plot_category_myy(processed_records, category, outputs)
    for category in FOCUS_CATEGORIES:
        plot_manifest["suspicious_cut_stages"][category] = _plot_focus_before_after(processed_records, category, outputs)
    write_json(plot_manifest, outputs / "plots" / "manifest.json")

    write_text(
        _shape_audit_markdown(data_yields, prompt_yields, audit_rows, plot_manifest, run_metadata),
        outputs / "m_yy_shape_audit.md",
    )

    status = {
        "status": "ok",
        "outputs": str(outputs),
        "artifacts": [
            "run_metadata.json",
            "inclusive_data_cutflow.csv",
            "inclusive_process_cutflow.csv",
            "sample_cutflow_debug.csv",
            "category_data_yields.csv",
            "category_mc_signal_yields.csv",
            "category_prompt_diphoton_yields.csv",
            "category_assignment_audit.csv",
            "variable_summaries.csv",
            "m_yy_shape_audit.md",
            "plots/manifest.json",
        ],
    }
    write_json(status, outputs / "phase2_status.json")
    return status


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Section 8 cut-based debugging diagnostics on a deterministic subset.")
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--inputs", required=True, type=Path)
    parser.add_argument("--outputs", required=True, type=Path)
    parser.add_argument("--analysis-version", default="round2_section8_bdt")
    parser.add_argument("--section8-bdt-artifacts", type=Path)
    parser.add_argument("--sample-modulo", type=int, default=10)
    parser.add_argument("--sample-remainder", type=int, default=0)
    parser.add_argument("--max-selected-events", type=int, default=100000)
    parser.add_argument("--trigger-policy", choices=["input_preselected", "trigP", "none"], default="input_preselected")
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.sample_modulo <= 0:
        raise SystemExit("--sample-modulo must be positive")
    if not 0 <= args.sample_remainder < args.sample_modulo:
        raise SystemExit("--sample-remainder must satisfy 0 <= remainder < modulo")
    run_cutbased_debug(
        summary=args.summary,
        inputs=args.inputs,
        outputs=args.outputs,
        analysis_version=args.analysis_version,
        section8_bdt_artifacts=args.section8_bdt_artifacts,
        sample_modulo=args.sample_modulo,
        sample_remainder=args.sample_remainder,
        max_selected_events=args.max_selected_events,
        trigger_policy=args.trigger_policy,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
