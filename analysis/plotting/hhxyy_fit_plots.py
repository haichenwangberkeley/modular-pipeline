from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from analysis.common import read_json, utcnow_iso, write_json
from analysis.plotting.blinded_regions import generate_plots
from analysis.report.artifacts import write_background_template_smoothing_artifacts, write_verification_status
from analysis.report.make_report import build_report
from analysis.stats.fit import FIT_ID, run_fit
from analysis.stats.significance import run_significance


def _resolve_cache_path(path_value: str, outputs: Path) -> Path:
    path = Path(path_value)
    candidates = [path]
    if not path.is_absolute():
        candidates.append(outputs / path)
        candidates.append(outputs.parent / path)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Cached processed-sample array not found: {path_value}")


def load_processed_samples_from_artifacts(outputs: Path) -> list[dict[str, Any]]:
    manifest = read_json(outputs / "hists" / "processed_samples.json")
    processed_samples: list[dict[str, Any]] = []
    for item in manifest["samples"]:
        cache_path = _resolve_cache_path(str(item["cache_path"]), outputs)
        arrays = np.load(cache_path, allow_pickle=True)
        sample = dict(item)
        sample["cache_path"] = str(cache_path)
        sample["events"] = {key: arrays[key] for key in arrays.files}
        processed_samples.append(sample)
    return processed_samples


def regenerate_hhxyy_style_plots(outputs: Path, *, update_report: bool = True) -> dict[str, Any]:
    summary = read_json(outputs / "summary.normalized.json")
    registry = read_json(outputs / "samples.registry.json")
    cutflow_table = read_json(outputs / "report" / "cutflow_table.json")
    processed_samples = load_processed_samples_from_artifacts(outputs)

    fit_context = run_fit(processed_samples, registry, summary, outputs)
    run_significance(fit_context, summary, outputs)
    plot_manifest = generate_plots(processed_samples, summary, fit_context, outputs, cutflow_table)
    write_background_template_smoothing_artifacts(fit_context, outputs)
    verification = write_verification_status(plot_manifest, fit_context, outputs)

    if update_report:
        build_report(summary, outputs, outputs.parent / "reports")

    regeneration = {
        "status": "ok",
        "timestamp_utc": utcnow_iso(),
        "fit_id": FIT_ID,
        "source": "existing processed-sample and fit artifacts",
        "plot_semantics": plot_manifest.get("fit_visualization_semantics", {}),
        "verification_status": verification["status"],
        "plot_manifest": str(outputs / "report" / "plots" / "manifest.json"),
    }
    write_json(regeneration, outputs / "report" / "hhxyy_style_plot_regeneration.json")
    return regeneration


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Regenerate HHXYY-style analytic-PDF fit plots from an existing pipeline output directory."
    )
    parser.add_argument("--outputs", required=True, help="Existing pipeline output directory to update.")
    parser.add_argument("--no-report", action="store_true", help="Do not rebuild report.md/final_analysis_report.md.")
    args = parser.parse_args()

    result = regenerate_hhxyy_style_plots(Path(args.outputs), update_report=not args.no_report)
    print(result["plot_manifest"])


if __name__ == "__main__":
    main()
