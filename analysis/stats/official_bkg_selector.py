"""Optional HHXYY bkgParamTool background-selection bridge.

This module intentionally does not replace the local background selector.  It
parses, prepares, or runs the official HHXYY spurious-signal selection so its
answer can be compared with, or explicitly substituted for, the local choice.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


DEFAULT_MODELS = ("ExpPoly2", "ExpPoly3", "Exponential", "Pow", "Bern3", "Bern4", "Bern5")
DEFAULT_SORTING = "Default"

RESULT_FIELDS = (
    "max_s_over_ds_percent",
    "max_one_sigma_over_ds_percent",
    "max_two_sigma_over_ds_percent",
    "max_s_over_ref_percent",
    "max_s",
    "s_at_125",
    "sref_at_125",
    "ds_at_125",
    "sigma_s_at_125",
    "npars",
    "chi2_over_ndof",
    "prob_chi2_percent",
)

OFFICIAL_TO_LOCAL_EQUIVALENT = {
    "Bern2": "bernstein1",
    "Bern3": "bernstein2",
    "Bern4": "bernstein3",
    "Bern5": "bernstein4",
    "Exponential": "exponential",
    "ExpPoly2": None,
    "ExpPoly3": None,
    "Pow": None,
}

OFFICIAL_CONFIG_TEMPLATE = """# Generated from the modular-pipeline HHXYY official background-selector bridge.
Verbosity: {verbosity}
PerPointPlots: {per_point_plots}
PlotResidualPull: {plot_residual_pull}
CoreFunction: False
SignalCurve: False

Blind:  False
BlindMin: 121
BlindMax: 129
NBinsChi2: 55

Observable: {observable}[105,160]
Observable.NBins:110
Observable.Unit: GeV

Dataset.File: {dataset_file}
Dataset.HistogramName: {dataset_histogram}
Dataset.IntegratedLuminosity: 1.
Dataset.Title: Template
Dataset.Rebin: 1

Signal.Norm: nSignal[1,-1000,1000]
Background.Norm: nBkg[1,0,1E6]

Signal.PDF.Expression:
Signal.PDF.Parameters:
Signal.PDF.Name: {signal_pdf_name}
Signal.PDF.Workspace: signalWS
Signal.PDF.File: {signal_pdf_file}
Signal.PDF.FixParameters: True

Background.PDFs: {models}

Bern3.Expression: RooBernstein({observable}, {{ a1[0,-100,100], a2[0,-100,100], a3[0,-100,100], 1 }})
Bern3.SetInitialValuesFromDataset: True
Bern4.Expression: RooBernstein({observable}, {{ a1[0,-100,100], a2[0,-100,100], a3[0,-100,100], a4[0,-100,100], 1 }})
Bern4.SetInitialValuesFromDataset: True
Bern5.Expression: RooBernstein({observable}, {{ a1[0,-100,100], a2[0,-100,100], a3[0,-100,100], a4[0,-100,100], a5[0,-100,100], 1 }})
Bern5.SetInitialValuesFromDataset: True

ExpPoly2.Expression: exp(({observable} - 100)/100*(a1 + a2*({observable} - 100)/100))
ExpPoly2.Parameters: a1[0,-100,100] a2[0,-100,100]
ExpPoly2.SetInitialValuesFromDataset: True
ExpPoly3.Expression: exp(({observable} - 100)/100*(a1 + a2*({observable} - 100)/100 + a3*({observable} - 100)/100*({observable} - 100)/100))
ExpPoly3.Parameters: a1[0,-100,100] a2[0,-100,100] a3[0,-100,100]
ExpPoly3.SetInitialValuesFromDataset: True

Exponential.Expression: exp(xi*{observable});
Exponential.Parameters: xi[0,-100,100]
Exponential.SetInitialValuesFromDataset: True

Pow.Expression: pow({observable},xi);
Pow.Parameters: xi[0,-100,100]
Pow.SetInitialValuesFromDataset: True

Scan.Var: mH
Scan.Unit: GeV
Scan.Step: 1.0
Scan.Min: 121
Scan.Max: 129

Selection.MaxSignalOverError: {max_signal_over_error}
Selection.MaxSignalOverRef: {max_signal_over_ref}
Selection.MaxOneSigmaSignalOverError: {max_one_sigma_signal_over_error}
Selection.MaxTwoSigmaSignalOverError: {max_two_sigma_signal_over_error}
Selection.MinChiSquarePvalue: {min_chi_square_pvalue}
Selection.IntegratedLuminosity: 1.
Selection.SortingOption: {sorting_option}

RefSignalYield: {ref_signal_yield}
RefSignalCrossSection:
RefSignalYieldVar.Name:
RefSignalYieldVar.Workspace:
RefSignalYieldVar.File:
RefSignalYield.IntegratedLuminosity: 1

Show.MaxSignalOverRef: True
Show.SpuriousSignal_125: True
Show.RefSignalYield_125: True
Show.delta_S_125: True
Show.sigma_S_125: True
Show.Total_Error: False
"""


@dataclass(frozen=True)
class SelectionThresholds:
    max_signal_over_error: float = 0.20
    max_signal_over_ref: float = 0.10
    max_one_sigma_signal_over_error: float = 0.20
    max_two_sigma_signal_over_error: float = 0.20
    min_chi_square_pvalue: float = 0.0


@dataclass
class Candidate:
    name: str
    category: str | None = None
    max_s_over_ds_percent: float | None = None
    max_one_sigma_over_ds_percent: float | None = None
    max_two_sigma_over_ds_percent: float | None = None
    max_s_over_ref_percent: float | None = None
    max_s: float | None = None
    s_at_125: float | None = None
    sref_at_125: float | None = None
    ds_at_125: float | None = None
    sigma_s_at_125: float | None = None
    npars: int | None = None
    chi2_over_ndof: float | None = None
    prob_chi2_percent: float | None = None
    official_selected: bool = False
    marker_pass: bool | None = None
    passes_official_thresholds: bool | None = None
    local_equivalent: str | None = None
    hhxyy_xml_model: str | None = None

    def enrich(self, thresholds: SelectionThresholds) -> "Candidate":
        self.local_equivalent = OFFICIAL_TO_LOCAL_EQUIVALENT.get(self.name)
        self.hhxyy_xml_model = self.name
        self.passes_official_thresholds = passes_thresholds(self, thresholds)
        return self


def _float_or_none(value: str | None) -> float | None:
    if value in (None, "", "-", "nan", "NaN"):
        return None
    try:
        parsed = float(value)
    except ValueError:
        return None
    if math.isnan(parsed):
        return None
    return parsed


def _int_or_none(value: str | None) -> int | None:
    parsed = _float_or_none(value)
    if parsed is None:
        return None
    return int(round(parsed))


def _fraction(percent: float | None) -> float | None:
    return None if percent is None else percent / 100.0


def _abs_fraction(percent: float | None) -> float | None:
    value = _fraction(percent)
    return None if value is None else abs(value)


def passes_thresholds(candidate: Candidate, thresholds: SelectionThresholds) -> bool:
    criteria = []
    values = (
        (candidate.max_s_over_ds_percent, thresholds.max_signal_over_error),
        (candidate.max_s_over_ref_percent, thresholds.max_signal_over_ref),
        (candidate.max_one_sigma_over_ds_percent, thresholds.max_one_sigma_signal_over_error),
        (candidate.max_two_sigma_over_ds_percent, thresholds.max_two_sigma_signal_over_error),
    )
    for observed_percent, limit in values:
        observed = _abs_fraction(observed_percent)
        if observed is not None:
            criteria.append(observed < limit)
    if not criteria:
        return False

    prob_chi2 = _fraction(candidate.prob_chi2_percent)
    chi2_ok = True if prob_chi2 is None else prob_chi2 > thresholds.min_chi_square_pvalue
    return any(criteria) and chi2_ok


def parse_result_line(line: str, category: str | None, thresholds: SelectionThresholds) -> Candidate | None:
    stripped = line.strip()
    if not stripped or stripped.startswith("#") or stripped.startswith("Name"):
        return None

    tokens = stripped.split()
    if not tokens:
        return None

    name = tokens[0]
    if name.startswith("=") or name in {"PASS", "FAIL"}:
        return None

    selected = "<==" in tokens and "Selected" in tokens
    marker_pass = None
    if "PASS" in tokens:
        marker_pass = True
    elif "FAIL" in tokens:
        marker_pass = False

    values = [token for token in tokens[1:] if token not in {"<==", "Selected", "PASS", "FAIL"}]
    parsed: dict[str, float | int | None] = {}
    for field, value in zip(RESULT_FIELDS, values):
        if field == "npars":
            parsed[field] = _int_or_none(value)
        else:
            parsed[field] = _float_or_none(value)

    candidate = Candidate(
        name=name,
        category=category,
        official_selected=selected,
        marker_pass=marker_pass,
        **parsed,
    )
    return candidate.enrich(thresholds)


def parse_results_file(path: Path, category: str | None = None, thresholds: SelectionThresholds | None = None) -> list[Candidate]:
    thresholds = thresholds or SelectionThresholds()
    candidates: list[Candidate] = []
    for line in path.read_text().splitlines():
        candidate = parse_result_line(line, category, thresholds)
        if candidate is not None:
            candidates.append(candidate)
    return candidates


def official_rank_key(candidate: Candidate, sorting_option: str = DEFAULT_SORTING) -> tuple[Any, ...]:
    passes = bool(candidate.passes_official_thresholds)
    tier = 0 if passes else 4
    npars = candidate.npars if candidate.npars is not None else 9999
    max_s_abs = abs(candidate.max_s) if candidate.max_s is not None else float("inf")

    if sorting_option in {"TieredDefault", "TieredTotalError"}:
        marker_tier = 0 if candidate.marker_pass is True else 4 if candidate.marker_pass is False else tier
        return (not passes, marker_tier, npars, max_s_abs, candidate.name)

    if sorting_option == "TotalError":
        total_proxy = abs(candidate.s_at_125 or 0.0) + abs(candidate.ds_at_125 or 0.0)
        ref = abs(candidate.sref_at_125 or 0.0) or 1.0
        return (not passes, total_proxy / ref, npars, max_s_abs, candidate.name)

    return (not passes, npars, max_s_abs, candidate.name)


def select_candidate(candidates: list[Candidate], sorting_option: str = DEFAULT_SORTING) -> tuple[Candidate | None, str]:
    marked = [candidate for candidate in candidates if candidate.official_selected]
    if marked:
        return marked[0], "results_txt_marker"
    if not candidates:
        return None, "no_candidates"
    return sorted(candidates, key=lambda candidate: official_rank_key(candidate, sorting_option))[0], "recomputed_official_ranking"


def category_sort_key(path: Path) -> tuple[int, str]:
    match = re.search(r"(\d+)(?:\.[^.]+)?$", path.name)
    if match:
        return int(match.group(1)), path.name
    return 999999, path.name


def parse_results_dir(
    bkg_model_dir: Path,
    thresholds: SelectionThresholds | None = None,
    sorting_option: str = DEFAULT_SORTING,
) -> dict[str, Any]:
    thresholds = thresholds or SelectionThresholds()
    categories: dict[str, Any] = {}
    for result_file in sorted(bkg_model_dir.glob("cat*/results.txt"), key=lambda path: category_sort_key(path.parent)):
        category = result_file.parent.name
        candidates = parse_results_file(result_file, category, thresholds)
        selected, selected_source = select_candidate(candidates, sorting_option)
        categories[category] = {
            "results_file": str(result_file),
            "selected_model": selected.name if selected else None,
            "selected_source": selected_source,
            "selected_local_equivalent": selected.local_equivalent if selected else None,
            "selected_hhxyy_xml_model": selected.hhxyy_xml_model if selected else None,
            "candidates": [asdict(candidate) for candidate in candidates],
        }

    return {
        "source": "HHXYY bkgParamTool official spurious-signal selector",
        "sorting_option": sorting_option,
        "thresholds": asdict(thresholds),
        "model_equivalents": OFFICIAL_TO_LOCAL_EQUIVALENT,
        "categories": categories,
    }


def write_csv(summary: dict[str, Any], csv_path: Path) -> None:
    rows = []
    for category, payload in summary["categories"].items():
        for candidate in payload["candidates"]:
            rows.append(
                {
                    "category": category,
                    "model": candidate["name"],
                    "official_selected": candidate["official_selected"],
                    "passes_official_thresholds": candidate["passes_official_thresholds"],
                    "local_equivalent": candidate["local_equivalent"],
                    "hhxyy_xml_model": candidate["hhxyy_xml_model"],
                    "max_s_over_ds_percent": candidate["max_s_over_ds_percent"],
                    "max_s_over_ref_percent": candidate["max_s_over_ref_percent"],
                    "max_s": candidate["max_s"],
                    "s_at_125": candidate["s_at_125"],
                    "sref_at_125": candidate["sref_at_125"],
                    "ds_at_125": candidate["ds_at_125"],
                    "sigma_s_at_125": candidate["sigma_s_at_125"],
                    "npars": candidate["npars"],
                    "chi2_over_ndof": candidate["chi2_over_ndof"],
                    "prob_chi2_percent": candidate["prob_chi2_percent"],
                }
            )

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else ["category", "model"])
        writer.writeheader()
        writer.writerows(rows)


def _load_signal_yields(path: Path) -> dict[str, float]:
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"Signal-yield YAML must be a mapping: {path}")
    signal = data.get("signal", data)
    if not isinstance(signal, dict):
        raise ValueError(f"Signal-yield YAML must contain a mapping under 'signal': {path}")
    return {str(key): float(value) for key, value in signal.items()}


def _category_id(category: str) -> int:
    match = re.search(r"(\d+)$", category)
    if not match:
        raise ValueError(f"Cannot determine numeric category id from {category!r}")
    return int(match.group(1))


def make_configs(
    output_dir: Path,
    dataset_file: Path,
    signal_pdf_file: Path,
    signal_yields_file: Path,
    models: list[str],
    categories: list[str] | None = None,
    sorting_option: str = DEFAULT_SORTING,
    thresholds: SelectionThresholds | None = None,
    verbosity: int = 1,
    per_point_plots: bool = True,
    plot_residual_pull: bool = True,
    dataset_histogram_template: str = "{cat}",
    observable_template: str = "atlas_invMass_gamgam_cat{cat}",
    signal_pdf_template: str = "signalPdf_cat{cat}",
) -> list[Path]:
    thresholds = thresholds or SelectionThresholds()
    signal_yields = _load_signal_yields(signal_yields_file)
    categories = categories or sorted(signal_yields, key=lambda item: (_category_id(item), item))

    output_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for category in categories:
        cat_id = _category_id(category)
        ref_signal_yield = signal_yields[category]
        text = OFFICIAL_CONFIG_TEMPLATE.format(
            verbosity=verbosity,
            per_point_plots=str(per_point_plots),
            plot_residual_pull=str(plot_residual_pull),
            observable=observable_template.format(cat=cat_id, category=category),
            dataset_file=dataset_file,
            dataset_histogram=dataset_histogram_template.format(cat=cat_id, category=category),
            signal_pdf_name=signal_pdf_template.format(cat=cat_id, category=category),
            signal_pdf_file=signal_pdf_file,
            models=" ".join(models),
            ref_signal_yield=ref_signal_yield,
            sorting_option=sorting_option,
            **asdict(thresholds),
        )
        config_path = output_dir / f"spurious_cat{cat_id}.config"
        config_path.write_text(text)
        written.append(config_path)
    return written


def run_check_bkg_model(check_bkg_model: Path, config_dir: Path, output_dir: Path, categories: list[str] | None = None) -> None:
    if not check_bkg_model.exists():
        raise FileNotFoundError(f"checkBkgModel executable not found: {check_bkg_model}")

    config_paths = sorted(config_dir.glob("spurious_cat*.config"), key=category_sort_key)
    if categories:
        wanted = {f"spurious_cat{_category_id(category)}.config" for category in categories}
        config_paths = [path for path in config_paths if path.name in wanted]
    if not config_paths:
        raise FileNotFoundError(f"No spurious_cat*.config files found in {config_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    for config_path in config_paths:
        cat_name = config_path.stem.replace("spurious_", "")
        cat_out = output_dir / cat_name
        cat_out.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [str(check_bkg_model), str(config_path), str(cat_out), "spuriousSig"],
            check=True,
            cwd=output_dir,
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    parse = subparsers.add_parser("parse", help="Parse official bkgParamTool results.txt files.")
    parse.add_argument("--bkg-model-dir", type=Path, required=True)
    parse.add_argument("--out", type=Path, required=True)
    parse.add_argument("--csv", type=Path)
    parse.add_argument("--sorting-option", default=DEFAULT_SORTING)

    configs = subparsers.add_parser("make-configs", help="Generate official spurious-signal configs.")
    configs.add_argument("--out-dir", type=Path, required=True)
    configs.add_argument("--dataset-file", type=Path, required=True)
    configs.add_argument("--signal-pdf-file", type=Path, required=True)
    configs.add_argument("--signal-yields", type=Path, required=True)
    configs.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    configs.add_argument("--categories", nargs="+")
    configs.add_argument("--sorting-option", default=DEFAULT_SORTING)
    configs.add_argument("--dataset-histogram-template", default="{cat}")
    configs.add_argument("--observable-template", default="atlas_invMass_gamgam_cat{cat}")
    configs.add_argument("--signal-pdf-template", default="signalPdf_cat{cat}")

    run = subparsers.add_parser("run", help="Run checkBkgModel on generated configs.")
    run.add_argument("--check-bkg-model", type=Path, required=True)
    run.add_argument("--config-dir", type=Path, required=True)
    run.add_argument("--out-dir", type=Path, required=True)
    run.add_argument("--categories", nargs="+")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "parse":
        summary = parse_results_dir(args.bkg_model_dir, sorting_option=args.sorting_option)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        if args.csv:
            write_csv(summary, args.csv)
        return 0

    if args.command == "make-configs":
        written = make_configs(
            output_dir=args.out_dir,
            dataset_file=args.dataset_file,
            signal_pdf_file=args.signal_pdf_file,
            signal_yields_file=args.signal_yields,
            models=args.models,
            categories=args.categories,
            sorting_option=args.sorting_option,
            dataset_histogram_template=args.dataset_histogram_template,
            observable_template=args.observable_template,
            signal_pdf_template=args.signal_pdf_template,
        )
        for path in written:
            print(path)
        return 0

    if args.command == "run":
        run_check_bkg_model(args.check_bkg_model, args.config_dir, args.out_dir, args.categories)
        return 0

    parser.error(f"Unhandled command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
