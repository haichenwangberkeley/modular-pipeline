#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any


CATEGORY_ORDER = [
    "central_high_ptt",
    "central_low_ptt",
    "rest_high_ptt",
    "rest_low_ptt",
    "two_jet_vbf_enriched",
]


def fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return value
    try:
        number = float(value)
    except Exception:
        return str(value)
    if not math.isfinite(number):
        return str(number)
    return f"{number:.12g}"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def read_workspace_vars(path: Path, workspace_name: str) -> dict[str, dict[str, Any]]:
    import ROOT

    root_file = ROOT.TFile.Open(str(path))
    if not root_file or root_file.IsZombie():
        raise RuntimeError(f"Could not open {path}")
    try:
        workspace = root_file.Get(workspace_name)
        if not workspace:
            raise RuntimeError(f"Workspace {workspace_name!r} not found in {path}")
        values: dict[str, dict[str, Any]] = {}
        for var in workspace.allVars():
            values[var.GetName()] = {
                "value": float(var.getVal()),
                "error": float(var.getError()) if hasattr(var, "getError") else None,
                "constant": bool(var.isConstant()) if hasattr(var, "isConstant") else None,
            }
        return values
    finally:
        root_file.Close()


def read_quickfit(path: Path) -> dict[str, Any]:
    import ROOT

    root_file = ROOT.TFile.Open(str(path))
    if not root_file or root_file.IsZombie():
        raise RuntimeError(f"Could not open {path}")
    try:
        tree = root_file.Get("nllscan")
        if not tree or tree.GetEntries() < 1:
            raise RuntimeError(f"Missing nllscan in {path}")
        tree.GetEntry(0)
        output: dict[str, Any] = {
            "tree": {branch.GetName(): float(getattr(tree, branch.GetName())) for branch in tree.GetListOfBranches()},
            "float": {},
            "const": {},
            "status": None,
            "cov_qual": None,
        }
        fit_result = root_file.Get("fitResult")
        if fit_result:
            output["status"] = int(fit_result.status())
            output["cov_qual"] = int(fit_result.covQual())
            for collection_name, target in (("floatParsFinal", "float"), ("constPars", "const")):
                collection = getattr(fit_result, collection_name)()
                for idx in range(collection.getSize()):
                    var = collection.at(idx)
                    output[target][var.GetName()] = {
                        "value": float(var.getVal()),
                        "error": float(var.getError()) if hasattr(var, "getError") else None,
                        "constant": target == "const",
                    }
        return output
    finally:
        root_file.Close()


def value(payload: dict[str, Any] | None, key: str) -> Any:
    if payload is None:
        return None
    return payload.get(key)


def delta(a: Any, b: Any) -> float | None:
    if a is None or b is None:
        return None
    return float(a) - float(b)


def rel_delta(a: Any, b: Any) -> float | None:
    if a is None or b is None or float(b) == 0.0:
        return None
    return (float(a) - float(b)) / float(b)


def qf_param(quickfit: dict[str, Any], name: str) -> dict[str, Any] | None:
    return quickfit["float"].get(name) or quickfit["const"].get(name)


def qf_workspace_value(workspace: dict[str, dict[str, Any]], name: str) -> dict[str, Any] | None:
    return workspace.get(name)


def local_var(workspace: dict[str, dict[str, Any]], name: str) -> dict[str, Any] | None:
    return workspace.get(name)


def add_row(
    rows: list[dict[str, Any]],
    *,
    section: str,
    category: str,
    parameter: str,
    local_name: str | None = None,
    local: dict[str, Any] | None = None,
    hhxyy_name: str | None = None,
    hhxyy_free: dict[str, Any] | None = None,
    hhxyy_mu0: dict[str, Any] | None = None,
    hhxyy_fixed: dict[str, Any] | None = None,
    note: str = "",
) -> None:
    local_value = value(local, "value")
    free_value = value(hhxyy_free, "value")
    mu0_value = value(hhxyy_mu0, "value")
    fixed_value = value(hhxyy_fixed, "value")
    comparison_value = free_value if free_value is not None else fixed_value
    rows.append(
        {
            "section": section,
            "category": category,
            "parameter": parameter,
            "local_name": local_name,
            "local_value": local_value,
            "local_error": value(local, "error"),
            "local_constant": value(local, "constant"),
            "hhxyy_name": hhxyy_name,
            "hhxyy_free_value": free_value,
            "hhxyy_free_error": value(hhxyy_free, "error"),
            "hhxyy_mu0_value": mu0_value,
            "hhxyy_mu0_error": value(hhxyy_mu0, "error"),
            "hhxyy_fixed_value": fixed_value,
            "hhxyy_fixed_error": value(hhxyy_fixed, "error"),
            "free_or_fixed_minus_local": delta(comparison_value, local_value),
            "free_or_fixed_rel_delta": rel_delta(comparison_value, local_value),
            "note": note,
        }
    )


def markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(column)) for column in columns) + " |")
    return "\n".join(lines)


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fieldnames = [
        "section",
        "category",
        "parameter",
        "local_name",
        "local_value",
        "local_error",
        "local_constant",
        "hhxyy_name",
        "hhxyy_free_value",
        "hhxyy_free_error",
        "hhxyy_mu0_value",
        "hhxyy_mu0_error",
        "hhxyy_fixed_value",
        "hhxyy_fixed_error",
        "free_or_fixed_minus_local",
        "free_or_fixed_rel_delta",
        "note",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, help="Pipeline output directory")
    parser.add_argument("--fit-id", default="FIT1")
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    output = Path(args.output)
    fit_dir = output / "fit" / args.fit_id
    out_dir = Path(args.out_dir) if args.out_dir else fit_dir / "parameter_comparison"
    out_dir.mkdir(parents=True, exist_ok=True)

    local_results = read_json(fit_dir / "results.json")
    local_sig = read_json(fit_dir / "significance_asimov_plot_payload.json")
    hhxyy_sig = read_json(fit_dir / "significance_asimov.json")
    local_ws = read_workspace_vars(fit_dir / "workspace.root", "w")
    hhxyy_ws = read_workspace_vars(fit_dir / "hhxyy_workspace" / "fitting" / "workspace" / "ws_combined.root", "combined")
    qf_dir = fit_dir / "hhxyy_workspace" / "fitting" / "fit"
    qf_free = read_quickfit(qf_dir / "bestfit_mu_asimovData_1.root")
    qf_mu0 = read_quickfit(qf_dir / "bestfit_mu0_asimovData_1.root")

    rows: list[dict[str, Any]] = []
    add_row(
        rows,
        section="combined",
        category="combined",
        parameter="mu",
        local_name="mu",
        local=local_var(local_ws, "mu"),
        hhxyy_name="mu",
        hhxyy_free=qf_param(qf_free, "mu"),
        hhxyy_mu0={"value": 0.0, "error": 0.0, "constant": True},
        note="POI; local value is from local RooFit measurement workspace, HHXYY free value is quickFit final",
    )
    for parameter, local_value, hhxyy_value, note in [
        ("nll_free", local_results.get("min_nll"), qf_free["tree"].get("nll"), "local measurement NLL vs HHXYY free Asimov NLL"),
        ("nll_mu0", None, qf_mu0["tree"].get("nll"), "HHXYY mu=0 Asimov NLL"),
        ("q0", None, hhxyy_sig.get("q0"), "central HHXYY quickFit discovery test statistic"),
        ("z_discovery", None, hhxyy_sig.get("z_discovery"), "central HHXYY quickFit expected discovery Z"),
        ("fit_status_free", local_sig.get("free_fit", {}).get("fit_status"), qf_free.get("status"), "fitResult status"),
        ("fit_status_mu0", local_sig.get("mu0_fit", {}).get("fit_status"), qf_mu0.get("status"), "fitResult status"),
        ("cov_qual_free", local_sig.get("free_fit", {}).get("cov_qual"), qf_free.get("cov_qual"), "fitResult covariance quality"),
        ("cov_qual_mu0", local_sig.get("mu0_fit", {}).get("cov_qual"), qf_mu0.get("cov_qual"), "fitResult covariance quality"),
    ]:
        add_row(
            rows,
            section="combined",
            category="combined",
            parameter=parameter,
            local={"value": local_value} if local_value is not None else None,
            hhxyy_free={"value": hhxyy_value} if hhxyy_value is not None and "mu0" not in parameter else None,
            hhxyy_mu0={"value": hhxyy_value} if hhxyy_value is not None and "mu0" in parameter else None,
            note=note,
        )

    for idx, category in enumerate(CATEGORY_ORDER):
        if category not in local_results.get("categories", []):
            continue

        add_row(
            rows,
            section="normalization",
            category=category,
            parameter="nbkg",
            local_name=f"nbkg_{category}",
            local=local_var(local_ws, f"nbkg_{category}"),
            hhxyy_name=f"nbkg_category_{idx}",
            hhxyy_free=qf_param(qf_free, f"nbkg_category_{idx}"),
            hhxyy_mu0=qf_param(qf_mu0, f"nbkg_category_{idx}"),
            note="floating background normalization",
        )
        add_row(
            rows,
            section="normalization",
            category=category,
            parameter="signal_yield_nominal",
            local_name=f"sconst_{category}",
            local=local_var(local_ws, f"sconst_{category}"),
            hhxyy_name=f"yield_signal_category_{idx}_category_{idx}",
            hhxyy_fixed=qf_workspace_value(hhxyy_ws, f"yield_signal_category_{idx}_category_{idx}"),
            note="fixed nominal signal yield before multiplying by mu",
        )

        signal_map = [
            ("mean", f"mean_final_{category}", f"meanCB_category_{idx}_category_{idx}", "both fixed"),
            ("sigmaCB", f"sigmaCB_final_{category}", f"sigmaCB_category_{idx}", "both fixed"),
            ("alphaL", f"alphaL_final_{category}", f"alphaCBLo_category_{idx}", "signal-shape nuisance fixed by quickFit -n pattern"),
            ("nL", f"nL_final_{category}", f"nCBLo_category_{idx}", "signal-shape nuisance fixed by quickFit -n pattern"),
            ("alphaR", f"alphaR_final_{category}", f"alphaCBHi_category_{idx}", "signal-shape nuisance fixed by quickFit -n pattern"),
            ("nR", f"nR_final_{category}", f"nCBHi_category_{idx}", "signal-shape nuisance fixed by quickFit -n pattern"),
        ]
        for parameter, local_name, hhxyy_name, note in signal_map:
            add_row(
                rows,
                section="signal_shape",
                category=category,
                parameter=parameter,
                local_name=local_name,
                local=local_var(local_ws, local_name) if local_name else None,
                hhxyy_name=hhxyy_name,
                hhxyy_fixed=qf_workspace_value(hhxyy_ws, hhxyy_name) if hhxyy_name else None,
                note=note,
            )

        for local_name, payload in sorted(local_ws.items()):
            if local_name.startswith(f"tau_final_{category}"):
                add_row(
                    rows,
                    section="background_shape",
                    category=category,
                    parameter="tau",
                    local_name=local_name,
                    local=payload,
                    hhxyy_name=f"p1_category_{idx}",
                    hhxyy_free=qf_param(qf_free, f"p1_category_{idx}"),
                    hhxyy_mu0=qf_param(qf_mu0, f"p1_category_{idx}"),
                    note="exponential slope; local uses exp(m*tau), HHXYY uses exp((m-125)*p1)",
                )
            match = re.fullmatch(rf"c(\d+)_final_{re.escape(category)}_bernstein(\d+)", local_name)
            p_match = re.fullmatch(rf"p(\d+)_final_{re.escape(category)}", local_name)
            if match or p_match:
                coeff_idx = int(match.group(1)) + 1 if match else int(p_match.group(1))
                hhxyy_name = f"p{coeff_idx}_category_{idx}"
                if qf_param(qf_free, hhxyy_name) is None and qf_param(qf_mu0, hhxyy_name) is None:
                    hhxyy_name = None
                    note = "fixed trailing Bernstein coefficient"
                else:
                    note = "floating Bernstein shape coefficient"
                add_row(
                    rows,
                    section="background_shape",
                    category=category,
                    parameter=f"bernstein_p{coeff_idx}",
                    local_name=local_name,
                    local=payload,
                    hhxyy_name=hhxyy_name,
                    hhxyy_free=qf_param(qf_free, hhxyy_name) if hhxyy_name else None,
                    hhxyy_mu0=qf_param(qf_mu0, hhxyy_name) if hhxyy_name else None,
                    hhxyy_fixed={"value": 1.0, "error": 0.0, "constant": True} if hhxyy_name is None else None,
                    note=note,
                )

    csv_path = out_dir / "local_roofit_vs_hhxyy_all_fit_parameters.csv"
    json_path = out_dir / "local_roofit_vs_hhxyy_all_fit_parameters.json"
    md_path = out_dir / "local_roofit_vs_hhxyy_all_fit_parameters.md"
    write_csv(rows, csv_path)
    json_path.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")

    columns = [
        "category",
        "parameter",
        "local_value",
        "local_error",
        "hhxyy_free_value",
        "hhxyy_free_error",
        "hhxyy_mu0_value",
        "hhxyy_fixed_value",
        "note",
    ]
    sections = []
    for section in ["combined", "normalization", "signal_shape", "background_shape"]:
        section_rows = [row for row in rows if row["section"] == section]
        sections.extend([f"## {section.replace('_', ' ').title()}", "", markdown_table(section_rows, columns), ""])
    md_path.write_text(
        "\n".join(
            [
                "# Local RooFit vs HHXYY Fit Parameters",
                "",
                f"Output: `{output}`",
                "",
                "Local columns come from `fit/FIT1/workspace.root` and local JSON fit summaries. HHXYY free/mu0 columns come from quickFit `fitResult` outputs; fixed HHXYY constants come from `ws_combined.root`.",
                "",
                *sections,
            ]
        )
    )
    print(md_path)
    print(csv_path)
    print(json_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
