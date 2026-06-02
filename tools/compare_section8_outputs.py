from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


EVENT_TABLE_NAMES = (
    "section8_events.npz",
    "events.npz",
    "selected_events.npz",
    "processed_events.npz",
)
SUMMARY_ARTIFACTS = ("cutflow_baseline.json", "category_yields.json")
EXACT_FIELDS = {
    "event_number",
    "run_number",
    "trigger_passed",
    "baseline_selected",
    "is_sideband",
    "is_signal_window",
    "N_jets_25",
    "N_jets_30",
    "N_jets_25_jvt_diagnostic",
    "N_jets_30_jvt_diagnostic",
    "N_central_jets_25",
    "N_forward_jets_25",
    "N_btag_25",
    "N_lep",
    "Z_ll_veto",
    "m_e_gamma_veto",
    "training_mask_tth",
    "training_mask_vh",
    "training_mask_vbf",
    "assigned_category",
    "assignment_blocked",
    "assignment_reason",
    "category",
    "blocked",
    "reason",
    "photon_region",
    "nominal_photon_region",
    "anti_id_or_iso_control_region",
    "bdt_subregion",
}


def _find_event_table(path: Path) -> Path | None:
    if path.is_file() and path.suffix == ".npz":
        return path
    if not path.is_dir():
        return None
    for name in EVENT_TABLE_NAMES:
        candidate = path / name
        if candidate.exists():
            return candidate
    npz_files = sorted(path.glob("*.npz"))
    return npz_files[0] if len(npz_files) == 1 else None


def _load_event_table(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {name: np.asarray(payload[name]) for name in payload.files}


def _event_index(table: dict[str, np.ndarray]) -> tuple[dict[tuple[int | None, int], int], list[tuple[int | None, int]]]:
    event_numbers = table.get("event_number")
    if event_numbers is None:
        raise ValueError("event table is missing required field 'event_number'")
    run_numbers = table.get("run_number")
    index: dict[tuple[int | None, int], int] = {}
    duplicates: list[tuple[int | None, int]] = []
    for idx, event_number in enumerate(event_numbers):
        run_number = None if run_numbers is None else int(run_numbers[idx])
        key = (run_number, int(event_number))
        if key in index:
            duplicates.append(key)
        index[key] = idx
    return index, duplicates


def _is_float_field(name: str, values: np.ndarray) -> bool:
    return name not in EXACT_FIELDS and np.issubdtype(values.dtype, np.floating)


def _max_float_diffs(reference: np.ndarray, candidate: np.ndarray) -> tuple[float, float]:
    ref = reference.astype(float)
    cand = candidate.astype(float)
    finite_pair = np.isfinite(ref) & np.isfinite(cand)
    if not np.any(finite_pair):
        return 0.0, 0.0
    diff = np.abs(ref[finite_pair] - cand[finite_pair])
    denom = np.maximum(np.abs(ref[finite_pair]), np.finfo(float).tiny)
    return float(np.max(diff)), float(np.max(diff / denom))


def _compare_event_tables(
    reference: dict[str, np.ndarray],
    candidate: dict[str, np.ndarray],
    *,
    abs_tol: float,
    rel_tol: float,
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "status": "ok",
        "reference_events": int(len(reference.get("event_number", []))),
        "candidate_events": int(len(candidate.get("event_number", []))),
        "overlap_events": 0,
        "fields_compared": [],
        "field_differences": {},
        "missing_fields": {},
        "failures": [],
    }
    try:
        reference_index, reference_duplicates = _event_index(reference)
        candidate_index, candidate_duplicates = _event_index(candidate)
    except ValueError as exc:
        summary["status"] = "fail"
        summary["failures"].append(str(exc))
        return summary

    if reference_duplicates or candidate_duplicates:
        summary["status"] = "fail"
        summary["failures"].append(
            "duplicate event identifiers found in reference or candidate"
        )
        summary["duplicate_reference_events"] = reference_duplicates[:10]
        summary["duplicate_candidate_events"] = candidate_duplicates[:10]
        return summary

    common_keys = sorted(set(reference_index) & set(candidate_index))
    summary["overlap_events"] = len(common_keys)
    if not common_keys:
        summary["status"] = "fail"
        summary["failures"].append("no overlapping event identifiers")
        return summary

    ref_indices = np.asarray([reference_index[key] for key in common_keys], dtype=int)
    cand_indices = np.asarray([candidate_index[key] for key in common_keys], dtype=int)
    reference_only = sorted(set(reference) - set(candidate))
    candidate_only = sorted(set(candidate) - set(reference))
    if reference_only or candidate_only:
        summary["missing_fields"] = {
            "reference_only": reference_only,
            "candidate_only": candidate_only,
        }
        summary["status"] = "fail"
        summary["failures"].append("event tables do not expose the same fields")

    for field in sorted(set(reference) & set(candidate)):
        ref_values = np.asarray(reference[field])[ref_indices]
        cand_values = np.asarray(candidate[field])[cand_indices]
        if _is_float_field(field, ref_values):
            passed = bool(np.allclose(ref_values, cand_values, rtol=rel_tol, atol=abs_tol, equal_nan=True))
            max_abs, max_rel = _max_float_diffs(ref_values, cand_values)
            field_summary = {
                "comparison": "float",
                "passed": passed,
                "max_abs_diff": max_abs,
                "max_rel_diff": max_rel,
            }
        else:
            passed = bool(np.array_equal(ref_values, cand_values))
            field_summary = {"comparison": "exact", "passed": passed}
        summary["fields_compared"].append(field)
        if not passed:
            summary["field_differences"][field] = field_summary
            summary["status"] = "fail"
            summary["failures"].append(f"event field '{field}' differs")
    return summary


def _artifact_path(base: Path, name: str) -> Path | None:
    if base.is_dir():
        candidate = base / name
        return candidate if candidate.exists() else None
    return base if base.is_file() and base.name == name else None


def _load_json(path: Path) -> Any:
    with path.open() as handle:
        return json.load(handle)


def _compare_json_value(reference: Any, candidate: Any, path: str, *, abs_tol: float, rel_tol: float, failures: list[str]) -> None:
    if isinstance(reference, dict) and isinstance(candidate, dict):
        for key in sorted(set(reference) | set(candidate)):
            child_path = f"{path}.{key}" if path else str(key)
            if key not in reference or key not in candidate:
                failures.append(f"missing JSON key at {child_path}")
                continue
            _compare_json_value(reference[key], candidate[key], child_path, abs_tol=abs_tol, rel_tol=rel_tol, failures=failures)
        return
    if isinstance(reference, list) and isinstance(candidate, list):
        if len(reference) != len(candidate):
            failures.append(f"list length differs at {path}: {len(reference)} != {len(candidate)}")
            return
        for idx, (ref_item, cand_item) in enumerate(zip(reference, candidate)):
            _compare_json_value(ref_item, cand_item, f"{path}[{idx}]", abs_tol=abs_tol, rel_tol=rel_tol, failures=failures)
        return
    if isinstance(reference, (int, float)) and isinstance(candidate, (int, float)):
        if not np.isclose(float(reference), float(candidate), rtol=rel_tol, atol=abs_tol, equal_nan=True):
            failures.append(f"numeric JSON value differs at {path}: {reference} != {candidate}")
        return
    if reference != candidate:
        failures.append(f"JSON value differs at {path}: {reference!r} != {candidate!r}")


def _compare_json_artifact(reference_path: Path | None, candidate_path: Path | None, *, abs_tol: float, rel_tol: float) -> dict[str, Any]:
    if reference_path is None and candidate_path is None:
        return {"status": "unavailable", "message": "artifact not present in either output"}
    if reference_path is None or candidate_path is None:
        return {
            "status": "fail",
            "message": "artifact present in only one output",
            "reference_path": None if reference_path is None else str(reference_path),
            "candidate_path": None if candidate_path is None else str(candidate_path),
        }
    failures: list[str] = []
    _compare_json_value(
        _load_json(reference_path),
        _load_json(candidate_path),
        "",
        abs_tol=abs_tol,
        rel_tol=rel_tol,
        failures=failures,
    )
    return {
        "status": "ok" if not failures else "fail",
        "reference_path": str(reference_path),
        "candidate_path": str(candidate_path),
        "failures": failures,
    }


def compare_section8_outputs(reference: Path, candidate: Path, *, abs_tol: float = 1e-9, rel_tol: float = 1e-9) -> dict[str, Any]:
    reference = Path(reference)
    candidate = Path(candidate)
    summary: dict[str, Any] = {
        "status": "ok",
        "reference": str(reference),
        "candidate": str(candidate),
        "abs_tol": abs_tol,
        "rel_tol": rel_tol,
        "event_table": {},
        "summary_artifacts": {},
        "failures": [],
    }

    reference_table_path = _find_event_table(reference)
    candidate_table_path = _find_event_table(candidate)
    if reference_table_path is None and candidate_table_path is None:
        summary["event_table"] = {
            "status": "unavailable",
            "message": "No event NPZ artifact found. Expected one of: " + ", ".join(EVENT_TABLE_NAMES),
        }
    elif reference_table_path is None or candidate_table_path is None:
        summary["event_table"] = {
            "status": "fail",
            "message": "event NPZ artifact present in only one output",
            "reference_path": None if reference_table_path is None else str(reference_table_path),
            "candidate_path": None if candidate_table_path is None else str(candidate_table_path),
        }
        summary["status"] = "fail"
        summary["failures"].append("event NPZ artifact present in only one output")
    else:
        event_summary = _compare_event_tables(
            _load_event_table(reference_table_path),
            _load_event_table(candidate_table_path),
            abs_tol=abs_tol,
            rel_tol=rel_tol,
        )
        event_summary["reference_path"] = str(reference_table_path)
        event_summary["candidate_path"] = str(candidate_table_path)
        summary["event_table"] = event_summary
        if event_summary["status"] != "ok":
            summary["status"] = "fail"
            summary["failures"].extend(event_summary.get("failures", []))

    for artifact_name in SUMMARY_ARTIFACTS:
        artifact_summary = _compare_json_artifact(
            _artifact_path(reference, artifact_name),
            _artifact_path(candidate, artifact_name),
            abs_tol=abs_tol,
            rel_tol=rel_tol,
        )
        summary["summary_artifacts"][artifact_name] = artifact_summary
        if artifact_summary["status"] == "fail":
            summary["status"] = "fail"
            summary["failures"].append(f"{artifact_name} differs or is missing")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare bounded Section 8 output artifacts.")
    parser.add_argument("--reference", required=True, type=Path, help="Pre-refactor or reference Section 8 output directory or event NPZ file.")
    parser.add_argument("--candidate", required=True, type=Path, help="Post-refactor or candidate Section 8 output directory or event NPZ file.")
    parser.add_argument("--abs-tol", type=float, default=1e-9, help="Absolute tolerance for floating-point fields.")
    parser.add_argument("--rel-tol", type=float, default=1e-9, help="Relative tolerance for floating-point fields.")
    args = parser.parse_args(argv)
    summary = compare_section8_outputs(args.reference, args.candidate, abs_tol=args.abs_tol, rel_tol=args.rel_tol)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
