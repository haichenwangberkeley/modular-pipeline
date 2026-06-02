from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from tools.compare_section8_outputs import compare_section8_outputs


def _write_npz(path: Path, *, mass_shift: float = 0.0, category: str = "ggH 0J Cen") -> None:
    np.savez(
        path,
        run_number=np.asarray([1, 1], dtype=np.int64),
        event_number=np.asarray([101, 102], dtype=np.int64),
        weight=np.asarray([1.0, 2.0], dtype=float),
        m_gammagamma=np.asarray([125.0 + mass_shift, 130.0], dtype=float),
        assigned_category=np.asarray([category, "blocked_missing_input"]),
        assignment_blocked=np.asarray([0, 1], dtype=np.int64),
        assignment_reason=np.asarray(["matched", "BDT missing"]),
    )


def test_compare_section8_outputs_passes_tolerated_event_differences(tmp_path: Path) -> None:
    reference = tmp_path / "reference.npz"
    candidate = tmp_path / "candidate.npz"
    _write_npz(reference)
    _write_npz(candidate, mass_shift=1e-10)

    summary = compare_section8_outputs(reference, candidate, abs_tol=1e-8, rel_tol=1e-8)

    assert summary["status"] == "ok"
    assert summary["event_table"]["overlap_events"] == 2
    assert "m_gammagamma" in summary["event_table"]["fields_compared"]


def test_compare_section8_outputs_fails_on_category_or_float_mismatch(tmp_path: Path) -> None:
    reference = tmp_path / "reference.npz"
    candidate = tmp_path / "candidate.npz"
    _write_npz(reference)
    _write_npz(candidate, mass_shift=0.1, category="ggH 0J Fwd")

    summary = compare_section8_outputs(reference, candidate, abs_tol=1e-8, rel_tol=1e-8)

    assert summary["status"] == "fail"
    assert "m_gammagamma" in summary["event_table"]["field_differences"]
    assert "assigned_category" in summary["event_table"]["field_differences"]


def test_compare_section8_outputs_compares_directory_summaries(tmp_path: Path) -> None:
    reference = tmp_path / "reference"
    candidate = tmp_path / "candidate"
    reference.mkdir()
    candidate.mkdir()
    _write_npz(reference / "section8_events.npz")
    _write_npz(candidate / "section8_events.npz")
    cutflow = {"status": "ok", "aggregated": {"mass_window": {"events_after": 2, "weighted_after": 3.0}}}
    yields = {"status": "ok", "rows": [{"sample_id": "data", "category": "ggH 0J Cen", "count": 1}]}
    for directory in (reference, candidate):
        (directory / "cutflow_baseline.json").write_text(json.dumps(cutflow) + "\n")
        (directory / "category_yields.json").write_text(json.dumps(yields) + "\n")

    summary = compare_section8_outputs(reference, candidate)

    assert summary["status"] == "ok"
    assert summary["summary_artifacts"]["cutflow_baseline.json"]["status"] == "ok"
    assert summary["summary_artifacts"]["category_yields.json"]["status"] == "ok"
