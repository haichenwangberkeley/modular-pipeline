from __future__ import annotations

from pathlib import Path

import pytest

from modular_pipeline.components import COMPONENTS, available_components, parse_mask, run_modular_pipeline
from modular_pipeline.tracking import inspect_outputs


def test_parse_mask() -> None:
    assert parse_mask("") == set()
    assert parse_mask("fit, plots,stats") == {"fit", "plots", "stats"}


def test_available_components_include_fit_and_significance() -> None:
    names = {component["name"] for component in available_components()}
    assert "fit" in names
    assert "significance" in names


def test_unknown_mask_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unknown component"):
        run_modular_pipeline(
            summary=Path("analysis/analysis.summary.json"),
            inputs=Path("input"),
            outputs=tmp_path / "out",
            mask={"does_not_exist"},
        )


def test_inspect_outputs_reports_ready_summary_entrypoint(tmp_path: Path) -> None:
    state = inspect_outputs(tmp_path, COMPONENTS)
    ready = {item["component"]: item["readiness"] for item in state["entrypoints"]}
    assert ready["summary"] == "ready_without_prior_artifacts"
    assert ready["fit"] == "blocked_missing_artifacts"


def test_inspect_outputs_recognizes_fit_inputs_from_artifacts(tmp_path: Path) -> None:
    for rel in [
        "summary.normalized.json",
        "validation/inventory.json",
        "validation/diagnostics.json",
        "validation/overlap_policy.json",
        "samples.registry.json",
        "report/mc_sample_selection.json",
        "normalization/norm_table.json",
        "hists/templates.json",
        "hists/processed_samples.json",
        "cache/sample.npz",
    ]:
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}")

    state = inspect_outputs(tmp_path, COMPONENTS)
    ready = {item["component"]: item["readiness"] for item in state["entrypoints"]}
    assert ready["fit"] == "ready_from_artifacts"
