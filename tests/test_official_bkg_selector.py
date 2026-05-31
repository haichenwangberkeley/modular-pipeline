import json
from pathlib import Path

from analysis.stats.official_bkg_selector import (
    SelectionThresholds,
    make_configs,
    parse_results_dir,
    parse_results_file,
    select_candidate,
)


def test_parse_compact_hhxyy_results_marks_selected(tmp_path: Path) -> None:
    result = tmp_path / "results.txt"
    result.write_text(
        "\n".join(
            [
                "Name max(S/dS) max(1sigma/dS) max(2sigma/dS) max(S/Sref) max(S) S[125] Sref[125] dS[125]",
                "Bern2 0 0 0 0 16.711 16.711 66.964 1 <== Selected",
            ]
        )
    )

    candidates = parse_results_file(result, category="cat0")

    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate.name == "Bern2"
    assert candidate.official_selected is True
    assert candidate.local_equivalent == "bernstein1"
    assert candidate.hhxyy_xml_model == "Bern2"
    assert candidate.passes_official_thresholds is True


def test_recomputes_default_official_ranking_without_selected_marker() -> None:
    candidates = []
    thresholds = SelectionThresholds()
    for line in [
        "Bern4 1 1 1 1 10 10 100 2 3 4 1.2 90",
        "Bern3 1 1 1 1 20 20 100 2 3 3 1.1 90",
        "ExpPoly2 50 50 50 50 1 1 100 2 3 2 1.0 90",
    ]:
        parsed = parse_results_file_from_line(line, thresholds)
        candidates.append(parsed)

    selected, source = select_candidate(candidates)

    assert source == "recomputed_official_ranking"
    assert selected is not None
    assert selected.name == "Bern3"


def test_parse_results_dir_writes_substitution_payload(tmp_path: Path) -> None:
    cat0 = tmp_path / "cat0"
    cat0.mkdir()
    (cat0 / "results.txt").write_text(
        "Name max(S/dS) max(1sigma/dS) max(2sigma/dS) max(S/Sref) max(S) S[125] Sref[125] dS[125]\n"
        "Bern3 0 0 0 0 1 1 100 2 <== Selected\n"
    )

    summary = parse_results_dir(tmp_path)

    assert summary["categories"]["cat0"]["selected_model"] == "Bern3"
    assert summary["categories"]["cat0"]["selected_local_equivalent"] == "bernstein2"
    json.dumps(summary)


def test_make_configs_uses_official_pdf_expressions(tmp_path: Path) -> None:
    yields = tmp_path / "category_yield.yaml"
    yields.write_text("signal:\n  category_0: 66.5\n")

    written = make_configs(
        output_dir=tmp_path / "configs",
        dataset_file=tmp_path / "yyjets.root",
        signal_pdf_file=tmp_path / "signal.root",
        signal_yields_file=yields,
        models=["ExpPoly2", "Bern3"],
        dataset_histogram_template="category{cat}",
    )

    assert len(written) == 1
    text = written[0].read_text()
    assert "Background.PDFs: ExpPoly2 Bern3" in text
    assert "RooBernstein(atlas_invMass_gamgam_cat0" in text
    assert "RefSignalYield: 66.5" in text
    assert "Dataset.HistogramName: category0" in text


def parse_results_file_from_line(line: str, thresholds: SelectionThresholds):
    path = Path("/tmp/nonexistent")
    from analysis.stats.official_bkg_selector import parse_result_line

    candidate = parse_result_line(line, category=None, thresholds=thresholds)
    assert candidate is not None, path
    return candidate
