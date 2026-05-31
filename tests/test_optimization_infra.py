from __future__ import annotations

import json
from pathlib import Path

import yaml

from optimization_infra.create_decision_packet import create_decision_packet
from optimization_infra.plan_candidate_run import (
    config_subset,
    load_structured,
    plan_candidate_run,
    stable_hash,
    write_execution_plan,
    write_yaml,
)
from optimization_infra.run_scan import append_run_registry, run_scan, validate_scan_spec
from optimization_infra.summarize_scan import create_observations, summarize_scan


def workflow() -> dict:
    return {
        "workflow_id": "synthetic",
        "version": "0.1.0",
        "stages": [
            {
                "stage_id": "summary",
                "name": "Summary",
                "service_id": "summary_normalizer",
                "depends_on": [],
                "config_dependencies": ["a"],
                "outputs": [{"artifact_type": "summary_out", "schema_version": 1}],
                "expected_runtime_class": "fast",
                "failure_handling": "fail",
            },
            {
                "stage_id": "fit",
                "name": "Fit",
                "service_id": "fit_builder",
                "depends_on": ["summary"],
                "config_dependencies": ["fit.range"],
                "outputs": [{"artifact_type": "fit_out", "schema_version": 1}],
                "expected_runtime_class": "medium",
                "failure_handling": "fail",
            },
        ],
    }


def ledger() -> dict:
    return {
        "services": [
            {"service_id": "summary_normalizer", "version": "0.1.0"},
            {"service_id": "fit_builder", "version": "0.1.0"},
        ]
    }


def invariants() -> dict:
    return {"scientific_invariants": [{"id": "sci_blinding_default"}]}


def artifact(stage_id: str, artifact_type: str, config: dict, deps: list[str]) -> dict:
    subset = config_subset(config, deps)
    service_id = "summary_normalizer" if stage_id == "summary" else "fit_builder"
    return {
        "artifact_id": f"{stage_id}_artifact",
        "artifact_type": artifact_type,
        "producing_run_id": "parent",
        "producing_stage_id": stage_id,
        "created_at": "2026-05-31T00:00:00+00:00",
        "path": f"runs/parent/{artifact_type}.json",
        "content_hash": f"hash_{stage_id}",
        "schema_version": 1,
        "service_id": service_id,
        "service_version": "0.1.0",
        "git": {"commit": "abc", "dirty": False},
        "config_subset": subset,
        "config_subset_hash": stable_hash(subset),
        "upstream_artifact_hashes": {},
        "verifier_status": "pass",
        "reusable": True,
    }


def parent_config() -> dict:
    return {"a": 1, "fit": {"range": [105, 160]}, "runtime_defaults": {"blinding": {"observed_significance_allowed": False}}}


def test_workflow_graph_parsing() -> None:
    graph = load_structured("optimization_infra/workflow_graph.yaml")
    assert graph["workflow_id"] == "main_analysis_workflow"
    assert any(stage["stage_id"] == "significance" for stage in graph["stages"])


def test_artifact_manifest_schema_parsing() -> None:
    schema = load_structured("optimization_infra/artifact_manifest_schema.yaml")
    assert "artifact_id" in schema["required_fields"]
    assert "verifier_status" in schema["fields"]


def test_reuse_decisions_when_provenance_matches() -> None:
    cfg = parent_config()
    plan = plan_candidate_run(
        run_id="candidate",
        parent_run_id="parent",
        parent_config=cfg,
        candidate_config=cfg,
        workflow_graph=workflow(),
        artifact_manifests=[
            artifact("summary", "summary_out", cfg, ["a"]),
            artifact("fit", "fit_out", cfg, ["fit.range"]),
        ],
        executable_services_ledger=ledger(),
        global_invariants=invariants(),
    )
    assert [stage["decision"] for stage in plan["stages"]] == ["reuse", "reuse"]
    assert plan["earliest_stage_to_rerun"] == "none"


def test_invalidation_decision_for_relevant_config_change() -> None:
    cfg = parent_config()
    candidate = {"a": 2, "fit": {"range": [105, 160]}, "runtime_defaults": {"blinding": {"observed_significance_allowed": False}}}
    plan = plan_candidate_run(
        run_id="candidate",
        parent_run_id="parent",
        parent_config=cfg,
        candidate_config=candidate,
        workflow_graph=workflow(),
        artifact_manifests=[
            artifact("summary", "summary_out", cfg, ["a"]),
            artifact("fit", "fit_out", cfg, ["fit.range"]),
        ],
        executable_services_ledger=ledger(),
        global_invariants=invariants(),
    )
    assert plan["stages"][0]["decision"] == "rerun"
    assert plan["stages"][1]["decision"] == "rerun"
    assert plan["earliest_stage_to_rerun"] == "summary"


def test_dry_run_execution_plan_written(tmp_path: Path) -> None:
    cfg = parent_config()
    plan = plan_candidate_run(
        run_id="candidate",
        parent_run_id="parent",
        parent_config=cfg,
        candidate_config=cfg,
        workflow_graph=workflow(),
        artifact_manifests=[
            artifact("summary", "summary_out", cfg, ["a"]),
            artifact("fit", "fit_out", cfg, ["fit.range"]),
        ],
        executable_services_ledger=ledger(),
        global_invariants=invariants(),
    )
    write_execution_plan(tmp_path / "candidate", plan)
    assert (tmp_path / "candidate" / "EXECUTION_PLAN.md").exists()
    assert yaml.safe_load((tmp_path / "candidate" / "execution_plan.yaml").read_text())["run_id"] == "candidate"


def test_scan_spec_parsing() -> None:
    spec = load_structured("optimization_infra/example_scan.yaml")
    validate_scan_spec(spec)
    assert spec["execution"]["dry_run_only"] is True


def test_run_registry_append_behavior(tmp_path: Path) -> None:
    registry = tmp_path / "runs.jsonl"
    entry = {
        "run_id": "r1",
        "timestamp": "now",
        "parent_run_id": None,
        "branch_id": "b",
        "strategy_id": "s",
        "run_type": "baseline",
        "objective": "test",
        "configuration_snapshot_path": "cfg.yaml",
        "changed_parameters": [],
        "reused_artifacts": [],
        "invalidated_artifacts": [],
        "stages_executed": [],
        "stages_skipped": [],
        "verifier_status": "not_run",
        "cut_flow_path": None,
        "yield_table_path": None,
        "fit_output_path": None,
        "metric_values": {},
        "expected_significance": None,
        "observed_significance": None,
        "runtime": None,
        "warnings": [],
        "failure_reason": None,
        "human_or_agent_note": None,
        "git_state": {},
        "service_versions": {},
    }
    append_run_registry(registry, entry)
    append_run_registry(registry, {**entry, "run_id": "r2"})
    assert len([line for line in registry.read_text().splitlines() if line.strip()]) == 2


def test_observation_summary_generation() -> None:
    obs = create_observations(
        run_id="candidate",
        parent_run_id="parent",
        changed_parameters=[{"config_path": "x", "parent": 1, "candidate": 2}],
        baseline_metrics={"expected_significance": 1.0},
        candidate_metrics={"expected_significance": 1.2},
        execution_plan={"status": "planned", "stages": [], "reused_artifacts": []},
    )
    assert obs["interpretation"]["status"] == "credible improvement"
    assert obs["primary_metric_response"]["absolute_change"] == 0.19999999999999996


def test_decision_packet_generation(tmp_path: Path) -> None:
    obs_path = tmp_path / "observations.yaml"
    write_yaml(
        obs_path,
        {
            "run_id": "candidate",
            "interpretation": {"status": "ambiguous result"},
            "primary_metric_response": {},
        },
    )
    packet = create_decision_packet(
        decision_id="decision_001",
        objective={"scientific_goal": "test", "optimization_metric": "expected_significance"},
        lineage={"baseline_run": "baseline", "parent_run": "parent", "branch": "b"},
        evidence_paths=[str(obs_path)],
        runs_dir=tmp_path,
    )
    assert "request human clarification" in packet["suggested_decision_types"]
    assert (tmp_path / "decision_001" / "DECISION_PACKET.md").exists()


def test_branch_creation_recorded_through_scan_registry(tmp_path: Path) -> None:
    base = tmp_path / "base.yaml"
    write_yaml(base, parent_config())
    graph = tmp_path / "graph.yaml"
    write_yaml(graph, workflow())
    led = tmp_path / "ledger.yaml"
    write_yaml(led, ledger())
    inv = tmp_path / "invariants.yaml"
    write_yaml(inv, invariants())
    spec = {
        "scan_id": "branch_scan",
        "parent_run_id": "parent",
        "branch_id": "qual_branch_001",
        "strategy_id": "strategy_branch",
        "objective_metric": "expected_significance",
        "base_config": str(base),
        "parent_config": str(base),
        "workflow_graph": str(graph),
        "artifact_manifests": [],
        "ledger": str(led),
        "global_invariants": str(inv),
        "parameters": [{"config_path": "fit.range", "values": [[110, 150]]}],
        "execution": {"dry_run_only": True, "reuse_valid_artifacts": True, "max_parallel_jobs": 1, "stop_on_verifier_failure": False},
        "reporting": {"summarize_cut_flow_changes": True, "summarize_yield_changes": True, "summarize_metric_changes": True},
    }
    registry = tmp_path / "runs.jsonl"
    run_scan(scan_spec=spec, runs_dir=tmp_path / "runs", registry_path=registry)
    record = json.loads(registry.read_text().splitlines()[0])
    assert record["branch_id"] == "qual_branch_001"


def test_failure_when_provenance_is_incomplete() -> None:
    cfg = parent_config()
    incomplete = {"artifact_id": "bad", "artifact_type": "summary_out", "producing_run_id": "parent", "producing_stage_id": "summary"}
    plan = plan_candidate_run(
        run_id="candidate",
        parent_run_id="parent",
        parent_config=cfg,
        candidate_config=cfg,
        workflow_graph=workflow(),
        artifact_manifests=[incomplete, artifact("fit", "fit_out", cfg, ["fit.range"])],
        executable_services_ledger=ledger(),
        global_invariants=invariants(),
    )
    assert plan["stages"][0]["decision"] == "rerun"
    assert "incomplete provenance" in " ".join(plan["stages"][0]["reasons"])


def test_failure_when_invariant_would_be_violated() -> None:
    cfg = parent_config()
    candidate = {"a": 1, "fit": {"range": [105, 160]}, "runtime_defaults": {"blinding": {"observed_significance_allowed": True}}}
    plan = plan_candidate_run(
        run_id="candidate",
        parent_run_id="parent",
        parent_config=cfg,
        candidate_config=candidate,
        workflow_graph=workflow(),
        artifact_manifests=[
            artifact("summary", "summary_out", cfg, ["a"]),
            artifact("fit", "fit_out", cfg, ["fit.range"]),
        ],
        executable_services_ledger=ledger(),
        global_invariants=invariants(),
    )
    assert plan["status"] == "blocked"
    assert plan["invariant_violations"]


def test_scan_summary_generation(tmp_path: Path) -> None:
    obs1 = tmp_path / "obs1.yaml"
    obs2 = tmp_path / "obs2.yaml"
    write_yaml(obs1, create_observations(run_id="c1", parent_run_id="p", changed_parameters=[], baseline_metrics={"expected_significance": 1.0}, candidate_metrics={"expected_significance": 1.1}, execution_plan={"status": "planned", "stages": []}))
    write_yaml(obs2, create_observations(run_id="c2", parent_run_id="p", changed_parameters=[], baseline_metrics={"expected_significance": 1.0}, candidate_metrics={"expected_significance": 0.9}, execution_plan={"status": "planned", "stages": []}))
    summary = summarize_scan("scan", [obs1, obs2], runs_dir=tmp_path)
    assert summary["best_candidate"]["run_id"] == "c1"
    assert (tmp_path / "scan" / "SCAN_SUMMARY.md").exists()
