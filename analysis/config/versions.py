from __future__ import annotations

from pathlib import Path
from typing import Any

from analysis.common import read_json


DEFAULT_VERSION_REGISTRY = Path(__file__).resolve().parents[1] / "analysis_versions.json"


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_analysis_versions(path: Path | None = None) -> dict[str, Any]:
    return read_json(path or DEFAULT_VERSION_REGISTRY)


def apply_analysis_version(
    runtime_defaults: dict[str, Any],
    *,
    version_name: str | None,
    version_registry: Path | None = None,
    section8_ads_path: Path | None = None,
    section8_bdt_artifacts: Path | None = None,
    routing_config: Path | None = None,
) -> dict[str, Any]:
    runtime = dict(runtime_defaults)
    if version_name:
        versions = load_analysis_versions(version_registry)
        if version_name not in versions:
            raise ValueError(f"Unknown analysis version '{version_name}'. Available versions: {sorted(versions)}")
        runtime = deep_merge(runtime, versions[version_name])
    if section8_ads_path is not None or section8_bdt_artifacts is not None:
        section8 = dict(runtime.get("section8_ads", {}))
        if section8_ads_path is not None:
            section8["ads_path"] = str(section8_ads_path)
        if section8_bdt_artifacts is not None:
            section8["bdt_artifacts_dir"] = str(section8_bdt_artifacts)
        runtime["section8_ads"] = section8
    if routing_config is not None:
        implementation = dict(runtime.get("analysis_implementation", {}))
        implementation["routing_config"] = str(routing_config)
        runtime["analysis_implementation"] = implementation
    return runtime
