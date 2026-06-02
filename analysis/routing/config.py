from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from analysis.routing.predicates import PredicateError, predicate_fields, validate_predicate


REPO_ROOT = Path(__file__).resolve().parents[2]


class RoutingConfigError(ValueError):
    pass


@dataclass(frozen=True)
class CategoryRule:
    id: str
    label: str
    priority: int
    required_inputs: tuple[str, ...]
    select_when: dict[str, Any]
    eligible_when: dict[str, Any]
    block_if_missing: tuple[str, ...]
    reason: str
    block_reason: str


@dataclass(frozen=True)
class RoutingConfig:
    mode: str
    categories: tuple[CategoryRule, ...]
    active_when: dict[str, Any]
    path: Path | None = None
    description: str | None = None

    @property
    def category_ids(self) -> list[str]:
        return [category.id for category in self.categories]

    @property
    def label_by_id(self) -> dict[str, str]:
        return {category.id: category.label for category in self.categories}


def resolve_repo_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else REPO_ROOT / candidate


def routing_config_path_from_runtime(cfg: dict[str, Any], override: str | Path | None = None) -> Path:
    raw_path = override or cfg.get("analysis_implementation", {}).get("routing_config")
    if raw_path is None:
        raise RoutingConfigError("Runtime configuration is missing analysis_implementation.routing_config")
    return resolve_repo_path(raw_path)


def load_routing_config(path: str | Path) -> RoutingConfig:
    resolved = resolve_repo_path(path)
    if not resolved.exists():
        raise RoutingConfigError(f"Routing config does not exist: {resolved}")
    with resolved.open() as handle:
        payload = yaml.safe_load(handle)
    return parse_routing_config(payload, path=resolved)


def parse_routing_config(payload: dict[str, Any], *, path: Path | None = None) -> RoutingConfig:
    if not isinstance(payload, dict):
        raise RoutingConfigError("Routing config must be a mapping")
    routing = payload.get("routing", {})
    if not isinstance(routing, dict):
        raise RoutingConfigError("'routing' must be a mapping")
    mode = routing.get("mode", "ordered_first_match")
    if mode != "ordered_first_match":
        raise RoutingConfigError(f"Invalid routing mode '{mode}'; expected 'ordered_first_match'")
    active_when = routing.get("active_when", {"always": True})
    if not isinstance(active_when, dict):
        raise RoutingConfigError("routing.active_when must be a predicate mapping")
    _validate_predicate(active_when, "routing.active_when")

    raw_categories = payload.get("categories")
    if not isinstance(raw_categories, list) or not raw_categories:
        raise RoutingConfigError("Routing config requires a non-empty 'categories' list")

    ids: set[str] = set()
    priorities: set[int] = set()
    categories: list[CategoryRule] = []
    for idx, raw_category in enumerate(raw_categories):
        if not isinstance(raw_category, dict):
            raise RoutingConfigError(f"Category entry {idx} must be a mapping")
        category_id = raw_category.get("id")
        if not isinstance(category_id, str) or not category_id:
            raise RoutingConfigError(f"Category entry {idx} is missing a non-empty 'id'")
        if category_id in ids:
            raise RoutingConfigError(f"Duplicate category id '{category_id}'")
        ids.add(category_id)
        priority = raw_category.get("priority")
        if not isinstance(priority, int):
            raise RoutingConfigError(f"Category '{category_id}' is missing integer 'priority'")
        if priority in priorities:
            raise RoutingConfigError(f"Duplicate category priority '{priority}'")
        priorities.add(priority)
        label = raw_category.get("label", category_id)
        if not isinstance(label, str) or not label:
            raise RoutingConfigError(f"Category '{category_id}' has invalid 'label'")
        if "required_inputs" not in raw_category:
            raise RoutingConfigError(f"Category '{category_id}' is missing 'required_inputs'")
        required_inputs = raw_category["required_inputs"]
        if not isinstance(required_inputs, list) or not all(isinstance(item, str) for item in required_inputs):
            raise RoutingConfigError(f"Category '{category_id}' required_inputs must be a list of strings")
        if "select_when" not in raw_category:
            raise RoutingConfigError(f"Category '{category_id}' is missing 'select_when'")
        select_when = raw_category["select_when"]
        if not isinstance(select_when, dict):
            raise RoutingConfigError(f"Category '{category_id}' select_when must be a predicate mapping")
        eligible_when = raw_category.get("eligible_when", {"always": True})
        if not isinstance(eligible_when, dict):
            raise RoutingConfigError(f"Category '{category_id}' eligible_when must be a predicate mapping")
        block_if_missing = raw_category.get("block_if_missing", [])
        if not isinstance(block_if_missing, list) or not all(isinstance(item, str) for item in block_if_missing):
            raise RoutingConfigError(f"Category '{category_id}' block_if_missing must be a list of strings")
        if not set(block_if_missing).issubset(set(required_inputs)):
            raise RoutingConfigError(f"Category '{category_id}' block_if_missing entries must also appear in required_inputs")
        _validate_predicate(eligible_when, f"category '{category_id}' eligible_when")
        _validate_predicate(select_when, f"category '{category_id}' select_when")
        declared = set(required_inputs)
        used = predicate_fields(eligible_when) | predicate_fields(select_when)
        missing_declarations = sorted(used - declared)
        if missing_declarations:
            raise RoutingConfigError(f"Category '{category_id}' predicates use undeclared required inputs: {missing_declarations}")
        reason = raw_category.get("reason", f"matched category {category_id}")
        block_reason = raw_category.get("block_reason", "blocked by missing classifier or derived input")
        if not isinstance(reason, str) or not isinstance(block_reason, str):
            raise RoutingConfigError(f"Category '{category_id}' reasons must be strings")
        categories.append(
            CategoryRule(
                id=category_id,
                label=label,
                priority=priority,
                required_inputs=tuple(required_inputs),
                select_when=select_when,
                eligible_when=eligible_when,
                block_if_missing=tuple(block_if_missing),
                reason=reason,
                block_reason=block_reason,
            )
        )

    categories.sort(key=lambda item: item.priority)
    description = payload.get("description")
    return RoutingConfig(
        mode=mode,
        categories=tuple(categories),
        active_when=active_when,
        path=path,
        description=description if isinstance(description, str) else None,
    )


def validate_observables_for_config(observables: dict[str, Any], config: RoutingConfig) -> None:
    available = set(observables)
    for category in config.categories:
        missing = sorted(set(category.required_inputs) - available - set(category.block_if_missing))
        if missing:
            raise RoutingConfigError(f"Missing required observable(s) for category '{category.id}': {missing}")


def _validate_predicate(predicate: dict[str, Any], context: str) -> None:
    try:
        validate_predicate(predicate, context=context)
    except PredicateError as exc:
        raise RoutingConfigError(str(exc)) from exc
