from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from analysis.routing.config import RoutingConfig, RoutingConfigError, validate_observables_for_config
from analysis.routing.predicates import PredicateError, evaluate_predicate


@dataclass(frozen=True)
class RoutingResult:
    assigned_category: np.ndarray
    assignment_reason: np.ndarray
    assignment_blocked: np.ndarray
    category_label: np.ndarray


def _event_count(observables: dict[str, np.ndarray]) -> int:
    if not observables:
        raise RoutingConfigError("Cannot route an empty observable table")
    lengths = {len(np.asarray(values)) for values in observables.values()}
    if len(lengths) != 1:
        raise RoutingConfigError(f"Observable arrays must have equal lengths, got {sorted(lengths)}")
    return lengths.pop()


def _missing_mask(observables: dict[str, np.ndarray], fields: tuple[str, ...], n_events: int) -> np.ndarray:
    missing = np.zeros(n_events, dtype=bool)
    for field in fields:
        if field not in observables:
            missing |= np.ones(n_events, dtype=bool)
            continue
        values = np.asarray(observables[field])
        try:
            missing |= ~np.isfinite(values.astype(float))
        except (TypeError, ValueError):
            missing |= values == None  # noqa: E711
    return missing


def route_categories(
    observables: dict[str, np.ndarray],
    routing_config: RoutingConfig,
) -> RoutingResult:
    if routing_config.mode != "ordered_first_match":
        raise RoutingConfigError(f"Invalid routing mode '{routing_config.mode}'")
    n_events = _event_count(observables)
    validate_observables_for_config(observables, routing_config)
    assigned = np.full(n_events, "unassigned", dtype=object)
    labels = np.full(n_events, "", dtype=object)
    reasons = np.full(n_events, "", dtype=object)
    blocked = np.zeros(n_events, dtype=bool)
    try:
        active = evaluate_predicate(routing_config.active_when, observables, n_events=n_events)
    except PredicateError as exc:
        raise RoutingConfigError(str(exc)) from exc

    for category in routing_config.categories:
        missing_fields = set(category.block_if_missing)
        try:
            eligible = evaluate_predicate(
                category.eligible_when,
                observables,
                n_events=n_events,
                missing_false_fields=missing_fields,
            )
        except PredicateError as exc:
            raise RoutingConfigError(f"Category '{category.id}' eligible_when failed: {exc}") from exc
        missing = _missing_mask(observables, category.block_if_missing, n_events)
        block_mask = active & (assigned == "unassigned") & eligible & missing
        if np.any(block_mask):
            blocked[block_mask] = True
            reasons[block_mask] = category.block_reason
        try:
            selected = evaluate_predicate(
                category.select_when,
                observables,
                n_events=n_events,
                missing_false_fields=missing_fields,
            )
        except PredicateError as exc:
            raise RoutingConfigError(f"Category '{category.id}' select_when failed: {exc}") from exc
        claim_mask = active & ~blocked & (assigned == "unassigned") & eligible & selected
        if np.any(claim_mask):
            assigned[claim_mask] = category.id
            labels[claim_mask] = category.label
            reasons[claim_mask] = category.reason

    final_blocked = active & (assigned == "unassigned") & blocked
    assigned[final_blocked] = "blocked_missing_input"
    labels[final_blocked] = "blocked_missing_input"
    empty_block_reasons = final_blocked & (reasons == "")
    reasons[empty_block_reasons] = "blocked by missing classifier or derived input"
    unassigned_active = active & (assigned == "unassigned")
    reasons[unassigned_active] = "no category matched after full priority scan"
    return RoutingResult(
        assigned_category=assigned.astype(str),
        assignment_reason=reasons.astype(str),
        assignment_blocked=blocked.astype(bool),
        category_label=labels.astype(str),
    )
