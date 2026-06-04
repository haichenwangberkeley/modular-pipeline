from __future__ import annotations

from typing import Any

import numpy as np


COMPARISON_OPERATORS = {"==", "!=", ">", ">=", "<", "<="}


class PredicateError(ValueError):
    pass


def predicate_fields(predicate: dict[str, Any] | None) -> set[str]:
    if predicate is None:
        return set()
    if not isinstance(predicate, dict):
        raise PredicateError(f"Predicate must be a mapping, got {type(predicate).__name__}")
    if predicate.get("always") is True:
        return set()
    if "all" in predicate or "any" in predicate:
        key = "all" if "all" in predicate else "any"
        children = predicate[key]
        if not isinstance(children, list) or not children:
            raise PredicateError(f"Predicate '{key}' must contain a non-empty list")
        fields: set[str] = set()
        for child in children:
            fields.update(predicate_fields(child))
        return fields
    if "not" in predicate:
        return predicate_fields(predicate["not"])
    if "finite" in predicate:
        finite = predicate["finite"]
        if not isinstance(finite, dict) or not isinstance(finite.get("field"), str):
            raise PredicateError("Predicate 'finite' must be a mapping with a string 'field'")
        return {finite["field"]}
    if "field" in predicate:
        field = predicate.get("field")
        op = predicate.get("op")
        if not isinstance(field, str):
            raise PredicateError("Comparison predicate requires string 'field'")
        if op not in COMPARISON_OPERATORS:
            raise PredicateError(f"Unsupported comparison operator '{op}'")
        if "value" not in predicate:
            raise PredicateError(f"Comparison predicate for '{field}' is missing 'value'")
        return {field}
    raise PredicateError(f"Unknown predicate shape: {predicate}")


def validate_predicate(predicate: dict[str, Any], *, context: str) -> None:
    try:
        predicate_fields(predicate)
    except PredicateError as exc:
        raise PredicateError(f"{context}: {exc}") from exc


def _missing_false(name: str, n_events: int, missing_false_fields: set[str], observables: dict[str, np.ndarray]) -> np.ndarray | None:
    if name in observables:
        return None
    if name in missing_false_fields:
        return np.zeros(n_events, dtype=bool)
    raise PredicateError(f"Missing observable '{name}'")


def evaluate_predicate(
    predicate: dict[str, Any],
    observables: dict[str, np.ndarray],
    *,
    n_events: int,
    missing_false_fields: set[str] | None = None,
) -> np.ndarray:
    missing_false_fields = missing_false_fields or set()
    if predicate.get("always") is True:
        return np.ones(n_events, dtype=bool)
    if "all" in predicate:
        children = [evaluate_predicate(child, observables, n_events=n_events, missing_false_fields=missing_false_fields) for child in predicate["all"]]
        return np.logical_and.reduce(children) if children else np.ones(n_events, dtype=bool)
    if "any" in predicate:
        children = [evaluate_predicate(child, observables, n_events=n_events, missing_false_fields=missing_false_fields) for child in predicate["any"]]
        return np.logical_or.reduce(children) if children else np.zeros(n_events, dtype=bool)
    if "not" in predicate:
        return ~evaluate_predicate(predicate["not"], observables, n_events=n_events, missing_false_fields=missing_false_fields)
    if "finite" in predicate:
        field = predicate["finite"]["field"]
        missing = _missing_false(field, n_events, missing_false_fields, observables)
        if missing is not None:
            return missing
        return np.isfinite(np.asarray(observables[field], dtype=float))
    field = predicate["field"]
    missing = _missing_false(field, n_events, missing_false_fields, observables)
    if missing is not None:
        return missing
    values = np.asarray(observables[field])
    target = predicate["value"]
    op = predicate["op"]
    if op == "==":
        return values == target
    if op == "!=":
        return values != target
    numeric_values = values.astype(float)
    numeric_target = float(target)
    if op == ">":
        return numeric_values > numeric_target
    if op == ">=":
        return numeric_values >= numeric_target
    if op == "<":
        return numeric_values < numeric_target
    if op == "<=":
        return numeric_values <= numeric_target
    raise PredicateError(f"Unsupported comparison operator '{op}'")
