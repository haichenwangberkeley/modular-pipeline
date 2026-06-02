from __future__ import annotations

import pytest

from analysis.routing.config import RoutingConfigError, parse_routing_config
from analysis.routing.router import route_categories


def _payload(category: dict) -> dict:
    return {"routing": {"mode": "ordered_first_match"}, "categories": [category]}


def _category(**overrides) -> dict:
    category = {
        "id": "cat",
        "label": "Cat",
        "priority": 10,
        "required_inputs": ["x"],
        "select_when": {"all": [{"field": "x", "op": ">", "value": 1.0}]},
    }
    category.update(overrides)
    return category


def test_duplicate_category_id_is_rejected() -> None:
    payload = {"categories": [_category(), _category(priority=20)]}
    with pytest.raises(RoutingConfigError, match="Duplicate category id"):
        parse_routing_config(payload)


def test_duplicate_priority_is_rejected() -> None:
    payload = {"categories": [_category(), _category(id="other")]}
    with pytest.raises(RoutingConfigError, match="Duplicate category priority"):
        parse_routing_config(payload)


def test_unsupported_operator_is_rejected() -> None:
    with pytest.raises(RoutingConfigError, match="Unsupported comparison operator"):
        parse_routing_config(_payload(_category(select_when={"all": [{"field": "x", "op": "approx", "value": 1.0}]})))


def test_missing_select_when_is_rejected() -> None:
    category = _category()
    category.pop("select_when")
    with pytest.raises(RoutingConfigError, match="missing 'select_when'"):
        parse_routing_config(_payload(category))


def test_malformed_predicate_is_rejected() -> None:
    with pytest.raises(RoutingConfigError, match="must contain a non-empty list"):
        parse_routing_config(_payload(_category(select_when={"all": []})))


def test_invalid_block_if_missing_declaration_is_rejected() -> None:
    with pytest.raises(RoutingConfigError, match="block_if_missing entries must also appear"):
        parse_routing_config(_payload(_category(block_if_missing=["score"])))


def test_missing_required_observable_is_reported() -> None:
    config = parse_routing_config(_payload(_category()))
    with pytest.raises(RoutingConfigError, match="Missing required observable"):
        route_categories({"y": [1.0]}, config)


def test_nan_learned_score_blocks_when_declared() -> None:
    config = parse_routing_config(
        _payload(
            _category(
                id="score_bin",
                required_inputs=["x", "score"],
                eligible_when={"all": [{"field": "x", "op": ">", "value": 0}]},
                select_when={"all": [{"field": "score", "op": ">", "value": 0.5}]},
                block_if_missing=["score"],
                block_reason="score missing",
            )
        )
    )
    result = route_categories({"x": [1.0], "score": [float("nan")]}, config)
    assert result.assigned_category.tolist() == ["blocked_missing_input"]
    assert result.assignment_blocked.tolist() == [True]
    assert result.assignment_reason.tolist() == ["score missing"]
