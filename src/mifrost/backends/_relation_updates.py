"""Planner-neutral validation for public relation-schema replacement."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def relation_arities(value: Any) -> dict[str, int]:
    if not isinstance(value, Mapping):
        raise TypeError(
            "update_relations expects mifrost.RelationDict or a mapping[str, int]"
        )
    result: dict[str, int] = {}
    for name, arity in value.items():
        if not isinstance(name, str):
            raise TypeError("relation names must be strings")
        if not isinstance(arity, int) or isinstance(arity, bool):
            raise TypeError("relation arities must be integers")
        if not name:
            raise ValueError("relation name must not be empty")
        if arity < 0:
            raise ValueError("relation arity must be non-negative")
        result[name] = arity
    return result


__all__ = ["relation_arities"]
