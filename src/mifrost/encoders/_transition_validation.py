"""Planner-neutral validation for transition-only public lanes."""

from __future__ import annotations

from collections.abc import Iterable

from .types import BatchParam


_STR_BYTES = (str, bytes, bytearray)


def _payload_has_values(value: object, *, history: bool = False) -> bool:
    if value is None:
        return False
    if isinstance(value, BatchParam):
        if value.kind == "none":
            return False
        return _payload_has_values(value.value, history=history)
    if history and isinstance(value, tuple) and len(value) == 2:
        return _payload_has_values(value[1], history=False)
    if isinstance(value, _STR_BYTES):
        return bool(value)
    if isinstance(value, Iterable):
        return any(_payload_has_values(item, history=history) for item in value)
    return True


def require_single_successor(successor: object) -> None:
    if successor is None:
        raise ValueError("successor must be provided for transition encoding")


def require_batch_successors(successors: object) -> None:
    if successors is None or (
        isinstance(successors, BatchParam) and successors.kind == "none"
    ):
        raise ValueError("successors must be provided for transition batch encoding")


def validate_single_unsupported_lanes(
    *,
    actions: object,
    history_subgoals: object,
    history_max_steps: int | None,
) -> None:
    if _payload_has_values(actions):
        raise ValueError("Transition encoders do not support explicit action payloads")
    if (
        _payload_has_values(history_subgoals, history=True)
        or history_max_steps is not None
    ):
        raise ValueError("Transition encoders do not support history_subgoals payloads")


def validate_batch_unsupported_lanes(
    *,
    actions: object,
    history_subgoals: object,
    history_max_steps: int | None,
) -> None:
    if _payload_has_values(actions):
        raise ValueError(
            "Transition batch encoding does not support explicit action payloads"
        )
    if (
        _payload_has_values(history_subgoals, history=True)
        or history_max_steps is not None
    ):
        raise ValueError(
            "Transition batch encoding does not support history_subgoals payloads"
        )


__all__ = [
    "require_batch_successors",
    "require_single_successor",
    "validate_batch_unsupported_lanes",
    "validate_single_unsupported_lanes",
]
