"""Planner-neutral validation for Horizon-only public lanes."""

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
        return _payload_has_values(value[1])
    if isinstance(value, _STR_BYTES):
        return bool(value)
    if isinstance(value, Iterable):
        return any(_payload_has_values(item, history=history) for item in value)
    return True


def validate_single_unsupported_lanes(
    *,
    actions: object,
    history_subgoals: object,
    history_max_steps: int | None,
) -> None:
    if _payload_has_values(actions):
        raise ValueError("HorizonEncoder does not support explicit action payloads")
    if (
        _payload_has_values(history_subgoals, history=True)
        or history_max_steps is not None
    ):
        raise ValueError("HorizonEncoder does not support history_subgoals payloads")


def validate_batch_unsupported_lanes(
    *,
    actions: object,
    history_subgoals: object,
    history_max_steps: int | None,
) -> None:
    if _payload_has_values(actions):
        raise ValueError(
            "Horizon batch encoding does not support explicit action payloads"
        )
    if (
        _payload_has_values(history_subgoals, history=True)
        or history_max_steps is not None
    ):
        raise ValueError(
            "Horizon batch encoding does not support history_subgoals payloads"
        )


__all__ = [
    "validate_batch_unsupported_lanes",
    "validate_single_unsupported_lanes",
]
