"""Backend-neutral validation for flat relation input lanes."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, TypeGuard


def _is_sequence_like(value: object) -> TypeGuard[Sequence[Any]]:
    return isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    )


def _subgoal_layers_looks_like_state_payload(value: object) -> bool:
    return bool(value) and _is_sequence_like(value) and _is_sequence_like(value[0])


def validate_subgoal_layers_state_payload(
    subgoal_layers: object,
    *,
    state_index: int,
    max_goal_level: int,
) -> None:
    if subgoal_layers is None:
        return
    if not _is_sequence_like(subgoal_layers):
        raise TypeError(
            f"subgoal_layers entry at state index {state_index} must be an "
            "iterable of goal-literal layers or None"
        )

    layers = list(subgoal_layers)
    for layer_index, layer in enumerate(layers):
        if not _is_sequence_like(layer):
            raise TypeError(
                f"subgoal_layers entry at state index {state_index} layer at "
                f"position {layer_index} must be an iterable of goal literals"
            )

    if len(layers) > max_goal_level:
        if layers and all(len(layer) == 1 for layer in layers):
            positions = ", ".join(str(idx) for idx in range(len(layers)))
            raise ValueError(
                f"subgoal_layers entry at state index {state_index} looks like "
                f"singleton layers at positions {positions}; this flat encoder "
                f"supports at most {max_goal_level} subgoal layer(s). If you "
                "intended one layer with multiple literals, use "
                "[[lit1, lit2, ...]] instead of [[lit1], [lit2], ...]."
            )
        raise ValueError(
            f"subgoal_layers entry at state index {state_index} has "
            f"{len(layers)} layers, but this flat encoder supports at most "
            f"{max_goal_level} subgoal layer(s). The first unsupported layer "
            f"is at position {max_goal_level}."
        )


def validate_subgoal_layers_batch_payload(
    subgoal_layers: object,
    *,
    state_count: int,
    max_goal_level: int,
) -> None:
    if subgoal_layers is None:
        return

    kind = getattr(subgoal_layers, "kind", None)
    if kind in {"none", "shared", "separate"} and hasattr(subgoal_layers, "value"):
        value = getattr(subgoal_layers, "value")
        if kind == "none":
            return
        if kind == "shared":
            validate_subgoal_layers_state_payload(
                value, state_index=0, max_goal_level=max_goal_level
            )
            return
        if not _is_sequence_like(value):
            raise TypeError("BatchParam(separate) value must be a sequence")
        if len(value) != state_count:
            raise ValueError("subgoal_layers length must match states length")
        for state_index, state_payload in enumerate(value):
            validate_subgoal_layers_state_payload(
                state_payload,
                state_index=state_index,
                max_goal_level=max_goal_level,
            )
        return

    if not _is_sequence_like(subgoal_layers):
        validate_subgoal_layers_state_payload(
            subgoal_layers, state_index=0, max_goal_level=max_goal_level
        )
        return

    outer = list(subgoal_layers)
    per_state_like = any(
        entry is None or _subgoal_layers_looks_like_state_payload(entry)
        for entry in outer
    )
    if per_state_like:
        if len(outer) != state_count:
            raise ValueError("subgoal_layers length must match states length")
        for state_index, state_payload in enumerate(outer):
            validate_subgoal_layers_state_payload(
                state_payload,
                state_index=state_index,
                max_goal_level=max_goal_level,
            )
        return

    validate_subgoal_layers_state_payload(
        outer, state_index=0, max_goal_level=max_goal_level
    )


__all__ = [
    "validate_subgoal_layers_batch_payload",
    "validate_subgoal_layers_state_payload",
]
