from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import cast

from .types import GroundActionInput, is_action_input

HGRAPH_NESTED_ACTIONS_ERROR = (
    "Nested/tuple action payloads are not supported by HGraphEncoder; "
    "use HorizonEncoder for IW lookahead."
)


_STR_BYTES = (str, bytes, bytearray)


def _is_sequence_like_but_not_str_bytes(value: object) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, _STR_BYTES)


def _is_iterable_but_not_str_bytes(value: object) -> bool:
    return isinstance(value, Iterable) and not isinstance(value, _STR_BYTES)


def validate_no_nested_actions(
    flat_actions: Iterable[GroundActionInput],
) -> list[GroundActionInput]:
    materialized = list(flat_actions)
    for action in materialized:
        if _is_sequence_like_but_not_str_bytes(action) and not is_action_input(action):
            raise ValueError(HGRAPH_NESTED_ACTIONS_ERROR)
    return cast(list[GroundActionInput], materialized)


@dataclass(frozen=True)
class BatchActionPlan:
    shared_actions: list[GroundActionInput] | None = None
    per_state_actions: list[list[GroundActionInput] | None] | None = None


def parse_flat_actions(
    actions: Iterable[GroundActionInput] | None,
) -> list[GroundActionInput]:
    if actions is None:
        return []
    return validate_no_nested_actions(actions)


def parse_actions_batch_plan(
    actions: (
        Iterable[GroundActionInput]
        | Sequence[Iterable[GroundActionInput] | None]
        | None
    ),
    *,
    state_count: int,
) -> BatchActionPlan:
    if actions is None:
        return BatchActionPlan(shared_actions=[])

    if not _is_sequence_like_but_not_str_bytes(actions):
        return BatchActionPlan(shared_actions=validate_no_nested_actions(actions))

    outer = list(actions)
    per_state_like = any(
        item is None
        or (_is_iterable_but_not_str_bytes(item) and not is_action_input(item))
        for item in outer
    )

    if per_state_like:
        if len(outer) != state_count:
            raise ValueError("actions length must match states length")
        per_state: list[list[GroundActionInput] | None] = []
        for entry in outer:
            if entry is None:
                per_state.append(None)
                continue
            if is_action_input(entry):
                raise TypeError(
                    "per-state action entries must be iterable action collections or None"
                )
            if not _is_iterable_but_not_str_bytes(entry):
                raise TypeError(
                    "per-state action entries must be iterable action collections or None"
                )
            per_state.append(
                validate_no_nested_actions(cast(Iterable[GroundActionInput], entry))
            )
        return BatchActionPlan(per_state_actions=per_state)

    return BatchActionPlan(shared_actions=validate_no_nested_actions(outer))
