from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any, cast

import pymimir.advanced.formalism as af

from .._core import GoalInputs, TransitionDAG
from ._rustworkx_dag import RXStateDAG, _normalize_dag_leaf
from .common import (
    _advanced_state,
    _prepare_actions,
    _prepare_history_subgoals,
    _split_goals,
)
from .types import (
    BatchParam,
    AdvancedGroundLiteral,
    GoalLiteralInput,
    GroundActionInput,
    HistorySubgoalInput,
    StateInput,
    default_goals_from_state,
    is_action_input,
)

_STR_BYTES = (str, bytes, bytearray)


@dataclass(frozen=True)
class EncoderLaneSpec:
    supports_lgan: bool = True
    single_required_error: str | None = None
    batch_required_error: str | None = None
    single_action_error: str | None = None
    batch_action_error: str | None = None
    single_history_error: str | None = None
    batch_history_error: str | None = None
    lgan_requires_actions: bool = False

    def allows_lgan(self, *, include_actions: bool) -> bool:
        if not self.supports_lgan:
            return False
        if not self.lgan_requires_actions:
            return True
        return include_actions


@dataclass(frozen=True)
class PreparedLaneOptionalPayloads:
    actions: list[af.GroundAction]
    history_subgoals: list[tuple[int, list[AdvancedGroundLiteral]]]


HGRAPH_BENCH_SPEC = EncoderLaneSpec(
    lgan_requires_actions=True,
)

HORIZON_LANE_SPEC = EncoderLaneSpec(
    single_action_error="HorizonEncoder does not support explicit action payloads",
    batch_action_error="Horizon batch encoding does not support explicit action payloads",
    single_history_error="HorizonEncoder does not support history_subgoals payloads",
    batch_history_error=(
        "Horizon batch encoding does not support history_subgoals payloads"
    ),
)

FLAT_HORIZON_LANE_SPEC = EncoderLaneSpec(
    single_action_error="FlatHorizonEncoder does not support explicit action payloads",
    batch_action_error=(
        "FlatHorizonEncoder batch encoding does not support explicit action payloads"
    ),
    single_history_error="FlatHorizonEncoder does not support history_subgoals payloads",
    batch_history_error=(
        "FlatHorizonEncoder batch encoding does not support history_subgoals payloads"
    ),
)

TRANSITION_LANE_SPEC = EncoderLaneSpec(
    supports_lgan=False,
    single_required_error="successor must be provided for transition encoding",
    batch_required_error="successors must be provided for transition batch encoding",
    single_action_error="Transition encoders do not support explicit action payloads",
    batch_action_error=(
        "Transition batch encoding does not support explicit action payloads"
    ),
    single_history_error="Transition encoders do not support history_subgoals payloads",
    batch_history_error=(
        "Transition batch encoding does not support history_subgoals payloads"
    ),
)


def _is_history_entry(value: object) -> bool:
    return isinstance(value, tuple) and len(value) == 2 and isinstance(value[0], int)


def _payload_has_non_empty_entries(
    value: object,
    *,
    is_leaf,
    is_entry=None,
) -> bool:
    if value is None:
        return False
    if isinstance(value, BatchParam):
        if value.kind == "none":
            return False
        if value.kind == "shared":
            return _payload_has_non_empty_entries(
                value.value,
                is_leaf=is_leaf,
                is_entry=is_entry,
            )
        if value.kind == "separate":
            if value.value is None:
                return False
            return any(
                _payload_has_non_empty_entries(
                    entry,
                    is_leaf=is_leaf,
                    is_entry=is_entry,
                )
                for entry in value.value
                if entry is not None
            )
    if is_entry is not None and is_entry(value):
        return bool(cast(tuple[Any, Any], value)[1])
    if is_leaf(value):
        return True
    if isinstance(value, _STR_BYTES):
        return bool(value)
    if isinstance(value, tuple):
        return any(
            _payload_has_non_empty_entries(
                item,
                is_leaf=is_leaf,
                is_entry=is_entry,
            )
            for item in value
        )
    if isinstance(value, list):
        if not value:
            return False
        if is_entry is not None and all(
            is_entry(item) for item in value if item is not None
        ):
            return any(bool(item[1]) for item in value if item is not None)
        return any(
            _payload_has_non_empty_entries(
                item,
                is_leaf=is_leaf,
                is_entry=is_entry,
            )
            for item in value
            if item is not None
        )
    if isinstance(value, Sequence) and not isinstance(value, _STR_BYTES):
        return any(
            _payload_has_non_empty_entries(
                item,
                is_leaf=is_leaf,
                is_entry=is_entry,
            )
            for item in value
            if item is not None
        )
    if isinstance(value, Iterable) and not isinstance(value, _STR_BYTES):
        return any(
            _payload_has_non_empty_entries(
                item,
                is_leaf=is_leaf,
                is_entry=is_entry,
            )
            for item in value
            if item is not None
        )
    return True


def batch_actions_have_values(actions: object) -> bool:
    return _payload_has_non_empty_entries(actions, is_leaf=is_action_input)


def batch_history_has_values(history_subgoals: object) -> bool:
    return _payload_has_non_empty_entries(
        history_subgoals,
        is_leaf=lambda _: False,
        is_entry=_is_history_entry,
    )


def require_single_payload(spec: EncoderLaneSpec, value: object) -> None:
    if value is None:
        if spec.single_required_error is None:
            raise ValueError("Required payload missing")
        raise ValueError(spec.single_required_error)


def require_batch_payload(spec: EncoderLaneSpec, value: object) -> None:
    if value is None:
        if spec.batch_required_error is None:
            raise ValueError("Required batch payload missing")
        raise ValueError(spec.batch_required_error)


def prepare_optional_payloads(
    *,
    actions: Iterable[GroundActionInput] | None,
    history_subgoals: HistorySubgoalInput | None,
) -> PreparedLaneOptionalPayloads:
    return PreparedLaneOptionalPayloads(
        actions=_prepare_actions(actions),
        history_subgoals=_prepare_history_subgoals(history_subgoals),
    )


def validate_single_optional_payloads(
    spec: EncoderLaneSpec,
    *,
    actions: Iterable[GroundActionInput] | None,
    history_subgoals: HistorySubgoalInput | None,
    history_max_steps: int | None,
) -> tuple[list[af.GroundAction], list[tuple[int, list[AdvancedGroundLiteral]]]]:
    payloads = prepare_optional_payloads(
        actions=actions,
        history_subgoals=history_subgoals,
    )
    action_list = payloads.actions
    history_list = payloads.history_subgoals
    if action_list and spec.single_action_error is not None:
        raise ValueError(spec.single_action_error)
    if (
        history_list or history_max_steps is not None
    ) and spec.single_history_error is not None:
        raise ValueError(spec.single_history_error)
    return action_list, history_list


def validate_batch_optional_payloads(
    spec: EncoderLaneSpec,
    *,
    actions: object,
    history_subgoals: object,
    history_max_steps: int | None,
) -> None:
    if batch_actions_have_values(actions) and spec.batch_action_error is not None:
        raise ValueError(spec.batch_action_error)
    if (
        batch_history_has_values(history_subgoals) or history_max_steps is not None
    ) and spec.batch_history_error is not None:
        raise ValueError(spec.batch_history_error)


def ensure_transition_dag(
    root: StateInput,
    dag: TransitionDAG | RXStateDAG | None,
) -> TransitionDAG:
    adv_root = _advanced_state(root)
    if dag is not None:
        normalized = _normalize_dag_leaf(dag)
        if isinstance(normalized, TransitionDAG):
            if normalized.root().get_index() != adv_root.get_index():
                raise ValueError("dag root must match root state")
            return normalized
        raise TypeError(
            "dag must be a TransitionDAG, rustworkx.PyDiGraph, or None, "
            f"got {type(dag)!r}"
        )
    return TransitionDAG(adv_root)


def prepare_goal_inputs(
    root: StateInput,
    goals: Iterable[GoalLiteralInput] | None,
    subgoal_layers: Iterable[Iterable[GoalLiteralInput]] | None,
) -> GoalInputs:
    if goals is None:
        goals = default_goals_from_state(root)
    return _split_goals(goals, subgoal_layers)


def single_transition_dag(
    current: StateInput,
    successor: StateInput,
) -> TransitionDAG:
    adv_current = _advanced_state(current)
    dag = TransitionDAG(adv_current)
    dag.register_transition(
        adv_current,
        _advanced_state(successor),
        None,
    )
    return dag
