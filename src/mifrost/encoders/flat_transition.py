from __future__ import annotations

from typing import Any, Mapping

from ._action_contract import parse_flat_actions
from ._batch_contract import (
    parse_states_batch,
    parse_successors_batch_param,
)
from .base import (
    ActionBatchInput,
    ActionBatchParam,
    CollateSpecParam,
    GoalBatchInput,
    GoalBatchParam,
    HistorySubgoalsBatchParam,
    StateBatchInput,
    SubgoalLayersInput,
    SubgoalLayersBatchParam,
    SuccessorBatchParam,
)
from .common import (
    _advanced_state,
    _convert_batch_payload,
    _prepare_history_subgoals,
)
from .flat_horizon import FlatHorizonEncoder
from .types import (
    BatchParam,
    HistorySubgoalInput,
    StateInput,
    is_action_input,
    is_state_input,
    to_advanced_state,
)


def _single_transition_dag(
    current: StateInput,
    successor: StateInput,
):
    from .._core import TransitionDAG

    adv_current = _advanced_state(current)
    dag = TransitionDAG(adv_current)
    dag.register_transition(
        adv_current,
        _advanced_state(successor),
        None,
    )
    return dag


def _action_payload_has_non_empty_entries(value: object) -> bool:
    if value is None:
        return False
    if isinstance(value, BatchParam):
        if value.kind == "none":
            return False
        if value.kind == "shared":
            return _action_payload_has_non_empty_entries(value.value)
        if value.kind == "separate":
            if value.value is None:
                return False
            return any(
                _action_payload_has_non_empty_entries(entry)
                for entry in value.value
                if entry is not None
            )
    if is_action_input(value):
        return True
    if isinstance(value, (str, bytes, bytearray)):
        return bool(value)
    if isinstance(value, tuple):
        return any(_action_payload_has_non_empty_entries(item) for item in value)
    if isinstance(value, list):
        if not value:
            return False
        if any(is_action_input(item) for item in value if item is not None):
            return True
        return any(
            _action_payload_has_non_empty_entries(item)
            for item in value
            if item is not None
        )
    return True


def _is_history_entry(value: object) -> bool:
    return isinstance(value, tuple) and len(value) == 2 and isinstance(value[0], int)


def _history_payload_has_non_empty_entries(value: object) -> bool:
    if value is None:
        return False
    if isinstance(value, BatchParam):
        if value.kind == "none":
            return False
        if value.kind == "shared":
            return _history_payload_has_non_empty_entries(value.value)
        if value.kind == "separate":
            if value.value is None:
                return False
            return any(
                _history_payload_has_non_empty_entries(entry)
                for entry in value.value
                if entry is not None
            )
    if _is_history_entry(value):
        return bool(value[1])
    if isinstance(value, (str, bytes, bytearray)):
        return bool(value)
    if isinstance(value, tuple):
        return any(_history_payload_has_non_empty_entries(item) for item in value)
    if isinstance(value, list):
        if not value:
            return False
        if all(_is_history_entry(item) for item in value if item is not None):
            return any(bool(item[1]) for item in value if item is not None)
        return any(
            _history_payload_has_non_empty_entries(item)
            for item in value
            if item is not None
        )
    return True


class _FlatTransitionEncoderBase(FlatHorizonEncoder):
    """Shared successor wrapper backed by the flat horizon engine."""

    def _accepted_kwargs(self) -> set[str]:
        return {"successor", "successors", "history_subgoals", "history_max_steps"}

    def _encode(
        self,
        current: StateInput,
        *,
        goals: GoalBatchInput = None,
        actions: ActionBatchInput = None,
        subgoal_layers: SubgoalLayersInput = None,
        successor: StateInput | None = None,
        history_subgoals: HistorySubgoalInput | None = None,
        history_max_steps: int | None = None,
    ):
        if successor is None:
            raise ValueError("successor must be provided for transition encoding")
        action_list = parse_flat_actions(actions)
        history_list = _prepare_history_subgoals(history_subgoals)
        if action_list:
            raise ValueError(
                "Transition encoders do not support explicit action payloads"
            )
        if history_list or history_max_steps is not None:
            raise ValueError(
                "Transition encoders do not support history_subgoals payloads"
            )
        dag = _single_transition_dag(current, successor)
        return FlatHorizonEncoder._encode(
            self,
            current,
            goals=goals,
            actions=None,
            subgoal_layers=subgoal_layers,
            dag=dag,
            history_subgoals=None,
            history_max_steps=None,
        )

    def encode(
        self,
        current: StateInput,
        *,
        goals: GoalBatchInput = None,
        actions: ActionBatchInput = None,
        subgoal_layers: SubgoalLayersInput = None,
        successor: StateInput | None = None,
        history_subgoals: HistorySubgoalInput | None = None,
        history_max_steps: int | None = None,
        include_metadata: bool = True,
        **kwargs: object,
    ):
        return super().encode(
            current,
            goals=goals,
            actions=actions,
            subgoal_layers=subgoal_layers,
            successor=successor,
            history_subgoals=history_subgoals,
            history_max_steps=history_max_steps,
            include_metadata=include_metadata,
            **kwargs,
        )

    def _encode_batch(
        self,
        states: StateBatchInput,
        *,
        goals: GoalBatchParam = None,
        actions: ActionBatchParam = None,
        subgoal_layers: SubgoalLayersBatchParam = None,
        successors: SuccessorBatchParam = None,
        history_subgoals: HistorySubgoalsBatchParam = None,
        history_max_steps: int | None = None,
    ):
        if successors is None:
            raise ValueError(
                "successors must be provided for transition batch encoding"
            )
        states_for_batch = _convert_batch_payload(
            states,
            is_leaf=is_state_input,
            convert_leaf=to_advanced_state,
        )
        successors_for_batch = _convert_batch_payload(
            successors,
            is_leaf=is_state_input,
            convert_leaf=to_advanced_state,
        )
        states_list = parse_states_batch(states_for_batch)
        if _action_payload_has_non_empty_entries(actions):
            raise ValueError(
                "Transition batch encoding does not support explicit action payloads"
            )
        if (
            _history_payload_has_non_empty_entries(history_subgoals)
            or history_max_steps is not None
        ):
            raise ValueError(
                "Transition batch encoding does not support history_subgoals payloads"
            )
        successors_list = parse_successors_batch_param(
            successors_for_batch,
            state_count=len(states_list),
        )
        dags = [
            _single_transition_dag(current, successor)
            for current, successor in zip(states_list, successors_list, strict=True)
        ]
        return FlatHorizonEncoder._encode_batch(
            self,
            states_list,
            dags=dags,
            goals=goals,
            subgoal_layers=subgoal_layers,
            actions=None,
            history_subgoals=None,
            history_max_steps=None,
        )

    def encode_batch(
        self,
        states: StateBatchInput,
        *,
        goals: GoalBatchParam = None,
        actions: ActionBatchParam = None,
        subgoal_layers: SubgoalLayersBatchParam = None,
        successors: SuccessorBatchParam = None,
        history_subgoals: HistorySubgoalsBatchParam = None,
        history_max_steps: int | None = None,
        batch_attrs: Mapping[str, Any] | None = None,
        collate_spec: CollateSpecParam = None,
        include_metadata: bool = True,
        **kwargs: object,
    ):
        return super().encode_batch(
            states,
            goals=goals,
            actions=actions,
            subgoal_layers=subgoal_layers,
            successors=successors,
            history_subgoals=history_subgoals,
            history_max_steps=history_max_steps,
            batch_attrs=batch_attrs,
            collate_spec=collate_spec,
            include_metadata=include_metadata,
            **kwargs,
        )


class FlatTransitionEncoder(_FlatTransitionEncoderBase):
    """Flat full successor encoder backed by state-target carrier rows."""

    def __init__(self, domain, **kwargs: object) -> None:
        super().__init__(
            domain,
            transition_mode="full",
            enable_parent_relation=False,
            enable_sibling_relation=False,
            enable_cousin_relation=False,
            exclude_root_candidate=True,
            ignore_actions=True,
            **kwargs,
        )


class FlatTransitionEffectsEncoder(_FlatTransitionEncoderBase):
    """Flat delta/effects successor encoder backed by state-target carrier rows."""

    def __init__(self, domain, **kwargs: object) -> None:
        super().__init__(
            domain,
            transition_mode="delta",
            enable_parent_relation=False,
            enable_sibling_relation=False,
            enable_cousin_relation=False,
            exclude_root_candidate=True,
            ignore_actions=True,
            **kwargs,
        )


__all__ = [
    "FlatTransitionEncoder",
    "FlatTransitionEffectsEncoder",
]
