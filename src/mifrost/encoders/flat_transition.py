from __future__ import annotations

from typing import Any, Mapping

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
    _convert_batch_payload,
)
from .flat_horizon import FlatHorizonEncoder
from ._lane_specs import (
    TRANSITION_LANE_SPEC,
    require_batch_payload,
    require_single_payload,
    single_transition_dag,
    validate_batch_optional_payloads,
    validate_single_optional_payloads,
)
from .types import (
    HistorySubgoalInput,
    StateInput,
    is_state_input,
    to_advanced_state,
)


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
        require_single_payload(TRANSITION_LANE_SPEC, successor)
        validate_single_optional_payloads(
            TRANSITION_LANE_SPEC,
            actions=actions,
            history_subgoals=history_subgoals,
            history_max_steps=history_max_steps,
        )
        dag = single_transition_dag(current, successor)
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
        require_batch_payload(TRANSITION_LANE_SPEC, successors)
        validate_batch_optional_payloads(
            TRANSITION_LANE_SPEC,
            actions=actions,
            history_subgoals=history_subgoals,
            history_max_steps=history_max_steps,
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
        successors_list = parse_successors_batch_param(
            successors_for_batch,
            state_count=len(states_list),
        )
        dags = [
            single_transition_dag(current, successor)
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
