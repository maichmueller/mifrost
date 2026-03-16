from __future__ import annotations

from dataclasses import dataclass
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
    StreamEncoderBase,
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
    """Shared flat transition wrapper.

    This lane builds a one-step `TransitionDAG` from `current` and `successor`
    and then reuses `FlatHorizonEncoder`.
    """

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
        **kwargs,
    ):
        """Encode one current/successor pair."""
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
        **kwargs,
    ):
        """Encode many current/successor pairs into one flat batch."""
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

    def stream(self) -> "_FlatTransitionEncoderStream":
        """Return a mutable stream backed by flat horizon streaming."""
        return _FlatTransitionEncoderStream(self)


@dataclass
class _FlatTransitionEncoderStream(StreamEncoderBase["FlatRelationData"]):
    """Mutable stream for flat transition encoders."""

    _encoder: _FlatTransitionEncoderBase

    def __post_init__(self) -> None:
        self._stream = FlatHorizonEncoder.mutable_stream(self._encoder)
        self._reset_builder()

    def append(
        self,
        current: StateInput,
        successor: StateInput,
        *,
        goals: GoalBatchInput = None,
        subgoal_layers: SubgoalLayersInput = None,
    ) -> int:
        """Append one current/successor pair and return its stream id."""
        dag = single_transition_dag(current, successor)
        return self._coerce_stream_id(
            self._stream.append(
                current,
                dag=dag,
                goals=goals,
                subgoal_layers=subgoal_layers,
            )
        )

    def remove(self, stream_id: int) -> None:
        self._stream.remove(stream_id)

    def update(
        self,
        stream_id: int,
        current: StateInput,
        successor: StateInput,
        *,
        goals: GoalBatchInput = None,
        subgoal_layers: SubgoalLayersInput = None,
    ) -> None:
        """Replace one current/successor pair in place."""
        dag = single_transition_dag(current, successor)
        self._stream.update(
            stream_id,
            current,
            dag=dag,
            goals=goals,
            subgoal_layers=subgoal_layers,
        )

    def _reset_builder(self) -> None:
        self._stream._reset_builder()


class FlatTransitionEncoder(_FlatTransitionEncoderBase):
    """Encode full current-to-successor structure on the flat carrier."""

    def __init__(self, domain, **kwargs) -> None:
        """Create a full flat transition encoder.

        LGAN, when enabled, uses successor-state candidate rows from the
        underlying flat horizon lane.
        """
        kwargs.setdefault("root_policy", "exclude")
        super().__init__(
            domain,
            transition_mode="full",
            enable_parent_relation=False,
            enable_sibling_relation=False,
            enable_cousin_relation=False,
            ignore_actions=True,
            **kwargs,
        )

    def stream(self) -> "FlatTransitionEncoderStream":
        """Return a mutable stream for full flat transitions."""
        return FlatTransitionEncoderStream(self)


class FlatTransitionEffectsEncoder(_FlatTransitionEncoderBase):
    """Encode only changed successor structure on the flat carrier."""

    def __init__(self, domain, **kwargs) -> None:
        """Create a delta/effects flat transition encoder."""
        kwargs.setdefault("root_policy", "exclude")
        super().__init__(
            domain,
            transition_mode="delta",
            enable_parent_relation=False,
            enable_sibling_relation=False,
            enable_cousin_relation=False,
            ignore_actions=True,
            **kwargs,
        )

    def stream(self) -> "FlatTransitionEffectsEncoderStream":
        """Return a mutable stream for flat delta/effects transitions."""
        return FlatTransitionEffectsEncoderStream(self)


@dataclass
class FlatTransitionEncoderStream(_FlatTransitionEncoderStream):
    _encoder: FlatTransitionEncoder


@dataclass
class FlatTransitionEffectsEncoderStream(_FlatTransitionEncoderStream):
    _encoder: FlatTransitionEffectsEncoder


__all__ = [
    "FlatTransitionEncoder",
    "FlatTransitionEffectsEncoder",
    "FlatTransitionEncoderStream",
    "FlatTransitionEffectsEncoderStream",
]
