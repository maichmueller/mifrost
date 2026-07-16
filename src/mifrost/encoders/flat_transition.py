from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

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
from .flat_horizon import FlatHorizonEncoder
from ._horizon_validation import (
    require_transition_successor,
    require_transition_successors,
    validate_transition_batch_unsupported_lanes,
    validate_transition_single_unsupported_lanes,
)
from .types import (
    HistorySubgoalInput,
    StateInput,
)


class _FlatTransitionEncoderBase(FlatHorizonEncoder):
    """Shared flat transition wrapper.

    This lane builds a one-step `TransitionDAG` from `current` and `successor`
    and then reuses `FlatHorizonEncoder`.
    """

    def _accepted_kwargs(self) -> set[str]:
        return {
            "successor",
            "successors",
            "history_subgoals",
            "history_max_steps",
        }

    def _single_transition_payload(
        self, current: StateInput, successor: StateInput
    ) -> Any:
        if self.backend != "pytyr":
            from ..backends.pymimir_lane_specs import single_transition_dag

            return single_transition_dag(current, successor)
        import rustworkx as rx

        dag = rx.PyDiGraph()
        root_index = dag.add_node(current)
        successor_index = dag.add_node(successor)
        dag.add_edge(root_index, successor_index, None)
        return dag

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
        require_transition_successor(successor)
        validate_transition_single_unsupported_lanes(
            actions=actions,
            history_subgoals=history_subgoals,
            history_max_steps=history_max_steps,
        )
        dag = self._single_transition_payload(current, successor)
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
        require_transition_successors(successors)
        validate_transition_batch_unsupported_lanes(
            actions=actions,
            history_subgoals=history_subgoals,
            history_max_steps=history_max_steps,
        )
        states_list, successors_list = self._transition_batch_values(states, successors)
        dags = [
            self._single_transition_payload(current, successor)
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

    def _transition_batch_values(
        self,
        states: StateBatchInput,
        successors: SuccessorBatchParam,
    ) -> tuple[list[Any], list[Any]]:
        if self.backend != "pytyr":
            from ._batch_contract import (
                parse_states_batch,
                parse_successors_batch_param,
                prepare_core_batch_inputs,
            )

            inputs = prepare_core_batch_inputs(states, successors=successors)
            states_list = parse_states_batch(inputs.states)
            successors_list = parse_successors_batch_param(
                inputs.successors,
                state_count=len(states_list),
            )
            return states_list, successors_list

        from ..backends.pytyr_flat import _batch_param, _is_state, _sequence

        states_list = (
            [states] if _is_state(states) else _sequence(states, field="states")
        )
        kind, payload = _batch_param(successors)
        if kind == "none" or payload is None:
            raise ValueError(
                "successors must be provided for transition batch encoding"
            )
        if kind == "shared" or _is_state(payload):
            successors_list = [payload] * len(states_list)
        else:
            successors_list = _sequence(payload, field="successors")
            if len(successors_list) != len(states_list):
                raise ValueError("successors length must match states length")
        for index, successor in enumerate(successors_list):
            if not _is_state(successor):
                raise TypeError(f"successors entry at index {index} has invalid type")
        return states_list, successors_list

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
        dag = self._encoder._single_transition_payload(current, successor)
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
        dag = self._encoder._single_transition_payload(current, successor)
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
