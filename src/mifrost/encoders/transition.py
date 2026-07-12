from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Mapping

from torch_geometric.data import HeteroData

from .._core import (
    DEFAULT_LGAN_RR_EDGE_POS,
    DEFAULT_LGAN_TN_EDGE_POS,
    DEFAULT_LGAN_NN_EDGE_POS,
    DEFAULT_SYMBOL_TYPE_ID,
    SuccessorEncoderConfig,
    SuccessorEncoderMode,
    SuccessorHGraphEncoderEngine,
    TransitionStreamEncoder as _TransitionStreamEncoder,
    BatchEncoding,
)
from dataclasses import dataclass

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
from ._batch_contract import prepare_core_batch_inputs
from .common import (
    _advanced_state,
    _split_goals,
)
from ._lane_specs import (
    TRANSITION_LANE_SPEC,
    require_batch_payload,
    require_single_payload,
    validate_batch_optional_payloads,
    validate_single_optional_payloads,
)
from .hgraph import HGraphEncoder
from .types import (
    DomainInput,
    HeteroEncoding,
    HistorySubgoalInput,
    StateInput,
    default_goals_from_state,
)


class _TransitionEncoderBase(HGraphEncoder):
    """Shared successor encoder base.

    Transition encoders reuse the horizon-style LGAN contract: LGAN anchors are
    state candidates from the one-step successor structure. They do not accept
    `lgan_anchor_sources`.
    """

    def __init__(
        self,
        domain: DomainInput,
        *,
        successor_mode: SuccessorEncoderMode,
        successor_suffix: str,
        include_successor_goal_satisfaction: bool = False,
        symbol_type_id: str = DEFAULT_SYMBOL_TYPE_ID,
        ignore_actions: bool = True,
        add_nullary_predicates: bool = False,
        include_lgan_edges: bool = False,
        include_static: bool = True,
        include_empty_edge_types: bool = True,
        export_node_names: bool = True,
        max_goal_level: int = 0,
        support_literals: bool = False,
        goal_derivations: Iterable[Any] | None = None,
        nullary_object_name: str = "![nullary_symbol]!",
        lgan_tn_edge_pos: str = DEFAULT_LGAN_TN_EDGE_POS,
        lgan_nn_edge_pos: str = DEFAULT_LGAN_NN_EDGE_POS,
        lgan_rr_edge_pos: str = DEFAULT_LGAN_RR_EDGE_POS,
    ) -> None:
        """Create a transition encoder.

        `include_lgan_edges` uses successor-state candidates as anchors. Unlike
        `HGraphEncoder`, this lane does not take `lgan_anchor_sources`.
        """
        super().__init__(
            domain,
            symbol_type_id=symbol_type_id,
            ignore_actions=ignore_actions,
            add_nullary_predicates=add_nullary_predicates,
            include_lgan_edges=include_lgan_edges,
            include_static=include_static,
            include_empty_edge_types=include_empty_edge_types,
            export_node_names=export_node_names,
            max_goal_level=max_goal_level,
            support_literals=support_literals,
            goal_derivations=goal_derivations,
            nullary_object_name=nullary_object_name,
            lgan_tn_edge_pos=lgan_tn_edge_pos,
            lgan_nn_edge_pos=lgan_nn_edge_pos,
            lgan_rr_edge_pos=lgan_rr_edge_pos,
            _config_cls=SuccessorEncoderConfig,
            _engine_cls=SuccessorHGraphEncoderEngine,
            successor_mode=successor_mode,
            successor_suffix=successor_suffix,
            include_successor_goal_satisfaction=include_successor_goal_satisfaction,
        )

    @property
    def engine(self) -> SuccessorHGraphEncoderEngine:
        """Expose the underlying successor encoder engine."""
        return self._engine

    def _accepted_kwargs(self) -> set[str]:
        """Accept successor/successors kwargs in generic base API calls."""
        return super()._accepted_kwargs() | {"successor", "successors"}

    def _encode(
        self,
        state: StateInput,
        *,
        goals: GoalBatchInput = None,
        actions: ActionBatchInput = None,
        subgoal_layers: SubgoalLayersInput = None,
        successor: StateInput | None = None,
        history_subgoals: HistorySubgoalInput | None = None,
        history_max_steps: int | None = None,
        **kwargs,
    ) -> BatchEncoding:
        """Encode one ``state -> successor`` transition."""
        require_single_payload(TRANSITION_LANE_SPEC, successor)
        _ = kwargs
        validate_single_optional_payloads(
            TRANSITION_LANE_SPEC,
            actions=actions,
            history_subgoals=history_subgoals,
            history_max_steps=history_max_steps,
        )
        adv_state = _advanced_state(state)
        adv_successor = _advanced_state(successor)
        if goals is None:
            goals = default_goals_from_state(state)
        inputs = _split_goals(goals, subgoal_layers)
        return self._engine.encode(adv_state, adv_successor, inputs)

    def encode(
        self,
        state: StateInput,
        *,
        goals: GoalBatchInput = None,
        actions: ActionBatchInput = None,
        subgoal_layers: SubgoalLayersInput = None,
        successor: StateInput | None = None,
        history_subgoals: HistorySubgoalInput | None = None,
        history_max_steps: int | None = None,
        include_metadata: bool = True,
        **kwargs,
    ) -> BatchEncoding:
        """Encode one ``state -> successor`` transition into native ``BatchEncoding``."""
        return super().encode(
            state,
            goals=goals,
            actions=actions,
            subgoal_layers=subgoal_layers,
            successor=successor,
            history_subgoals=history_subgoals,
            history_max_steps=history_max_steps,
            include_metadata=include_metadata,
            **kwargs,
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
    ) -> HeteroEncoding:
        """Encode one or many ``state -> successor`` transitions into native ``BatchEncoding``."""
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
        **kwargs,
    ) -> HeteroEncoding:
        """Encode many aligned ``states -> successors`` transitions."""
        require_batch_payload(TRANSITION_LANE_SPEC, successors)
        validate_batch_optional_payloads(
            TRANSITION_LANE_SPEC,
            actions=actions,
            history_subgoals=history_subgoals,
            history_max_steps=history_max_steps,
        )
        _ = kwargs
        inputs = prepare_core_batch_inputs(
            states,
            goals=goals,
            subgoal_layers=subgoal_layers,
            successors=successors,
        )
        return self._engine.encode_batch(
            self.__class__.__name__,
            inputs.states,
            inputs.successors,
            inputs.goals,
            None,
            inputs.subgoal_layers,
            None,
            history_max_steps,
        )

    def stream(self) -> "_TransitionEncoderStream":
        """Create a stream wrapper for transition encoding."""
        return _TransitionEncoderStream(self)

    def draw(self, data: HeteroData, **kwargs) -> Any:
        """Reuse generic heterogeneous graph drawing from ``HGraphEncoder``."""
        return HGraphEncoder.draw(self, data, **kwargs)


@dataclass
class _TransitionEncoderStream(StreamEncoderBase[HeteroData]):
    """Shared stream implementation for transition encoders."""

    _encoder: _TransitionEncoderBase

    def __post_init__(self) -> None:
        """Initialize an empty hetero builder for streaming."""
        self._stream = _TransitionStreamEncoder(self._encoder.engine)
        self._reset_builder()

    def append(
        self,
        current: StateInput,
        successor: StateInput,
        *,
        goals: GoalBatchInput = None,
        subgoal_layers: SubgoalLayersInput = None,
    ) -> int:
        """Append one transition graph to the stream."""
        adv_current = _advanced_state(current)
        adv_successor = _advanced_state(successor)
        if goals is None:
            goals = default_goals_from_state(current)
        inputs = _split_goals(goals, subgoal_layers)
        return self._coerce_stream_id(
            self._stream.append(adv_current, adv_successor, inputs)
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
        adv_current = _advanced_state(current)
        adv_successor = _advanced_state(successor)
        if goals is None:
            goals = default_goals_from_state(current)
        inputs = _split_goals(goals, subgoal_layers)
        self._stream.update(stream_id, adv_current, adv_successor, inputs)

    def _reset_builder(self) -> None:
        """Reset stream accumulation state."""
        self._stream.reset()


class TransitionHGraphEncoder(_TransitionEncoderBase):
    """Full successor encoder (encodes both source and successor structure)."""

    def __init__(
        self,
        domain: DomainInput,
        *,
        successor_suffix: str = "[suc]",
        **kwargs,
    ) -> None:
        super().__init__(
            domain,
            successor_mode=SuccessorEncoderMode.full,
            successor_suffix=successor_suffix,
            **kwargs,
        )

    def stream(self) -> "TransitionHGraphEncoderStream":
        return TransitionHGraphEncoderStream(self)


class TransitionEffectsHGraphEncoder(_TransitionEncoderBase):
    """Delta/effects successor encoder (focuses on transition effects)."""

    def __init__(
        self,
        domain: DomainInput,
        *,
        successor_suffix: str = "",
        **kwargs,
    ) -> None:
        super().__init__(
            domain,
            successor_mode=SuccessorEncoderMode.delta,
            successor_suffix=successor_suffix,
            **kwargs,
        )

    def stream(self) -> "TransitionEffectsHGraphEncoderStream":
        return TransitionEffectsHGraphEncoderStream(self)


@dataclass
class TransitionHGraphEncoderStream(_TransitionEncoderStream):
    """Stream encoder for ``TransitionHGraphEncoder``."""

    _encoder: TransitionHGraphEncoder


@dataclass
class TransitionEffectsHGraphEncoderStream(_TransitionEncoderStream):
    """Stream encoder for ``TransitionEffectsHGraphEncoder``."""

    _encoder: TransitionEffectsHGraphEncoder


__all__ = [
    "TransitionHGraphEncoder",
    "TransitionEffectsHGraphEncoder",
    "TransitionHGraphEncoderStream",
    "TransitionEffectsHGraphEncoderStream",
]
