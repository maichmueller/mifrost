from __future__ import annotations

from typing import Mapping

from torch_geometric.data import HeteroData

from .._core import (
    BatchBuilder,
    DEFAULT_LGAN_NN_EDGE_POS,
    DEFAULT_SYMBOL_TYPE_ID,
    GoalInputs,
    SuccessorEncoderConfig,
    SuccessorEncoderMode,
    SuccessorHGraphEncoderEngine,
)
from dataclasses import dataclass

from .base import (
    ActionBatchInput,
    EncoderBase,
    GoalBatchInput,
    StateBatchInput,
    StreamEncoderBase,
    SubgoalLayersInput,
)
from .common import _advanced_domain, _advanced_state, _parts_to_pyg, _split_goals
from .types import (
    DomainInput,
    StateInput,
    default_goals_from_state,
    is_state_input,
)


class _TransitionEncoderBase(EncoderBase[HeteroData]):
    """Shared implementation for transition-state encoders."""

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
        max_goal_level: int = 0,
        support_literals: bool = False,
        nullary_object_name: str = "![nullary_symbol]!",
        lgan_nn_edge_pos: str = DEFAULT_LGAN_NN_EDGE_POS,
    ) -> None:
        """Create a transition encoder C++ engine with the given mode/config."""
        config = SuccessorEncoderConfig()
        config.successor_mode = successor_mode
        config.successor_suffix = successor_suffix
        config.include_successor_goal_satisfaction = include_successor_goal_satisfaction
        config.symbol_type_id = symbol_type_id
        config.ignore_actions = ignore_actions
        config.add_nullary_predicates = add_nullary_predicates
        config.include_lgan_edges = include_lgan_edges
        config.include_static = include_static
        config.include_empty_edge_types = include_empty_edge_types
        config.max_goal_level = max_goal_level
        config.support_literals = support_literals
        config.nullary_object_name = nullary_object_name
        config.lgan_nn_edge_pos = lgan_nn_edge_pos
        self._engine = SuccessorHGraphEncoderEngine(_advanced_domain(domain), config)

    @property
    def engine(self) -> SuccessorHGraphEncoderEngine:
        """Expose the underlying successor encoder engine."""
        return self._engine

    def _accepted_kwargs(self) -> set[str]:
        """Accept successor/successors kwargs in generic base API calls."""
        return {"successor", "successors"}

    def encode_parts(
        self,
        state: StateInput,
        *,
        goals: GoalBatchInput = None,
        actions: ActionBatchInput = None,
        subgoal_layers: SubgoalLayersInput = None,
        successor: StateInput | None = None,
        **kwargs: object,
    ) -> Mapping[str, object]:
        """Encode one ``state -> successor`` transition into parts."""
        if successor is None:
            raise ValueError("successor must be provided for transition encoding")
        adv_state = _advanced_state(state)
        adv_successor = _advanced_state(successor)
        if goals is None:
            goals = default_goals_from_state(state)
        inputs, _ = _split_goals(goals, subgoal_layers)
        return self._engine.encode(adv_state, adv_successor, inputs)

    def encode_batch_parts(
        self,
        states: StateBatchInput,
        *,
        goals: GoalBatchInput = None,
        actions: ActionBatchInput = None,
        subgoal_layers: SubgoalLayersInput = None,
        successors: StateBatchInput | None = None,
        **kwargs: object,
    ) -> Mapping[str, object]:
        """Encode many aligned ``states -> successors`` transitions into parts."""
        if is_state_input(states):
            state_list = [states]
        else:
            state_list = list(states)

        if successors is None:
            raise ValueError(
                "successors must be provided for transition batch encoding"
            )
        if is_state_input(successors):
            succ_list = [successors]
        else:
            succ_list = list(successors)
        if len(succ_list) != len(state_list):
            raise ValueError("successors length must match states length")

        shared_inputs: GoalInputs | None = None
        if goals is not None:
            shared_inputs, _ = _split_goals(goals, subgoal_layers)

        builder = BatchBuilder()
        builder.set_graph_kind("hetero")
        for idx, state in enumerate(state_list):
            adv_state = _advanced_state(state)
            adv_successor = _advanced_state(succ_list[idx])
            if goals is None:
                goals_for_state = default_goals_from_state(state)
                inputs, _ = _split_goals(goals_for_state, subgoal_layers)
            else:
                inputs = shared_inputs
            self._engine.encode(adv_state, adv_successor, inputs, builder)
            builder.next_graph()
        return builder.build_parts()

    def _parts_to_pyg(
        self,
        parts: Mapping[str, object],
        *,
        as_batch: bool,
        include_metadata: bool = True,
    ) -> HeteroData:
        return _parts_to_pyg(
            parts, as_batch=as_batch, include_metadata=include_metadata
        )

    def stream(self) -> "_TransitionEncoderStream":
        """Create a stream wrapper for transition encoding."""
        return _TransitionEncoderStream(self)


@dataclass
class _TransitionEncoderStream(StreamEncoderBase[HeteroData]):
    """Shared stream implementation for transition encoders."""

    _encoder: _TransitionEncoderBase

    def __post_init__(self) -> None:
        """Initialize an empty hetero builder for streaming."""
        self._reset_builder()

    def append(
        self,
        current: StateInput,
        successor: StateInput,
        *,
        goals: GoalBatchInput = None,
        subgoal_layers: SubgoalLayersInput = None,
    ) -> None:
        """Append one transition graph to the stream."""
        adv_current = _advanced_state(current)
        adv_successor = _advanced_state(successor)
        if goals is None:
            goals = default_goals_from_state(current)
        inputs, _ = _split_goals(goals, subgoal_layers)
        self._encoder.engine.encode(adv_current, adv_successor, inputs, self._builder)
        self._builder.next_graph()

    def _reset_builder(self) -> None:
        """Reset stream accumulation state."""
        self._builder = BatchBuilder()
        self._builder.set_graph_kind("hetero")

    def _parts_to_pyg(
        self,
        parts: Mapping[str, object],
        *,
        as_batch: bool,
        include_metadata: bool = True,
    ) -> HeteroData:
        return _parts_to_pyg(
            parts, as_batch=as_batch, include_metadata=include_metadata
        )


class TransitionHGraphEncoder(_TransitionEncoderBase):
    """Full successor encoder (encodes both source and successor structure)."""

    def __init__(
        self,
        domain: DomainInput,
        *,
        successor_suffix: str = "[suc]",
        **kwargs: object,
    ) -> None:
        super().__init__(
            domain,
            successor_mode=SuccessorEncoderMode.Full,
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
        **kwargs: object,
    ) -> None:
        super().__init__(
            domain,
            successor_mode=SuccessorEncoderMode.Delta,
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
