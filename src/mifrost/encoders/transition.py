from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

from torch_geometric.data import HeteroData

from .._core import (
    BatchBuilder,
    GoalInputs,
    SuccessorEncoderConfig,
    SuccessorEncoderMode,
    SuccessorHGraphEncoderEngine,
)
from dataclasses import dataclass

from .base import EncoderBase, StreamEncoderBase
from .common import _advanced_domain, _advanced_state, _parts_to_pyg, _split_goals


class _TransitionEncoderBase(EncoderBase[HeteroData]):
    def __init__(
        self,
        domain: Any,
        *,
        successor_mode: SuccessorEncoderMode,
        successor_suffix: str,
        include_successor_goal_satisfaction: bool = False,
        symbol_type_id: str = "_symbol_",
        ignore_actions: bool = True,
        add_nullary_predicates: bool = False,
        include_lgan_edges: bool = False,
        include_static: bool = True,
        include_empty_edge_types: bool = True,
        max_goal_level: int = 0,
        support_literals: bool = False,
        nullary_object_name: str = "![nullary_symbol]!",
        lgan_nn_edge_pos: str = "lgan_nn",
    ) -> None:
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
        return self._engine

    def _accepted_kwargs(self) -> set[str]:
        return {"successor", "successors"}

    def encode_parts(
        self,
        state: Any,
        *,
        goals: Iterable[Any] | None = None,
        actions: Iterable[Any] | None = None,
        subgoal_layers: Iterable[Iterable[Any]] | None = None,
        successor: Any | None = None,
        **kwargs: Any,
    ) -> Mapping[str, Any]:
        if successor is None:
            raise ValueError("successor must be provided for transition encoding")
        adv_state = _advanced_state(state)
        adv_successor = _advanced_state(successor)
        if goals is None:
            if hasattr(state, "get_problem"):
                goals = list(state.get_problem().get_goal_condition().get_literals())
            else:
                raise ValueError(
                    "goals must be provided when passing an advanced state"
                )
        inputs, _ = _split_goals(goals, subgoal_layers)
        return self._engine.encode(adv_state, adv_successor, inputs)

    def encode_batch_parts(
        self,
        states: Iterable[Any] | Any,
        *,
        goals: Iterable[Any] | None = None,
        actions: Iterable[Any] | None = None,
        subgoal_layers: Iterable[Iterable[Any]] | None = None,
        successors: Iterable[Any] | Any | None = None,
        **kwargs: Any,
    ) -> Mapping[str, Any]:
        if hasattr(states, "get_problem"):
            state_list = [states]
        else:
            state_list = list(states)

        if successors is None:
            raise ValueError(
                "successors must be provided for transition batch encoding"
            )
        if hasattr(successors, "get_problem") or hasattr(successors, "_advanced_state"):
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
                if hasattr(state, "get_problem"):
                    goals_for_state = list(
                        state.get_problem().get_goal_condition().get_literals()
                    )
                    inputs, _ = _split_goals(goals_for_state, subgoal_layers)
                else:
                    raise ValueError(
                        "goals must be provided when passing an advanced state"
                    )
            else:
                inputs = shared_inputs
            self._engine.encode(adv_state, adv_successor, inputs, builder)
            if hasattr(builder, "next_graph"):
                builder.next_graph()
        return builder.build_parts()

    def _parts_to_pyg(
        self,
        parts: Mapping[str, Any],
        *,
        as_batch: bool,
        include_metadata: bool = True,
    ) -> HeteroData:
        return _parts_to_pyg(
            parts, as_batch=as_batch, include_metadata=include_metadata
        )

    def stream(self) -> "_TransitionEncoderStream":
        return _TransitionEncoderStream(self)


@dataclass
class _TransitionEncoderStream(StreamEncoderBase[HeteroData]):
    _encoder: _TransitionEncoderBase

    def __post_init__(self) -> None:
        self._reset_builder()

    def append(
        self,
        current: Any,
        successor: Any,
        *,
        goals: Iterable[Any] | None = None,
        subgoal_layers: Iterable[Iterable[Any]] | None = None,
    ) -> None:
        adv_current = _advanced_state(current)
        adv_successor = _advanced_state(successor)
        if goals is None:
            if hasattr(current, "get_problem"):
                goals = list(current.get_problem().get_goal_condition().get_literals())
            else:
                raise ValueError(
                    "goals must be provided when passing an advanced state"
                )
        inputs, _ = _split_goals(goals, subgoal_layers)
        self._encoder.engine.encode(adv_current, adv_successor, inputs, self._builder)
        if hasattr(self._builder, "next_graph"):
            self._builder.next_graph()

    def _reset_builder(self) -> None:
        self._builder = BatchBuilder()
        self._builder.set_graph_kind("hetero")

    def _parts_to_pyg(
        self,
        parts: Mapping[str, Any],
        *,
        as_batch: bool,
        include_metadata: bool = True,
    ) -> HeteroData:
        return _parts_to_pyg(
            parts, as_batch=as_batch, include_metadata=include_metadata
        )


class TransitionHGraphEncoder(_TransitionEncoderBase):
    def __init__(
        self,
        domain: Any,
        *,
        successor_suffix: str = "[suc]",
        **kwargs: Any,
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
    def __init__(
        self,
        domain: Any,
        *,
        successor_suffix: str = "",
        **kwargs: Any,
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
    _encoder: TransitionHGraphEncoder


@dataclass
class TransitionEffectsHGraphEncoderStream(_TransitionEncoderStream):
    _encoder: TransitionEffectsHGraphEncoder


__all__ = [
    "TransitionHGraphEncoder",
    "TransitionEffectsHGraphEncoder",
    "TransitionHGraphEncoderStream",
    "TransitionEffectsHGraphEncoderStream",
]
