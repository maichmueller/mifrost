from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from torch_geometric.data import HeteroData

from .._core import BatchBuilder, GoalInputs, HGraphEncoderConfig, HGraphEncoderEngine
from .common import (
    _advanced_domain,
    _advanced_state,
    _parts_to_pyg,
    _prepare_actions,
    _split_goals,
)


@dataclass
class HGraphEncoderStream:
    _engine: HGraphEncoderEngine

    def __post_init__(self) -> None:
        self._builder = BatchBuilder()
        self._builder.set_graph_kind("hetero")

    def append(
        self,
        state: Any,
        *,
        goals: Iterable[Any] | None = None,
        actions: Iterable[Any] | None = None,
        subgoal_layers: Iterable[Iterable[Any]] | None = None,
    ) -> None:
        adv_state = _advanced_state(state)
        action_list = _prepare_actions(actions)
        if goals is None and subgoal_layers is None and not action_list:
            # Fast path: let the engine derive goals from the state/problem.
            self._engine.encode(adv_state, self._builder)
        else:
            if goals is None:
                if hasattr(state, "get_problem"):
                    goals = list(
                        state.get_problem().get_goal_condition().get_literals()
                    )
                else:
                    raise ValueError(
                        "goals must be provided when passing an advanced state"
                    )
            inputs, _ = _split_goals(goals, subgoal_layers)
            self._engine.encode(
                adv_state,
                inputs,
                action_list,
                self._builder,
            )
        if hasattr(self._builder, "next_graph"):
            self._builder.next_graph()

    def flush(
        self, *, as_batch: bool = True, include_metadata: bool = True
    ) -> HeteroData:
        parts = self.flush_parts()
        return _parts_to_pyg(
            parts, as_batch=as_batch, include_metadata=include_metadata
        )

    def flush_parts(self) -> Mapping[str, Any]:
        parts = self._builder.build_parts()
        self._builder = BatchBuilder()
        self._builder.set_graph_kind("hetero")
        return parts


class HGraphEncoder:
    def __init__(
        self,
        domain: Any,
        *,
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
        config = HGraphEncoderConfig()
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
        self._engine = HGraphEncoderEngine(_advanced_domain(domain), config)

    @property
    def engine(self) -> HGraphEncoderEngine:
        return self._engine

    def encode_parts(
        self,
        state: Any,
        *,
        goals: Iterable[Any] | None = None,
        actions: Iterable[Any] | None = None,
        subgoal_layers: Iterable[Iterable[Any]] | None = None,
    ) -> Mapping[str, Any]:
        adv_state = _advanced_state(state)
        action_list = _prepare_actions(actions)
        if goals is None and subgoal_layers is None and not action_list:
            return self._engine.encode(adv_state)
        if goals is None:
            if hasattr(state, "get_problem"):
                goals = list(state.get_problem().get_goal_condition().get_literals())
            else:
                raise ValueError(
                    "goals must be provided when passing an advanced state"
                )
        inputs, _ = _split_goals(goals, subgoal_layers)
        # Explicitly pass goals/actions across the strict C++ boundary.
        return self._engine.encode(adv_state, inputs, action_list)

    def encode(
        self,
        state: Any,
        *,
        goals: Iterable[Any] | None = None,
        actions: Iterable[Any] | None = None,
        subgoal_layers: Iterable[Iterable[Any]] | None = None,
        include_metadata: bool = True,
    ) -> HeteroData:
        parts = self.encode_parts(
            state,
            goals=goals,
            actions=actions,
            subgoal_layers=subgoal_layers,
        )
        return _parts_to_pyg(parts, as_batch=False, include_metadata=include_metadata)

    def _encode_batch_parts(
        self,
        states: Iterable[Any] | Any,
        *,
        goals: Iterable[Any] | None = None,
        actions: Iterable[Any] | None = None,
        subgoal_layers: Iterable[Iterable[Any]] | None = None,
    ) -> Mapping[str, Any]:
        is_state_like = hasattr(states, "get_problem") or hasattr(
            states, "_advanced_state"
        )
        if is_state_like:
            state_list = [states]
        else:
            if isinstance(states, (str, bytes)):
                raise TypeError("encode_batch expects a state or an iterable of states")
            state_list = list(states)

        shared_actions: list[Any] | None = None
        per_state_actions: list[Any] | None = None
        if actions is not None:
            try:
                shared_actions = _prepare_actions(actions)
            except TypeError:
                if isinstance(actions, Sequence):
                    if len(actions) != len(state_list):
                        raise ValueError(
                            "actions length must match states when providing per-state actions"
                        ) from None
                    per_state_actions = list(actions)
                else:
                    raise

        shared_inputs: GoalInputs | None = None
        if goals is not None:
            shared_inputs, _ = _split_goals(goals, subgoal_layers)

        builder = BatchBuilder()
        builder.set_graph_kind("hetero")
        for idx, state in enumerate(state_list):
            adv_state = _advanced_state(state)
            if per_state_actions is not None:
                actions_for_state = per_state_actions[idx]
                action_list = (
                    _prepare_actions(actions_for_state)
                    if actions_for_state is not None
                    else []
                )
            else:
                action_list = shared_actions or []

            if goals is None and subgoal_layers is None and not action_list:
                # Fast path: let the engine derive goals from the state/problem.
                self._engine.encode(adv_state, builder)
            else:
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
                self._engine.encode(adv_state, inputs, action_list, builder)
            if hasattr(builder, "next_graph"):
                builder.next_graph()

        return builder.build_parts()

    def encode_batch(
        self,
        states: Iterable[Any] | Any,
        *,
        goals: Iterable[Any] | None = None,
        actions: Iterable[Any] | None = None,
        subgoal_layers: Iterable[Iterable[Any]] | None = None,
        include_metadata: bool = True,
    ) -> HeteroData:
        """
        Encode multiple states into a single PyG Batch using the C++ BatchBuilder.

        If a single state object is provided, it is treated as a batch of size 1.
        """
        parts = self._encode_batch_parts(
            states,
            goals=goals,
            actions=actions,
            subgoal_layers=subgoal_layers,
        )
        return _parts_to_pyg(parts, as_batch=True, include_metadata=include_metadata)

    def encode_batch_parts(
        self,
        states: Iterable[Any] | Any,
        *,
        goals: Iterable[Any] | None = None,
        actions: Iterable[Any] | None = None,
        subgoal_layers: Iterable[Iterable[Any]] | None = None,
    ) -> Mapping[str, Any]:
        """
        Encode multiple states and return engine parts without PyG assembly.
        """
        return self._encode_batch_parts(
            states,
            goals=goals,
            actions=actions,
            subgoal_layers=subgoal_layers,
        )

    def stream(self) -> HGraphEncoderStream:
        return HGraphEncoderStream(self._engine)


__all__ = ["HGraphEncoder", "HGraphEncoderStream"]
