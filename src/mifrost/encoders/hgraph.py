from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

from torch_geometric.data import HeteroData

from .._core import BatchBuilder, GoalInputs, HGraphEncoderConfig, HGraphEncoderEngine
from .base import (
    ActionBatchInput,
    EncoderBase,
    GoalBatchInput,
    StateBatchInput,
    StreamEncoderBase,
    SubgoalLayersInput,
)
from .common import (
    _advanced_domain,
    _advanced_state,
    _parts_to_pyg,
    _prepare_actions,
    _split_goals,
)
from .types import (
    GroundActionInput,
    GoalLiteralInput,
    DomainInput,
    StateInput,
    is_action_input,
    default_goals_from_state,
    is_state_input,
)


@dataclass
class HGraphEncoderStream(StreamEncoderBase[HeteroData]):
    """Streaming wrapper for ``HGraphEncoderEngine``."""

    _engine: HGraphEncoderEngine

    def __post_init__(self) -> None:
        """Initialize an empty hetero builder for streaming."""
        self._reset_builder()

    def append(
        self,
        state: StateInput,
        *,
        goals: Iterable[GoalLiteralInput] | None = None,
        actions: Iterable[GroundActionInput] | None = None,
        subgoal_layers: Iterable[Iterable[GoalLiteralInput]] | None = None,
    ) -> None:
        """
        Append one state encoding to the current stream.

        If goals/actions are omitted, the engine uses the state's problem goals.
        """
        adv_state = _advanced_state(state)
        action_list = _prepare_actions(actions)
        if goals is None and subgoal_layers is None and not action_list:
            # Fast path: let the engine derive goals from the state/problem.
            self._engine.encode(adv_state, self._builder)
        else:
            if goals is None:
                goals = default_goals_from_state(state)
            inputs, _ = _split_goals(goals, subgoal_layers)
            self._engine.encode(
                adv_state,
                inputs,
                action_list,
                self._builder,
            )
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


class HGraphEncoder(EncoderBase[HeteroData]):
    """
    General heterogeneous graph encoder backed by ``HGraphEncoderEngine``.

    Use this encoder when you need state-based atom/object/action graphs in
    hetero PyG format.
    """

    def __init__(
        self,
        domain: DomainInput,
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
        """Create an HGraph encoder for one domain."""
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
        """Expose the underlying C++ engine for advanced usage."""
        return self._engine

    def encode_parts(
        self,
        state: StateInput,
        *,
        goals: GoalBatchInput = None,
        actions: Iterable[GroundActionInput] | None = None,
        subgoal_layers: SubgoalLayersInput = None,
    ) -> Mapping[str, object]:
        """Encode one state to normalized parts."""
        adv_state = _advanced_state(state)
        action_list = _prepare_actions(actions)
        if goals is None and subgoal_layers is None and not action_list:
            return self._engine.encode(adv_state)
        if goals is None:
            goals = default_goals_from_state(state)
        inputs, _ = _split_goals(goals, subgoal_layers)
        # Explicitly pass goals/actions across the strict C++ boundary.
        return self._engine.encode(adv_state, inputs, action_list)

    def encode(
        self,
        state: StateInput,
        *,
        goals: GoalBatchInput = None,
        actions: Iterable[GroundActionInput] | None = None,
        subgoal_layers: SubgoalLayersInput = None,
        include_metadata: bool = True,
        **kwargs: object,
    ) -> HeteroData:
        """Encode one state into ``HeteroData``."""
        return super().encode(
            state,
            goals=goals,
            actions=actions,
            subgoal_layers=subgoal_layers,
            include_metadata=include_metadata,
            **kwargs,
        )

    def encode_batch_parts(
        self,
        states: StateBatchInput,
        *,
        goals: GoalBatchInput = None,
        actions: Iterable[GroundActionInput]
        | Sequence[Iterable[GroundActionInput] | None]
        | None = None,
        subgoal_layers: SubgoalLayersInput = None,
    ) -> Mapping[str, object]:
        """
        Encode one or many states to batch parts.

        Supports either shared actions or per-state action sequences.
        """
        if is_state_input(states):
            state_list = [states]
        else:
            if isinstance(states, (str, bytes)):
                raise TypeError("encode_batch expects a state or an iterable of states")
            state_list = list(states)

        shared_actions: list[GroundActionInput] | None = None
        shared_action_list: list[object] = []
        per_state_actions: list[Iterable[GroundActionInput] | None] | None = None
        if actions is not None:
            if isinstance(actions, Sequence) and actions:
                first = actions[0]
                if first is not None and not is_action_input(first):
                    if len(actions) != len(state_list):
                        raise ValueError(
                            "actions length must match states when providing per-state actions"
                        )
                    per_state_actions = list(actions)
                else:
                    shared_actions = list(actions)
                    shared_action_list = _prepare_actions(shared_actions)
            else:
                shared_actions = list(actions)
                shared_action_list = _prepare_actions(shared_actions)

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
                action_list = shared_action_list

            if goals is None and subgoal_layers is None and not action_list:
                # Fast path: let the engine derive goals from the state/problem.
                self._engine.encode(adv_state, builder)
            else:
                if goals is None:
                    goals_for_state = default_goals_from_state(state)
                    inputs, _ = _split_goals(goals_for_state, subgoal_layers)
                else:
                    inputs = shared_inputs
                self._engine.encode(adv_state, inputs, action_list, builder)
            builder.next_graph()

        return builder.build_parts()

    def encode_batch(
        self,
        states: StateBatchInput,
        *,
        goals: GoalBatchInput = None,
        actions: Iterable[GroundActionInput]
        | Sequence[Iterable[GroundActionInput] | None]
        | None = None,
        subgoal_layers: SubgoalLayersInput = None,
        include_metadata: bool = True,
        **kwargs: object,
    ) -> HeteroData:
        """Encode one or many states into a batched ``HeteroData`` object."""
        return super().encode_batch(
            states,
            goals=goals,
            actions=actions,
            subgoal_layers=subgoal_layers,
            include_metadata=include_metadata,
            **kwargs,
        )

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

    def stream(self) -> HGraphEncoderStream:
        """Create a streaming encoder sharing this encoder's C++ engine."""
        return HGraphEncoderStream(self._engine)


__all__ = ["HGraphEncoder", "HGraphEncoderStream"]
