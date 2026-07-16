from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Collection, Iterable, Sequence

import numpy as np
from torch_geometric.data import HeteroData

from .._core import (
    BatchBuilder,
    BatchEncoding,
    DEFAULT_LGAN_NN_EDGE_POS,
    DEFAULT_SYMBOL_TYPE_ID,
)
from ..backends._ilg_runtime import (
    ILGAtom,
    ILGBackendName,
    ILGInput,
    ILGLiteral,
    create_ilg_runtime,
)
from .base import (
    ActionBatchInput,
    ActionBatchParam,
    EncoderBase,
    GoalBatchInput,
    GoalBatchParam,
    StateBatchInput,
    StreamEncoderBase,
    SubgoalLayersInput,
    SubgoalLayersBatchParam,
)
from .types import StateInput


class AtomStatus:
    """Bit-packed status descriptor used for ILG atom node features."""

    def __init__(
        self,
        *,
        is_regular: bool = True,
        is_negated: bool = False,
        is_satisfied: bool = False,
        goal_levels: tuple[int, ...] = (),
    ) -> None:
        self.is_regular = is_regular
        self.is_negated = is_negated
        self.is_satisfied = is_satisfied
        self.goal_levels = tuple(goal_levels)

    def encode(self) -> int:
        """Encode flags and goal-level mask into a compact integer."""
        bits = self.is_regular | (self.is_negated << 1) | (self.is_satisfied << 2)
        return (self._mask() << 3) | bits

    def _mask(self) -> int:
        mask = 0
        for level in self.goal_levels:
            if level >= 0:
                mask |= 1 << level
        return mask


def _goal_levels(
    goals: Sequence[ILGLiteral],
    subgoal_layers: Iterable[Iterable[ILGLiteral]],
) -> dict[ILGLiteral, int]:
    layers = [list(goals), *(list(layer) for layer in subgoal_layers)]
    return {literal: depth for depth, layer in enumerate(layers) for literal in layer}


@dataclass
class ILGEncoderStream(StreamEncoderBase[HeteroData]):
    """Streaming wrapper for the planner-neutral Python ``ILGEncoder``."""

    _encoder: "ILGEncoder"

    def __post_init__(self) -> None:
        self._reset_builder()

    def append(
        self,
        state: StateInput | Iterable[Any],
        *,
        goals: GoalBatchInput = None,
        actions: ActionBatchInput = None,
        subgoal_layers: SubgoalLayersInput = None,
    ) -> int:
        """Append one ILG graph to the stream."""
        stream_id = getattr(self, "_next_stream_id", 0)
        self._next_stream_id = stream_id + 1
        self._encoder._encode_to_builder(
            self._builder,
            state,
            goals=goals,
            actions=actions,
            subgoal_layers=subgoal_layers,
        )
        self._builder.next_graph()
        return stream_id

    def _reset_builder(self) -> None:
        self._builder = BatchBuilder()
        self._builder.set_graph_kind("hetero")


class ILGEncoder(EncoderBase[HeteroData]):
    """Instance-Learning Graph encoder with per-instance planner backends."""

    def __init__(
        self,
        domain: Any,
        *,
        backend: ILGBackendName | str | None = None,
        symbol_type_id: str = DEFAULT_SYMBOL_TYPE_ID,
        action_type_id: str = "action",
        nullary_object_name: str = "![nullary_symbol]!",
        add_nullary_predicates: bool = False,
        include_lgan_edges: bool = False,
        lgan_nn_edge_pos: str = DEFAULT_LGAN_NN_EDGE_POS,
    ) -> None:
        """Create an ILG encoder for one Pymimir domain or PyTyr task."""
        self._domain = domain
        self._runtime = create_ilg_runtime(domain, backend=backend)
        self.backend = self._runtime.backend_name
        self.symbol_type_id = symbol_type_id
        self.action_type_id = action_type_id
        self.nullary_object_name = nullary_object_name
        self.add_nullary_predicates = add_nullary_predicates
        self.include_lgan_edges = include_lgan_edges
        self.lgan_nn_edge_pos = lgan_nn_edge_pos
        self._relation_arity = dict(self._runtime.predicate_arities)
        self._action_feature_dim = int(self._runtime.action_feature_dim)

    @staticmethod
    def _compute_statuses(
        facts: Collection[ILGAtom],
        goals: Sequence[ILGLiteral],
        goal_level_map: dict[ILGLiteral, int],
    ) -> tuple[list[ILGAtom], dict[tuple[str, tuple[str, ...]], AtomStatus]]:
        fact_signatures = {fact.signature for fact in facts}
        goal_matches = {
            goal.atom.signature
            for goal in goals
            if (goal.atom.signature in fact_signatures) == goal.positive
        }
        statuses: dict[tuple[str, tuple[str, ...]], AtomStatus] = {}
        missing_goal_facts: list[ILGAtom] = []
        for goal in goals:
            signature = goal.atom.signature
            is_satisfied = signature in goal_matches
            previous = statuses.get(signature, AtomStatus())
            new_levels = tuple(
                sorted(set(previous.goal_levels) | {goal_level_map[goal]})
            )
            statuses[signature] = AtomStatus(
                is_regular=False,
                is_negated=not goal.positive,
                is_satisfied=is_satisfied,
                goal_levels=new_levels,
            )
            if not is_satisfied:
                missing_goal_facts.append(goal.atom)
        return missing_goal_facts, statuses

    def _encode_input_to_builder(
        self,
        builder: BatchBuilder,
        input_value: ILGInput,
    ) -> None:
        facts = list(input_value.facts)
        goals = list(input_value.goals)
        actions = list(input_value.actions)
        goal_level_map = _goal_levels(goals, input_value.subgoal_layers)
        missing_goal_facts, statuses = self._compute_statuses(
            facts, goals, goal_level_map
        )

        symbol_names = list(input_value.objects)
        if self.add_nullary_predicates and self.nullary_object_name not in symbol_names:
            symbol_names.append(self.nullary_object_name)
        builder.add_node_features(
            self.symbol_type_id,
            "x",
            np.zeros((len(symbol_names), 2), dtype=np.float32),
        )
        builder.set_node_names(self.symbol_type_id, symbol_names)
        builder.set_object_names(symbol_names)

        encoded_atoms = [*facts, *missing_goal_facts]
        pred_atoms: dict[str, list[ILGAtom]] = {}
        for atom in encoded_atoms:
            pred_atoms.setdefault(atom.predicate, []).append(atom)

        for pred_name, atoms in pred_atoms.items():
            feature_dim = self._relation_arity.get(pred_name, 0) + 1
            rows = [
                [float(statuses.get(atom.signature, AtomStatus()).encode())]
                * feature_dim
                for atom in atoms
            ]
            builder.add_node_features(
                pred_name,
                "x",
                np.asarray(rows, dtype=np.float32),
            )
            builder.set_node_names(pred_name, [atom.display_name for atom in atoms])

        if actions:
            builder.add_node_features(
                self.action_type_id,
                "x",
                np.zeros((len(actions), self._action_feature_dim), dtype=np.float32),
            )
            builder.set_node_names(
                self.action_type_id,
                [action.display_name for action in actions],
            )

        object_index = {name: index for index, name in enumerate(symbol_names)}
        atom_index = {
            pred_name: {atom.display_name: index for index, atom in enumerate(atoms)}
            for pred_name, atoms in pred_atoms.items()
        }
        edge_map: dict[tuple[str, str, str], tuple[list[int], list[int]]] = {}

        def add_edge(
            src_type: str,
            relation: str,
            dst_type: str,
            source: int,
            target: int,
        ) -> None:
            sources, targets = edge_map.setdefault(
                (src_type, relation, dst_type), ([], [])
            )
            sources.append(source)
            targets.append(target)

        for pred_name, atoms in pred_atoms.items():
            for atom in atoms:
                arguments = atom.arguments
                if not arguments and self._relation_arity.get(pred_name, 0) == 0:
                    arguments = (
                        (self.nullary_object_name,)
                        if self.add_nullary_predicates
                        else ()
                    )
                atom_idx = atom_index[pred_name][atom.display_name]
                for position, object_value in enumerate(arguments):
                    object_idx = object_index[object_value]
                    add_edge(
                        self.symbol_type_id,
                        str(position),
                        pred_name,
                        object_idx,
                        atom_idx,
                    )
                    add_edge(
                        pred_name,
                        str(position),
                        self.symbol_type_id,
                        atom_idx,
                        object_idx,
                    )

        for action_idx, action in enumerate(actions):
            for position, object_value in enumerate(action.arguments):
                object_idx = object_index[object_value]
                add_edge(
                    self.symbol_type_id,
                    str(position),
                    self.action_type_id,
                    object_idx,
                    action_idx,
                )
                add_edge(
                    self.action_type_id,
                    str(position),
                    self.symbol_type_id,
                    action_idx,
                    object_idx,
                )

        if self.include_lgan_edges:
            object_neighbors: dict[str, set[str]] = {}
            object_atoms: dict[str, list[ILGAtom]] = {}
            for atom in encoded_atoms:
                for first in atom.arguments:
                    object_atoms.setdefault(first, []).append(atom)
                    for second in atom.arguments:
                        if first != second:
                            object_neighbors.setdefault(first, set()).add(second)
            for target_object in object_atoms:
                neighbors = object_neighbors.get(target_object, set())
                target_idx = object_index[target_object]
                for atom in encoded_atoms:
                    if (
                        not atom.arguments
                        or target_object in atom.arguments
                        or not all(value in neighbors for value in atom.arguments)
                    ):
                        continue
                    atom_idx = atom_index[atom.predicate][atom.display_name]
                    add_edge(
                        self.symbol_type_id,
                        self.lgan_nn_edge_pos,
                        atom.predicate,
                        target_idx,
                        atom_idx,
                    )
                    add_edge(
                        atom.predicate,
                        self.lgan_nn_edge_pos,
                        self.symbol_type_id,
                        atom_idx,
                        target_idx,
                    )

        for (src_type, relation, dst_type), (sources, targets) in edge_map.items():
            if sources:
                builder.add_edges(
                    src_type,
                    relation,
                    dst_type,
                    np.asarray(sources, dtype=np.int64),
                    np.asarray(targets, dtype=np.int64),
                )

    def _encode_to_builder(
        self,
        builder: BatchBuilder,
        state: object,
        *,
        goals: object = None,
        actions: object = None,
        subgoal_layers: object = None,
    ) -> None:
        self._encode_input_to_builder(
            builder,
            self._runtime.make_input(
                state,
                goals=goals,
                actions=actions,
                subgoal_layers=subgoal_layers,
            ),
        )

    def _encode(
        self,
        state: StateInput | Iterable[Any],
        *,
        goals: GoalBatchInput = None,
        actions: ActionBatchInput = None,
        subgoal_layers: SubgoalLayersInput = None,
        **kwargs: Any,
    ) -> BatchEncoding:
        builder = BatchBuilder()
        builder.set_graph_kind("hetero")
        self._encode_to_builder(
            builder,
            state,
            goals=goals,
            actions=actions,
            subgoal_layers=subgoal_layers,
        )
        return builder.build()

    def _encode_batch(
        self,
        states: StateBatchInput,
        *,
        goals: GoalBatchParam = None,
        actions: ActionBatchParam = None,
        subgoal_layers: SubgoalLayersBatchParam = None,
        **kwargs: Any,
    ) -> BatchEncoding:
        inputs = self._runtime.make_batch_inputs(
            states,
            goals=goals,
            actions=actions,
            subgoal_layers=subgoal_layers,
        )
        builder = BatchBuilder()
        builder.set_graph_kind("hetero")
        for input_value in inputs:
            self._encode_input_to_builder(builder, input_value)
            builder.next_graph()
        return builder.build()

    def stream(self) -> ILGEncoderStream:
        """Create a streaming ILG encoder."""
        return ILGEncoderStream(self)


__all__ = ["ILGEncoder", "ILGEncoderStream", "AtomStatus"]
