from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Collection, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch_geometric.data import HeteroData

from .. import _core
from .._core import BatchBuilder, DEFAULT_LGAN_NN_EDGE_POS, DEFAULT_SYMBOL_TYPE_ID
from .accessors import (
    action_arity,
    action_objects,
    atom_objects,
    atoms_equal,
    literal_atom,
    literal_polarity,
    object_name,
    predicate,
    predicate_arity,
    predicate_name,
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
from .common import (
    _advanced_action,
    _advanced_literal,
    _advanced_state,
    _encoding_dict_to_pyg,
)
from .types import (
    ATOM_TYPES,
    HeteroEncoding,
    NativeEncodingInput,
    StateInput,
    WRAPPER_STATE_TYPES,
    is_action_input,
    is_goal_literal_input,
    is_state_input,
)


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
        """Return bitmask with one bit per present goal layer."""
        mask = 0
        for lvl in self.goal_levels:
            if lvl < 0:
                continue
            mask |= 1 << lvl
        return mask


def _gather_objects(
    items: Iterable[Any],
) -> list[Any]:
    """Collect unique objects referenced by atoms/literals/actions."""
    objs: list[Any] = []
    seen: set[str] = set()
    for item in items:
        if is_action_input(item):
            candidates = list(action_objects(_advanced_action(item)))
        elif is_goal_literal_input(item):
            candidates = list(atom_objects(literal_atom(_advanced_literal(item))))
        elif isinstance(item, ATOM_TYPES):
            candidates = list(atom_objects(item))
        else:
            raise TypeError(f"Unsupported object source type: {type(item)!r}")
        for obj in candidates:
            name = object_name(obj)
            if name in seen:
                continue
            seen.add(name)
            objs.append(obj)
    return objs


def _goal_levels(
    goals: Sequence[Any],
    subgoal_layers: Iterable[Iterable[Any]] | None,
) -> dict[Any, int]:
    """Assign each literal to its layer depth (goals layer = 0)."""
    layers = [list(goals)]
    if subgoal_layers is not None:
        layers.extend(list(layer) for layer in subgoal_layers)
    mapping: dict[Any, int] = {}
    for depth, layer in enumerate(layers):
        for literal in layer:
            mapping[literal] = depth
    return mapping


@dataclass
class ILGEncoderStream(StreamEncoderBase[HeteroData, HeteroEncoding]):
    """Streaming wrapper for the pure-Python ``ILGEncoder``."""

    _encoder: "ILGEncoder"

    def __post_init__(self) -> None:
        """Initialize an empty hetero builder for streaming."""
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
        """Reset stream accumulation state."""
        self._builder = BatchBuilder()
        self._builder.set_graph_kind("hetero")

    def _dict_to_pyg(
        self,
        encoding_dict: NativeEncodingInput,
        *,
        as_batch: bool,
        include_metadata: bool = True,
    ) -> HeteroData:
        return _encoding_dict_to_pyg(
            encoding_dict, as_batch=as_batch, include_metadata=include_metadata
        )


class ILGEncoder(EncoderBase[HeteroData, HeteroEncoding]):
    """
    Instance‑Learning Graph encoder (ILG) implemented in Python.

    This encoder mirrors the ILG topology/features and emits native batch encodings
    via ``BatchBuilder``.
    """

    def __init__(
        self,
        domain: Any,
        *,
        symbol_type_id: str = DEFAULT_SYMBOL_TYPE_ID,
        action_type_id: str = "action",
        nullary_object_name: str = "![nullary_symbol]!",
        add_nullary_predicates: bool = False,
        include_lgan_edges: bool = False,
        lgan_nn_edge_pos: str = DEFAULT_LGAN_NN_EDGE_POS,
    ) -> None:
        """Create an ILG encoder for one domain."""
        self._domain = domain
        self.symbol_type_id = symbol_type_id
        self.action_type_id = action_type_id
        self.nullary_object_name = nullary_object_name
        self.add_nullary_predicates = add_nullary_predicates
        self.include_lgan_edges = include_lgan_edges
        self.lgan_nn_edge_pos = lgan_nn_edge_pos

        self._predicates = tuple(domain.get_predicates())
        self._relation_arity: dict[str, int] = {
            predicate_name(pred): predicate_arity(pred) for pred in self._predicates
        }

    def _compute_statuses(
        self,
        facts: Collection[Any],
        goals: Sequence[Any],
        goal_level_map: dict[Any, int],
    ) -> tuple[list[Any], dict[Any, AtomStatus]]:
        """
        Compute per-atom status flags and collect unsatisfied goal atoms.

        Returns ``(missing_goal_facts, status_by_atom)``.
        """
        goal_matches = {
            literal_atom(goal)
            for goal in goals
            if any(atoms_equal(literal_atom(goal), f) for f in facts)
            == literal_polarity(goal)
        }
        statuses: dict[Any, AtomStatus] = {}
        missing_goal_facts: list[Any] = []
        for goal in goals:
            atom = literal_atom(goal)
            is_satisfied = atom in goal_matches
            prev = statuses.get(atom, AtomStatus())
            new_levels = tuple(sorted(set(prev.goal_levels) | {goal_level_map[goal]}))
            statuses[atom] = AtomStatus(
                is_regular=False,
                is_negated=not literal_polarity(goal),
                is_satisfied=is_satisfied,
                goal_levels=new_levels,
            )
            if not is_satisfied:
                missing_goal_facts.append(atom)
        return missing_goal_facts, statuses

    def _encode_to_builder(
        self,
        builder: BatchBuilder,
        state: StateInput | Iterable[Any],
        *,
        goals: GoalBatchInput = None,
        actions: ActionBatchInput = None,
        subgoal_layers: SubgoalLayersInput = None,
    ) -> None:
        """
        Encode one sample into an existing builder.

        This is the common implementation used by single, batch and stream paths.
        """
        if isinstance(state, WRAPPER_STATE_TYPES):
            problem = state.get_problem()
            facts = list(state.get_atoms())
            if goals is None:
                goals = list(problem.get_goal_condition().get_literals())
            objects = list(problem.get_objects()) + list(
                problem.get_domain().get_constants()
            )
        elif is_state_input(state):
            advanced_state = _advanced_state(state)
            facts = list(advanced_state.get_fluent_atoms()) + list(
                advanced_state.get_derived_atoms()
            )
            if goals is None:
                goals = []
            objects = _gather_objects(facts)
        else:
            facts = list(state)
            if goals is None:
                goals = []
            objects = _gather_objects(facts)

        goals_list = (
            [_advanced_literal(goal) for goal in goals] if goals is not None else []
        )
        actions_list = (
            [_advanced_action(action) for action in actions]
            if actions is not None
            else []
        )
        subgoal_layers_list = (
            [[_advanced_literal(goal) for goal in layer] for layer in subgoal_layers]
            if subgoal_layers is not None
            else None
        )

        if not objects:
            objects = _gather_objects(list(facts) + goals_list + actions_list)

        goal_level_map = _goal_levels(goals_list, subgoal_layers_list)
        missing_goal_facts, statuses = self._compute_statuses(
            facts, goals_list, goal_level_map
        )

        symbol_names = [object_name(obj) for obj in objects]
        if self.add_nullary_predicates and self.nullary_object_name not in symbol_names:
            symbol_names.append(self.nullary_object_name)

        symbol_x = np.zeros((len(symbol_names), 2), dtype=np.float32)
        builder.add_node_features(self.symbol_type_id, "x", symbol_x)
        builder.set_node_names(self.symbol_type_id, symbol_names)
        builder.set_object_names(symbol_names)

        pred_atoms: dict[str, list[Any]] = {}
        for atom in list(facts) + missing_goal_facts:
            pred_name = predicate_name(predicate(atom))
            pred_atoms.setdefault(pred_name, []).append(atom)

        for pred_name, atoms in pred_atoms.items():
            arity = self._relation_arity.get(pred_name, 0)
            feature_dim = arity + 1
            rows: list[list[float]] = []
            names: list[str] = []
            for atom in atoms:
                status = statuses.get(atom, AtomStatus())
                value = float(status.encode())
                rows.append([value] * feature_dim)
                names.append(str(atom))
            pred_x = np.asarray(rows, dtype=np.float32)
            builder.add_node_features(pred_name, "x", pred_x)
            builder.set_node_names(pred_name, names)

        if actions_list:
            max_action_arity = max(action_arity(action) for action in actions_list)
            action_dim = max_action_arity + 1
            action_names = [str(action) for action in actions_list]
            action_x = np.zeros((len(action_names), action_dim), dtype=np.float32)
            builder.add_node_features(self.action_type_id, "x", action_x)
            builder.set_node_names(self.action_type_id, action_names)

        object_index = {name: idx for idx, name in enumerate(symbol_names)}
        atom_index: dict[str, dict[str, int]] = {}
        for pred_name, atoms in pred_atoms.items():
            atom_index[pred_name] = {str(atom): idx for idx, atom in enumerate(atoms)}

        edge_map: dict[tuple[str, str, str], tuple[list[int], list[int]]] = {}

        def add_edge(
            src_type: str, rel: str, dst_type: str, src: int, dst: int
        ) -> None:
            key = (src_type, rel, dst_type)
            if key not in edge_map:
                edge_map[key] = ([], [])
            edge_map[key][0].append(src)
            edge_map[key][1].append(dst)

        for pred_name, atoms in pred_atoms.items():
            for atom in atoms:
                obj_terms = list(atom_objects(atom))
                if not obj_terms and predicate_arity(predicate(atom)) == 0:
                    if self.add_nullary_predicates:
                        obj_terms = [self.nullary_object_name]
                    else:
                        continue
                atom_idx = atom_index[pred_name][str(atom)]
                for pos, obj in enumerate(obj_terms):
                    obj_name = obj if isinstance(obj, str) else object_name(obj)
                    obj_idx = object_index[obj_name]
                    add_edge(
                        self.symbol_type_id, str(pos), pred_name, obj_idx, atom_idx
                    )
                    add_edge(
                        pred_name, str(pos), self.symbol_type_id, atom_idx, obj_idx
                    )

        for action_idx, action in enumerate(actions_list):
            for pos, obj in enumerate(action_objects(action)):
                obj_name = object_name(obj)
                obj_idx = object_index[obj_name]
                add_edge(
                    self.symbol_type_id,
                    str(pos),
                    self.action_type_id,
                    obj_idx,
                    action_idx,
                )
                add_edge(
                    self.action_type_id,
                    str(pos),
                    self.symbol_type_id,
                    action_idx,
                    obj_idx,
                )

        if self.include_lgan_edges:
            object_neighbors: dict[Any, set[Any]] = {}
            obj_to_atoms: dict[Any, list[Any]] = {}
            for atom in list(facts) + missing_goal_facts:
                objs = list(atom_objects(atom))
                for o1 in objs:
                    obj_to_atoms.setdefault(o1, []).append(atom)
                    for o2 in objs:
                        if o1 != o2:
                            object_neighbors.setdefault(o1, set()).add(o2)
            for target_obj, atoms in obj_to_atoms.items():
                neighbors = object_neighbors.get(target_obj, set())
                target_name = (
                    target_obj
                    if isinstance(target_obj, str)
                    else object_name(target_obj)
                )
                target_idx = object_index[target_name]
                for atom in list(facts) + missing_goal_facts:
                    atom_objs = list(atom_objects(atom))
                    if not atom_objs:
                        continue
                    if target_obj in atom_objs:
                        continue
                    if all(o in neighbors for o in atom_objs):
                        pred_name = predicate_name(predicate(atom))
                        atom_idx = atom_index[pred_name][str(atom)]
                        add_edge(
                            self.symbol_type_id,
                            self.lgan_nn_edge_pos,
                            pred_name,
                            target_idx,
                            atom_idx,
                        )
                        add_edge(
                            pred_name,
                            self.lgan_nn_edge_pos,
                            self.symbol_type_id,
                            atom_idx,
                            target_idx,
                        )

        for (src_type, rel, dst_type), (src_list, dst_list) in edge_map.items():
            if not src_list:
                continue
            src_arr = np.asarray(src_list, dtype=np.int64)
            dst_arr = np.asarray(dst_list, dtype=np.int64)
            builder.add_edges(src_type, rel, dst_type, src_arr, dst_arr)

    def _encode(
        self,
        state: StateInput | Iterable[Any],
        *,
        goals: GoalBatchInput = None,
        actions: ActionBatchInput = None,
        subgoal_layers: SubgoalLayersInput = None,
        **kwargs: object,
    ) -> HeteroEncoding:
        """Encode one state into ILG format."""
        builder = BatchBuilder()
        builder.set_graph_kind("hetero")
        self._encode_to_builder(
            builder, state, goals=goals, actions=actions, subgoal_layers=subgoal_layers
        )
        return builder.build()

    def _encode_batch(
        self,
        states: StateBatchInput,
        *,
        goals: GoalBatchParam = None,
        actions: ActionBatchParam = None,
        subgoal_layers: SubgoalLayersBatchParam = None,
        **kwargs: object,
    ) -> HeteroEncoding:
        """Encode one or many states into ILG batch format."""
        state_list, goals_per_state, actions_per_state, subgoal_layers_per_state = (
            _core._parse_ilg_batch_inputs(
                states,
                goals=goals,
                actions=actions,
                subgoal_layers=subgoal_layers,
            )
        )

        builder = BatchBuilder()
        builder.set_graph_kind("hetero")
        for idx, state in enumerate(state_list):
            self._encode_to_builder(
                builder,
                state,
                goals=goals_per_state[idx],
                actions=actions_per_state[idx],
                subgoal_layers=subgoal_layers_per_state[idx],
            )
            builder.next_graph()
        return builder.build()

    def stream(self) -> ILGEncoderStream:
        """Create a streaming ILG encoder."""
        return ILGEncoderStream(self)


__all__ = ["ILGEncoder", "ILGEncoderStream", "AtomStatus"]
