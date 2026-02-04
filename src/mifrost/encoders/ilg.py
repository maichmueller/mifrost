from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Collection, Iterable, Mapping, Sequence

import torch
from torch_geometric.data import HeteroData

from .._core import BatchBuilder
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
from .base import EncoderBase, StreamEncoderBase
from .common import _parts_to_pyg


class AtomStatus:
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
        bits = self.is_regular | (self.is_negated << 1) | (self.is_satisfied << 2)
        return (self._mask() << 3) | bits

    def _mask(self) -> int:
        mask = 0
        for lvl in self.goal_levels:
            if lvl < 0:
                continue
            mask |= 1 << lvl
        return mask


def _gather_objects(
    items: Iterable[Any],
) -> list[Any]:
    objs: list[Any] = []
    seen: set[str] = set()
    for item in items:
        if hasattr(item, "get_objects"):
            candidates = list(action_objects(item))
        elif hasattr(item, "get_atom") and hasattr(item, "get_polarity"):
            candidates = list(atom_objects(literal_atom(item)))
        elif hasattr(item, "get_terms"):
            candidates = list(atom_objects(item))
        else:
            candidates = []
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
    layers = [list(goals)]
    if subgoal_layers is not None:
        layers.extend(list(layer) for layer in subgoal_layers)
    mapping: dict[Any, int] = {}
    for depth, layer in enumerate(layers):
        for literal in layer:
            mapping[literal] = depth
    return mapping


@dataclass
class ILGEncoderStream(StreamEncoderBase[HeteroData]):
    _encoder: "ILGEncoder"

    def __post_init__(self) -> None:
        self._reset_builder()

    def append(
        self,
        state: Any,
        *,
        goals: Iterable[Any] | None = None,
        actions: Iterable[Any] | None = None,
        subgoal_layers: Iterable[Iterable[Any]] | None = None,
    ) -> None:
        self._encoder._encode_to_builder(
            self._builder,
            state,
            goals=goals,
            actions=actions,
            subgoal_layers=subgoal_layers,
        )
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


class ILGEncoder(EncoderBase[HeteroData]):
    """
    Instance‑Learning Graph encoder (ILG).
    """

    def __init__(
        self,
        domain: Any,
        *,
        symbol_type_id: str = "_symbol_",
        action_type_id: str = "action",
        nullary_object_name: str = "![nullary_symbol]!",
        add_nullary_predicates: bool = False,
        include_lgan_edges: bool = False,
        lgan_nn_edge_pos: str = "lgan_nn",
    ) -> None:
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
        state: Any,
        *,
        goals: Iterable[Any] | None = None,
        actions: Iterable[Any] | None = None,
        subgoal_layers: Iterable[Iterable[Any]] | None = None,
    ) -> None:
        if hasattr(state, "get_problem"):
            problem = state.get_problem()
            facts = list(state.get_atoms())
            if goals is None:
                goals = list(problem.get_goal_condition().get_literals())
            objects = list(problem.get_objects()) + list(
                problem.get_domain().get_constants()
            )
        else:
            facts = list(state)
            if goals is None:
                goals = []
            objects = _gather_objects(facts)

        actions_list = list(actions) if actions is not None else []
        if not objects:
            objects = _gather_objects(list(facts) + list(goals) + actions_list)

        goal_level_map = _goal_levels(list(goals), subgoal_layers)
        missing_goal_facts, statuses = self._compute_statuses(
            facts, list(goals), goal_level_map
        )

        symbol_names = [object_name(obj) for obj in objects]
        if self.add_nullary_predicates and self.nullary_object_name not in symbol_names:
            symbol_names.append(self.nullary_object_name)

        builder.add_node_features(
            self.symbol_type_id,
            "x",
            [0.0] * (len(symbol_names) * 2),
            2,
        )
        builder.set_node_names(self.symbol_type_id, symbol_names)
        builder.set_object_names(symbol_names)

        pred_atoms: dict[str, list[Any]] = {}
        for atom in list(facts) + missing_goal_facts:
            pred_name = predicate_name(predicate(atom))
            pred_atoms.setdefault(pred_name, []).append(atom)

        for pred_name, atoms in pred_atoms.items():
            arity = self._relation_arity.get(pred_name, 0)
            feature_dim = arity + 1
            data: list[float] = []
            names: list[str] = []
            for atom in atoms:
                status = statuses.get(atom, AtomStatus())
                value = float(status.encode())
                data.extend([value] * feature_dim)
                names.append(str(atom))
            builder.add_node_features(pred_name, "x", data, feature_dim)
            builder.set_node_names(pred_name, names)

        if actions_list:
            max_action_arity = max(action_arity(action) for action in actions_list)
            action_dim = max_action_arity + 1
            action_names = [str(action) for action in actions_list]
            builder.add_node_features(
                self.action_type_id,
                "x",
                [0.0] * (len(action_names) * action_dim),
                action_dim,
            )
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
            builder.add_edges(src_type, rel, dst_type, src_list, dst_list)

    def encode_parts(
        self,
        state: Any,
        *,
        goals: Iterable[Any] | None = None,
        actions: Iterable[Any] | None = None,
        subgoal_layers: Iterable[Iterable[Any]] | None = None,
        **kwargs: Any,
    ) -> Mapping[str, Any]:
        builder = BatchBuilder()
        builder.set_graph_kind("hetero")
        self._encode_to_builder(
            builder, state, goals=goals, actions=actions, subgoal_layers=subgoal_layers
        )
        return builder.build_parts()

    def encode_batch_parts(
        self,
        states: Iterable[Any] | Any,
        *,
        goals: Iterable[Any] | None = None,
        actions: Iterable[Any] | None = None,
        subgoal_layers: Iterable[Iterable[Any]] | None = None,
        **kwargs: Any,
    ) -> Mapping[str, Any]:
        if hasattr(states, "get_problem"):
            state_list = [states]
        else:
            state_list = list(states)

        builder = BatchBuilder()
        builder.set_graph_kind("hetero")
        for state in state_list:
            self._encode_to_builder(
                builder,
                state,
                goals=goals,
                actions=actions,
                subgoal_layers=subgoal_layers,
            )
            if hasattr(builder, "next_graph"):
                builder.next_graph()
        return builder.build_parts()

    def stream(self) -> ILGEncoderStream:
        return ILGEncoderStream(self)


__all__ = ["ILGEncoder", "ILGEncoderStream", "AtomStatus"]
