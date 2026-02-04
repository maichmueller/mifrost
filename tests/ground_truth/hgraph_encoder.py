from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from itertools import chain
from typing import Iterable, Mapping, Sequence

import pymimir
import torch
from torch_geometric.data import HeteroData

from .pyg_builder import PygHeteroBuilder


class RelationFormatter:
    positive_prefix = "[+]"
    negative_prefix = "[-]"
    goal_suffix = "[g]"
    subgoal_suffix = "[sg]"
    subsubgoal_suffix = "[ssg]"
    subsubsubgoal_suffix = "[sssg]"
    goal_satisfied_suffix = "[sat]"
    goal_unsatisfied_suffix = "[unsat]"
    goal_satisfied_added_suffix = "[sat+]"
    goal_satisfied_removed_suffix = "[sat-]"
    default_nullary_symbol_name = "![nullary_symbol]!"

    goal_level_suffixes = {
        0: goal_suffix,
        1: subgoal_suffix,
        2: subsubgoal_suffix,
        3: subsubsubgoal_suffix,
        None: "",
    }
    polarity_prefixes = {
        True: positive_prefix,
        False: negative_prefix,
        None: "",
    }
    goal_satisfaction_suffixes = {
        True: goal_satisfied_suffix,
        False: goal_unsatisfied_suffix,
        "+": goal_satisfied_added_suffix,
        "-": goal_satisfied_removed_suffix,
        None: "",
    }

    @classmethod
    def format_predicate(
        cls,
        name: str,
        goal_level: int | None = None,
        goal_satisfaction: bool | str | None = None,
        polarity: bool | None = None,
    ) -> str:
        return (
            cls.polarity_prefixes[polarity]
            + name
            + cls.goal_level_suffixes[goal_level]
            + cls.goal_satisfaction_suffixes[goal_satisfaction]
        )

    @classmethod
    def format_atom(cls, atom: pymimir.GroundAtom) -> str:
        return str(atom)

    @classmethod
    def format_literal(
        cls,
        literal: pymimir.GroundLiteral,
        goal_level: int | None = None,
        goal_satisfaction: bool | str | None = None,
    ) -> str:
        polarity = literal.get_polarity()
        atom_str = cls.format_atom(literal.get_atom())
        literal_str = f"{cls.polarity_prefixes[polarity]}{atom_str}"
        return (
            f"{literal_str}"
            f"{cls.goal_level_suffixes[goal_level]}"
            f"{cls.goal_satisfaction_suffixes[goal_satisfaction]}"
        )

    @staticmethod
    def format_action_schema(action: pymimir.Action) -> str:
        return action.get_name()

    @staticmethod
    def format_action(action: pymimir.GroundAction) -> str:
        parts = [action.get_action().get_name()]
        parts.extend(obj.get_name() for obj in action.get_objects())
        return "(" + " ".join(parts) + ")"

    @staticmethod
    def format_object(obj: pymimir.Object) -> str:
        return obj.get_name()


def _parse_goal_satisfaction(derivations: Iterable[bool | str] | None) -> set:
    out = set()
    for entry in derivations or []:
        if isinstance(entry, str):
            lowered = entry.lower().strip()
            if lowered == str(True).lower():
                out.add(True)
            elif lowered == str(False).lower():
                out.add(False)
            else:
                out.add(entry)
        else:
            out.add(entry)
        out.add(None)
    return out


@dataclass
class RelationDict:
    arity: dict[str, int]
    max_goal_level: int
    support_literals: bool
    goal_satisfaction_derivations: set

    @classmethod
    def build(
        cls,
        predicates: Sequence[pymimir.Predicate],
        actions: Sequence[pymimir.Action],
        *,
        max_goal_level: int = 0,
        support_literals: bool = False,
        top_type_predicates: Sequence[str] = (
            "object",
            "number",
            "_symbol_",
            "_action_",
        ),
        goal_satisfaction_derivations: Iterable[bool | str] | None = None,
    ) -> "RelationDict":
        max_goal_level = max(0, max_goal_level)
        regular_predicates = [
            pred
            for pred in predicates
            if pred.get_name() not in set(top_type_predicates)
        ]
        levels: list[int | None] = list(range(max_goal_level + 1))
        if support_literals:
            levels.append(None)

        relations: dict[str, int] = {}
        for pred in predicates:
            relations[RelationFormatter.format_predicate(pred.get_name())] = (
                pred.get_arity()
            )

        for pred in regular_predicates:
            for level in levels:
                for polarity in (True, False):
                    key = RelationFormatter.format_predicate(
                        pred.get_name(), goal_level=level, polarity=polarity
                    )
                    relations[key] = pred.get_arity()

        derivations = _parse_goal_satisfaction(goal_satisfaction_derivations or (True,))
        for deriv in derivations:
            for pred in regular_predicates:
                for level in levels:
                    for polarity in (True, False):
                        key = RelationFormatter.format_predicate(
                            pred.get_name(),
                            goal_level=level,
                            polarity=polarity,
                            goal_satisfaction=deriv,
                        )
                        relations[key] = pred.get_arity()

        for action in actions:
            relations[RelationFormatter.format_action_schema(action)] = (
                action.get_arity() + 1
            )

        return cls(
            arity=dict(sorted(relations.items())),
            max_goal_level=max_goal_level,
            support_literals=support_literals,
            goal_satisfaction_derivations=derivations,
        )


class HGraphEncoder:
    """pymimir-based HGraph encoder aligned with the C++ implementation."""

    def __init__(
        self,
        domain: pymimir.Domain,
        *,
        symbol_type_id: str = "_symbol_",
        ignore_actions: bool = True,
        add_nullary_predicates: bool = False,
        include_lgan_edges: bool = False,
        include_static: bool = True,
        max_goal_level: int = 0,
        support_literals: bool = False,
        goal_satisfaction_derivations: Iterable[bool | str] | None = None,
        nullary_object_name: str = RelationFormatter.default_nullary_symbol_name,
        lgan_nn_edge_pos: str = "lgan_nn",
    ) -> None:
        self.domain = domain
        self.symbol_type_id = symbol_type_id
        self.ignore_actions = ignore_actions
        self.add_nullary_predicates = add_nullary_predicates
        self.include_lgan_edges = include_lgan_edges
        self.include_static = include_static
        self.max_goal_level = max_goal_level
        self.support_literals = support_literals
        self.nullary_object_name = nullary_object_name
        self.lgan_nn_edge_pos = lgan_nn_edge_pos

        predicates = self.domain.get_predicates()
        actions = self.domain.get_actions() if not ignore_actions else []
        self.relation_dict = RelationDict.build(
            predicates,
            actions,
            max_goal_level=max_goal_level,
            support_literals=support_literals,
            goal_satisfaction_derivations=goal_satisfaction_derivations,
            top_type_predicates=(symbol_type_id, "object", "number", "_action_"),
        )

        edge_types: list[tuple[str, str, str]] = []
        for node_type, arity in self.relation_dict.arity.items():
            effective_arity = 1 if (add_nullary_predicates and arity == 0) else arity
            for pos in range(effective_arity):
                pos_str = str(pos)
                edge_types.append((symbol_type_id, pos_str, node_type))
                edge_types.append((node_type, pos_str, symbol_type_id))
        if include_lgan_edges:
            edge_types.append((lgan_nn_edge_pos, lgan_nn_edge_pos, symbol_type_id))
        self.all_edge_types = edge_types

    def encode_state(
        self,
        state: pymimir.State,
        *,
        goals: Iterable[pymimir.GroundLiteral] | None = None,
        actions: Iterable[pymimir.GroundAction] | None = None,
        subgoal_layers: Sequence[Iterable[pymimir.GroundLiteral]] | None = None,
    ) -> HeteroData:
        builder = PygHeteroBuilder()
        problem = state.get_problem()

        facts = state.get_atoms(
            ignore_static=not self.include_static,
            ignore_fluent=False,
            ignore_derived=False,
        )
        if goals is None:
            goals_list = list(problem.get_goal_condition().get_literals())
        else:
            goals_list = list(goals)

        goal_level_map: dict[pymimir.GroundLiteral, int] = {}
        layered_goals: list[pymimir.GroundLiteral] = []
        for depth, layer in enumerate(chain([goals_list], subgoal_layers or ())):
            layer_list = list(layer)
            layered_goals.extend(layer_list)
            for literal in layer_list:
                goal_level_map[literal] = depth

        actions_list = list(actions or [])

        objects = list(problem.get_objects()) + list(
            problem.get_domain().get_constants()
        )
        objects = sorted(objects, key=lambda obj: obj.get_index())

        relation_to_symbols: dict[str, set[str]] = defaultdict(set)
        symbol_to_relations: dict[str, set[str]] = defaultdict(set)

        self._encode_objects(objects, builder)
        self._encode_facts(
            facts,
            builder,
            relation_to_symbols=relation_to_symbols,
            symbol_to_relations=symbol_to_relations,
        )
        self._encode_literals(
            layered_goals,
            builder,
            goal_level_map=goal_level_map,
            relation_to_symbols=relation_to_symbols,
            symbol_to_relations=symbol_to_relations,
        )
        if not self.ignore_actions:
            for action in actions_list:
                self._encode_action(
                    action,
                    builder,
                    relation_to_symbols=relation_to_symbols,
                    symbol_to_relations=symbol_to_relations,
                )
        if layered_goals and self.relation_dict.goal_satisfaction_derivations:
            satisfied = self._goal_satisfaction_map(facts, layered_goals)
            self._encode_goal_satisfaction(
                satisfied,
                builder,
                goal_level_map=goal_level_map,
                relation_to_symbols=relation_to_symbols,
                symbol_to_relations=symbol_to_relations,
            )
        if self.include_lgan_edges:
            self._add_lgan_nn_edges(
                builder,
                relation_to_symbols=relation_to_symbols,
                symbol_to_relations=symbol_to_relations,
            )

        return self._build_data(builder)

    def _encode_objects(
        self, objects: Sequence[pymimir.Object], builder: PygHeteroBuilder
    ) -> None:
        for obj in objects:
            builder.add_node(
                RelationFormatter.format_object(obj),
                self.symbol_type_id,
            )
        if self.add_nullary_predicates:
            builder.add_node(self.nullary_object_name, self.symbol_type_id)

    def _encode_facts(
        self,
        facts: Iterable[pymimir.GroundAtom],
        builder: PygHeteroBuilder,
        *,
        relation_to_symbols: dict[str, set[str]],
        symbol_to_relations: dict[str, set[str]],
    ) -> None:
        for atom in facts:
            predicate = atom.get_predicate()
            arity = predicate.get_arity()
            if arity == 0:
                if not self.add_nullary_predicates:
                    continue
                objects = [self.nullary_object_name]
            else:
                objects = list(atom.get_terms())

            node = RelationFormatter.format_atom(atom)
            ntype = RelationFormatter.format_predicate(predicate.get_name())
            builder.add_node(node, ntype)
            for pos, obj in enumerate(objects):
                obj_key = (
                    obj
                    if isinstance(obj, str)
                    else RelationFormatter.format_object(obj)
                )
                builder.add_edge(
                    obj_key,
                    node,
                    self.symbol_type_id,
                    ntype,
                    str(pos),
                )
                builder.add_edge(
                    node,
                    obj_key,
                    ntype,
                    self.symbol_type_id,
                    str(pos),
                )

            rel_key = self._relation_key(ntype, node)
            for obj in objects:
                obj_key = (
                    obj
                    if isinstance(obj, str)
                    else RelationFormatter.format_object(obj)
                )
                relation_to_symbols[rel_key].add(obj_key)
                symbol_to_relations[obj_key].add(rel_key)

    def _encode_literals(
        self,
        literals: Iterable[pymimir.GroundLiteral],
        builder: PygHeteroBuilder,
        *,
        goal_level_map: Mapping[pymimir.GroundLiteral, int],
        relation_to_symbols: dict[str, set[str]],
        symbol_to_relations: dict[str, set[str]],
    ) -> None:
        for literal in literals:
            atom = literal.get_atom()
            predicate = atom.get_predicate()
            arity = predicate.get_arity()
            if arity == 0:
                if not self.add_nullary_predicates:
                    continue
                objects = [self.nullary_object_name]
            else:
                objects = list(atom.get_terms())

            goal_level = goal_level_map.get(literal)
            node = RelationFormatter.format_literal(
                literal, goal_level=goal_level, goal_satisfaction=None
            )
            ntype = RelationFormatter.format_predicate(
                predicate.get_name(),
                goal_level=goal_level,
                goal_satisfaction=None,
                polarity=literal.get_polarity(),
            )
            builder.add_node(node, ntype)
            for pos, obj in enumerate(objects):
                obj_key = (
                    obj
                    if isinstance(obj, str)
                    else RelationFormatter.format_object(obj)
                )
                builder.add_edge(
                    obj_key,
                    node,
                    self.symbol_type_id,
                    ntype,
                    str(pos),
                )
                builder.add_edge(
                    node,
                    obj_key,
                    ntype,
                    self.symbol_type_id,
                    str(pos),
                )

            rel_key = self._relation_key(ntype, node)
            for obj in objects:
                obj_key = (
                    obj
                    if isinstance(obj, str)
                    else RelationFormatter.format_object(obj)
                )
                relation_to_symbols[rel_key].add(obj_key)
                symbol_to_relations[obj_key].add(rel_key)

    def _encode_action(
        self,
        action: pymimir.GroundAction,
        builder: PygHeteroBuilder,
        *,
        relation_to_symbols: dict[str, set[str]],
        symbol_to_relations: dict[str, set[str]],
    ) -> None:
        node = RelationFormatter.format_action(action)
        ntype = RelationFormatter.format_action_schema(action.get_action())
        builder.add_node(node, ntype)

        action_symbol_node = f"target:{action.get_index()}|{node}"
        builder.add_node(action_symbol_node, self.symbol_type_id)

        objects: list[pymimir.Object | str] = [action_symbol_node]
        objects.extend(action.get_objects())

        for pos, obj in enumerate(objects):
            obj_key = (
                obj if isinstance(obj, str) else RelationFormatter.format_object(obj)
            )
            builder.add_edge(
                obj_key,
                node,
                self.symbol_type_id,
                ntype,
                str(pos),
            )
            builder.add_edge(
                node,
                obj_key,
                ntype,
                self.symbol_type_id,
                str(pos),
            )

        rel_key = self._relation_key(ntype, node)
        for obj in objects:
            obj_key = (
                obj if isinstance(obj, str) else RelationFormatter.format_object(obj)
            )
            relation_to_symbols[rel_key].add(obj_key)
            symbol_to_relations[obj_key].add(rel_key)

    def _encode_goal_satisfaction(
        self,
        satisfied_goals: Mapping[pymimir.GroundLiteral, bool],
        builder: PygHeteroBuilder,
        *,
        goal_level_map: Mapping[pymimir.GroundLiteral, int],
        relation_to_symbols: dict[str, set[str]],
        symbol_to_relations: dict[str, set[str]],
    ) -> None:
        supported = self.relation_dict.goal_satisfaction_derivations
        for goal, satisfied in satisfied_goals.items():
            if satisfied not in supported:
                continue

            atom = goal.get_atom()
            predicate = atom.get_predicate()
            arity = predicate.get_arity()
            if arity == 0:
                if not self.add_nullary_predicates:
                    continue
                objects = [self.nullary_object_name]
            else:
                objects = list(atom.get_terms())

            goal_level = goal_level_map.get(goal)
            node = RelationFormatter.format_literal(
                goal, goal_level=goal_level, goal_satisfaction=satisfied
            )
            ntype = RelationFormatter.format_predicate(
                predicate.get_name(),
                goal_level=goal_level,
                goal_satisfaction=satisfied,
                polarity=goal.get_polarity(),
            )
            builder.add_node(node, ntype)
            for pos, obj in enumerate(objects):
                obj_key = (
                    obj
                    if isinstance(obj, str)
                    else RelationFormatter.format_object(obj)
                )
                builder.add_edge(
                    obj_key,
                    node,
                    self.symbol_type_id,
                    ntype,
                    str(pos),
                )
                builder.add_edge(
                    node,
                    obj_key,
                    ntype,
                    self.symbol_type_id,
                    str(pos),
                )

            rel_key = self._relation_key(ntype, node)
            for obj in objects:
                obj_key = (
                    obj
                    if isinstance(obj, str)
                    else RelationFormatter.format_object(obj)
                )
                relation_to_symbols[rel_key].add(obj_key)
                symbol_to_relations[obj_key].add(rel_key)

    def _add_lgan_nn_edges(
        self,
        builder: PygHeteroBuilder,
        *,
        relation_to_symbols: Mapping[str, set[str]],
        symbol_to_relations: Mapping[str, set[str]],
    ) -> None:
        symbol_keys = builder.node_keys.get(self.symbol_type_id, [])
        if not symbol_keys:
            return

        target_to_tn: dict[str, set[str]] = {}
        for target in symbol_keys:
            tn = {target}
            for rel_key in symbol_to_relations.get(target, set()):
                tn.update(relation_to_symbols.get(rel_key, set()))
            target_to_tn[target] = tn

        for rel_key, arg_set in relation_to_symbols.items():
            if not arg_set:
                continue
            rel_type, rel_node = self._split_relation_key(rel_key)
            for target_key, tn in target_to_tn.items():
                if target_key in arg_set:
                    continue
                if not arg_set.issubset(tn):
                    continue
                builder.add_edge(
                    rel_node,
                    target_key,
                    rel_type,
                    self.symbol_type_id,
                    self.lgan_nn_edge_pos,
                )
                builder.add_edge(
                    target_key,
                    rel_node,
                    self.symbol_type_id,
                    rel_type,
                    self.lgan_nn_edge_pos,
                )

    @staticmethod
    def _goal_satisfaction_map(
        facts: Sequence[pymimir.GroundAtom],
        goals: Iterable[pymimir.GroundLiteral],
    ) -> dict[pymimir.GroundLiteral, bool]:
        fact_set = set(facts)
        return {
            goal: (goal.get_atom() in fact_set) == goal.get_polarity() for goal in goals
        }

    def _build_data(self, builder: PygHeteroBuilder) -> HeteroData:
        data = HeteroData()

        for node_type, nodes in builder.node_keys.items():
            dim = (
                1
                if node_type == self.symbol_type_id
                else self.relation_dict.arity[node_type]
            )
            data[node_type].x = torch.zeros((len(nodes), dim), dtype=torch.float32)
            data[node_type].node_names = list(nodes)

        for node_type in set(self.relation_dict.arity.keys()) - set(builder.node_keys):
            dim = (
                1
                if node_type == self.symbol_type_id
                else self.relation_dict.arity[node_type]
            )
            data[node_type].x = torch.empty((0, dim), dtype=torch.float32)
            data[node_type].node_names = []

        if self.symbol_type_id not in builder.node_keys:
            data[self.symbol_type_id].x = torch.empty((0, 1), dtype=torch.float32)
            data[self.symbol_type_id].node_names = []

        data.object_names = list(builder.node_keys.get(self.symbol_type_id, []))

        for edge_type, indices in builder.edge_indices.items():
            if indices:
                edge_index = torch.tensor(indices, dtype=torch.long).t().contiguous()
            else:
                edge_index = torch.empty((2, 0), dtype=torch.long)
            data[edge_type].edge_index = edge_index

        for edge_type in set(self.all_edge_types) - set(builder.edge_indices):
            data[edge_type].edge_index = torch.empty((2, 0), dtype=torch.long)

        return data

    @staticmethod
    def _relation_key(node_type: str, node_key: str) -> str:
        return f"{node_type}\n{node_key}"

    @staticmethod
    def _split_relation_key(key: str) -> tuple[str, str]:
        node_type, node_key = key.split("\n", 1)
        return node_type, node_key
