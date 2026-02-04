from __future__ import annotations

from itertools import chain
from typing import (
    Collection,
    Dict,
    Iterable,
    List,
    NamedTuple,
    Sequence,
)

import torch_geometric as pyg
from torch_geometric.data import HeteroData

from mifrost.utils.misc import monkeypatch
import pymimir

from .base_encoder import (
    EncoderFactory,
)
from .hetero_encoder import HGraphEncoder
from .pyg_builder import PygBuilderBase
from .relation_dict import RelationDict
from .accessors import (
    atom_objects,
    atom_signature,
    atoms_equal,
    literal_atom,
    literal_polarity,
    predicate,
    predicate_name,
)


class PredicateEdgeType(NamedTuple):
    src_type: str
    pos: str
    dst_type: str


class SuccessorPredicate:  # lightweight wrapper to mimic predicate interface
    def __init__(self, predicate: pymimir.Predicate, new_name: str):
        self._predicate = predicate
        self._name = new_name

    def get_name(self):
        return self._name

    def __getattr__(self, item):
        return getattr(self._predicate, item)

    # Forward common special methods explicitly
    def __eq__(self, other):
        # Prefer to compare underlying predicate semantics if needed:
        try:
            return predicate_name(self) == predicate_name(other)
        except AttributeError:
            return NotImplemented

    def __hash__(self):
        return hash(predicate_name(self))

    def __str__(self):
        return self._name


class SuccessorAtom:  # lightweight wrapper to mimic atom interface
    def __init__(self, atom: pymimir.GroundAtom, new_predicate: pymimir.Predicate):
        self._atom = atom
        self._predicate = new_predicate

    def get_predicate(self):
        return self._predicate

    def get_terms(self):
        return self._atom.get_terms()

    def __getattr__(self, item):
        return getattr(self._atom, item)

    # Forward common special methods explicitly
    def __eq__(self, other):
        # Prefer to compare underlying atom semantics if needed:
        try:
            return atoms_equal(self, other)
        except AttributeError:
            return NotImplemented

    def __hash__(self):
        return hash(atom_signature(self))

    def __str__(self):
        parts = [predicate_name(self._predicate)]
        parts.extend(object.get_name() for object in atom_objects(self))
        return "(" + " ".join(parts) + ")"


class SuccessorLiteral:  # lightweight wrapper to mimic literal interface
    def __init__(self, literal: pymimir.GroundLiteral, new_atom: pymimir.GroundAtom):
        self._literal = literal
        self._atom = new_atom

    def get_atom(self):
        return self._atom

    def __getattr__(self, item):
        return getattr(self._literal, item)

    # Forward common special methods explicitly
    def __eq__(self, other):
        # Prefer to compare underlying literal semantics if needed:
        try:
            return literal_polarity(self) == literal_polarity(other) and atoms_equal(
                literal_atom(self), literal_atom(other)
            )
        except AttributeError:
            return NotImplemented

    def __hash__(self):
        return hash((literal_polarity(self), atom_signature(literal_atom(self))))


class TransitionHGraphEncoder(HGraphEncoder):
    successor_predicate_suffix = "[suc]"

    def __init__(
        self,
        domain: pymimir.Domain,
        *args,
        include_lgan_edges: bool = False,
        lgan_nn_edge_pos: str = "lgan_nn",
        **kwargs,
    ) -> None:
        self.pred_to_succ_pred: Dict[pymimir.Predicate, SuccessorPredicate] = {
            p: SuccessorPredicate(
                p, predicate_name(p) + self.successor_predicate_suffix
            )
            for p in domain.get_predicates()
        }
        # monkeypatch domain's get_predicates for super().__init__ to include successor predicates
        with monkeypatch(
            domain,
            "get_predicates",
            lambda: tuple(
                chain(
                    self.pred_to_succ_pred.keys(),
                    self.pred_to_succ_pred.values(),
                )
            ),
        ):
            super().__init__(
                domain,
                *args,
                include_lgan_edges=include_lgan_edges,
                lgan_nn_edge_pos=lgan_nn_edge_pos,
                **kwargs,
            )

    def _encode(
        self,
        builder: PygBuilderBase,
        facts: Sequence[pymimir.GroundAtom],
        goals: Sequence[pymimir.GroundLiteral],
        actions: Sequence[pymimir.GroundAction],
        successor: pymimir.State | Iterable[pymimir.GroundAtom] = None,
        **kwargs,
    ) -> None:
        assert successor is not None, "Successor state must be provided."
        # patch all atoms in successor_facts and successor_subgoal_layers to have successor predicates
        patched_successor_facts = [
            self._patch_atom(atom)
            for atom in (
                successor.get_atoms(ignore_static=True)
                if isinstance(successor, pymimir.State)
                else successor
            )
        ]
        # combine facts
        all_facts = list(facts) + patched_successor_facts

        # we are not patching top-level goals, because doubling them would not be needed in the base setting
        super()._encode(
            builder,
            all_facts,
            goals,
            actions,
            **kwargs,
        )

        if self.relation_dict.goal_satisfaction_derivations and goals:
            glm = builder.graph_attrs["goal_level_map"]
            patched_goals = []
            for goal in goals:
                patched_goal = self._patch_literal(goal)
                patched_goals.append(patched_goal)
                if patched_goal not in glm:
                    glm[patched_goal] = glm[goal]
            satisfied_goals = self._goal_satisfaction_map(
                patched_successor_facts, patched_goals
            )
            self._encode_goal_satisfaction(satisfied_goals, builder, goal_level_map=glm)

    def encode(self, *args, **kwargs) -> HeteroData:
        # Call base encode which will call our _encode
        return super().encode(*args, **kwargs)

    def _patch_atom(self, atom: pymimir.GroundAtom):
        return SuccessorAtom(
            atom,
            self.pred_to_succ_pred[predicate(atom)],
        )

    def _patch_literal(self, literal: pymimir.GroundLiteral):
        patched_atom = self._patch_atom(literal_atom(literal))
        return SuccessorLiteral(
            literal,
            patched_atom,
        )


def get_facts(
    state: pymimir.State | Iterable[pymimir.GroundAtom], problem: pymimir.Problem | None
) -> tuple[set[pymimir.GroundAtom], pymimir.Problem | None]:
    if isinstance(state, pymimir.State):
        problem = problem or state.get_problem()
        successor_facts = set(state.get_atoms(ignore_static=True))
    else:
        successor_facts = set(state)
    return successor_facts, problem


class TransitionEffectsHGraphEncoder(HGraphEncoder):
    """
    An encoder to represent state transitions as heterogeneous graphs with objects and predicates as vertices
    and edges (i, j) whenever a predicate p(..., i, j, ...) holds in the state.
    Additionally, only predicates that change between the two states are included.

    """

    def __init__(
        self,
        *args,
        relation_dict: RelationDict | None = None,
        support_literals: bool = True,
        include_lgan_edges: bool = False,
        lgan_nn_edge_pos: str = "lgan_nn",
        **kwargs,
    ) -> None:
        if relation_dict is not None:
            assert relation_dict.support_literals, (
                f"{self.__class__.__name__} requires support_literals=True in RelationDict."
            )
        else:
            assert support_literals, (
                f"{self.__class__.__name__} requires support_literals=True."
            )
        super().__init__(
            *args,
            relation_dict=relation_dict,
            support_literals=support_literals,
            include_lgan_edges=include_lgan_edges,
            lgan_nn_edge_pos=lgan_nn_edge_pos,
            **kwargs,
        )

    def _encode(
        self,
        builder: PygBuilderBase,
        facts: Sequence[pymimir.GroundAtom],
        goals: Sequence[pymimir.GroundLiteral],
        actions: Sequence[pymimir.GroundAction],
        successor: pymimir.State | Iterable[pymimir.GroundAtom] = None,
        problem: pymimir.Problem | None = None,
        _state: pymimir.State | Iterable[pymimir.GroundAtom] | None = None,
        **kwargs,
    ) -> None:
        assert successor is not None, "Successor state must be provided."
        base_facts, problem = get_facts(_state or facts, problem)
        successor_facts, problem = get_facts(successor, problem)
        assert problem is not None, (
            "If state and successor is given as an iterable of atoms, problem must be provided."
        )
        successor_literals, added_facts, removed_facts = _compute_change_literals(
            base_facts, successor_facts, problem
        )

        super()._encode(
            builder,
            facts,
            goals,
            actions,
            **kwargs,
        )

        if self.relation_dict.goal_satisfaction_derivations and goals:
            sat_goals = _compute_change_satisfaction(added_facts, removed_facts, goals)
            self._encode_goal_satisfaction(sat_goals, builder)

        self._encode_literals(successor_literals, builder, goal_level_map={})

    def encode(self, *args, **kwargs) -> HeteroData:
        return super().encode(*args, **kwargs)


def _compute_change_satisfaction(
    added_facts: Collection[pymimir.GroundAtom],
    removed_facts: Collection[pymimir.GroundAtom],
    goals: Sequence[pymimir.GroundLiteral],
) -> dict[pymimir.GroundLiteral, str]:
    sat_goals = {
        goal: "+"
        for goal in goals
        if any(atoms_equal(literal_atom(goal), fact) for fact in added_facts)
        == literal_polarity(goal)
    } | {
        goal: "-"
        for goal in goals
        if any(atoms_equal(literal_atom(goal), fact) for fact in removed_facts)
        != literal_polarity(goal)
    }
    return sat_goals


def _compute_change_literals(
    facts: set[pymimir.GroundAtom],
    successor_facts: set[pymimir.GroundAtom],
    problem: pymimir.Problem,
) -> tuple[
    list[pymimir.GroundLiteral], set[pymimir.GroundAtom], set[pymimir.GroundAtom]
]:
    added = {
        atom
        for atom in successor_facts
        if not any(atoms_equal(atom, base_atom) for base_atom in facts)
    }
    removed = {
        atom
        for atom in facts
        if not any(atoms_equal(atom, succ_atom) for succ_atom in successor_facts)
    }
    literals: List[pymimir.GroundLiteral] = []
    for atom in added:
        literals.append(pymimir.GroundLiteral.new(atom, True, problem))
    for atom in removed:
        literals.append(pymimir.GroundLiteral.new(atom, False, problem))
    return literals, added, removed
