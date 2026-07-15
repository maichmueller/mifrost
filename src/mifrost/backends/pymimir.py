"""Pymimir implementation of the semantic snapshot contract."""

from __future__ import annotations

from typing import Any

import pymimir

from .semantic import (
    ActionSchemaKey,
    AtomKey,
    DomainSnapshot,
    GroundActionKey,
    LiteralKey,
    PredicateCategory,
    PredicateKey,
    ProblemSnapshot,
    StateSnapshot,
)


def _category(value: Any) -> PredicateCategory:
    if value.is_static():
        return PredicateCategory.STATIC
    if value.is_fluent():
        return PredicateCategory.FLUENT
    if value.is_derived():
        return PredicateCategory.DERIVED
    raise ValueError(f"pymimir value has no recognized predicate category: {value!r}")


def _predicate_key(predicate: Any, category: PredicateCategory) -> PredicateKey:
    return PredicateKey(category, str(predicate.get_name()), int(predicate.get_arity()))


def _atom_key(atom: Any) -> AtomKey:
    predicate = atom.get_predicate()
    objects = atom.get_objects() if hasattr(atom, "get_objects") else atom.get_terms()
    return AtomKey(
        _predicate_key(predicate, _category(atom)),
        tuple(str(value.get_name()) for value in objects),
    )


def _literal_key(literal: Any) -> LiteralKey:
    return LiteralKey(_atom_key(literal.get_atom()), bool(literal.get_polarity()))


class PymimirSnapshotReader:
    """Read canonical snapshots from one Pymimir wrapper problem."""

    backend_name = "pymimir"

    def __init__(self, problem: pymimir.Problem) -> None:
        if not isinstance(problem, pymimir.Problem):
            raise TypeError(
                f"PymimirSnapshotReader expects pymimir.Problem, got {type(problem)!r}"
            )
        self._problem = problem

    def domain_snapshot(self) -> DomainSnapshot:
        domain = self._problem.get_domain()
        predicates = (
            _predicate_key(predicate, _category(predicate))
            for predicate in domain.get_predicates()
        )
        actions = (
            ActionSchemaKey(str(action.get_name()), int(action.get_arity()))
            for action in domain.get_actions()
        )
        return DomainSnapshot.canonical(
            name=domain.get_name(), predicates=predicates, actions=actions
        )

    def problem_snapshot(self) -> ProblemSnapshot:
        initial_state = self._problem.get_initial_state()
        static_atoms = (
            _atom_key(atom) for atom in initial_state.get_atoms() if atom.is_static()
        )
        goals = (
            _literal_key(literal)
            for literal in self._problem.get_goal_condition().get_literals()
        )
        return ProblemSnapshot.canonical(
            name=self._problem.get_name(),
            domain_name=self._problem.get_domain().get_name(),
            objects=(value.get_name() for value in self._problem.get_objects()),
            static_atoms=static_atoms,
            goals=goals,
        )

    def state_snapshot(self, state: object) -> StateSnapshot:
        if not isinstance(state, pymimir.State):
            raise TypeError(
                f"pymimir state snapshot expects pymimir.State, got {type(state)!r}"
            )
        groups: dict[PredicateCategory, list[AtomKey]] = {
            category: [] for category in PredicateCategory
        }
        for atom in state.get_atoms():
            groups[_category(atom)].append(_atom_key(atom))
        return StateSnapshot.canonical(
            static_atoms=groups[PredicateCategory.STATIC],
            fluent_atoms=groups[PredicateCategory.FLUENT],
            derived_atoms=groups[PredicateCategory.DERIVED],
        )

    def action_key(self, action: object) -> GroundActionKey:
        if not isinstance(action, pymimir.GroundAction):
            raise TypeError(
                f"pymimir action key expects pymimir.GroundAction, got {type(action)!r}"
            )
        schema = action.get_action()
        return GroundActionKey(
            ActionSchemaKey(str(schema.get_name()), int(schema.get_arity())),
            tuple(str(value.get_name()) for value in action.get_objects()),
        )
