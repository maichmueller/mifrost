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


#: PDDL's builtin numeric-fluent type. Pymimir's ``Domain.get_types()``
#: always reports it -- even for domains with no ``:functions`` section at
#: all (verified against every fixture under ``data/pddl/``) -- but by PDDL
#: semantics it can only ever type a function's numeric return value, never
#: an object: no ``:objects``/``:constants`` declaration can name it.
#: Excluding it from the object-type vocabulary isn't a heuristic tuned to
#: today's fixtures, it is a structural fact about what the type means, so
#: unlike every other declared type it would otherwise be a permanently
#: unreachable id (see :func:`domain_snapshot`).
_NUMBER_TYPE_NAME = "number"


def _object_type_name(value: Any) -> str:
    """Resolve one pymimir ``Object``'s most specific declared PDDL type.

    ``Object.get_bases()`` returns the object's own declared type(s) -- a
    list because PDDL's ``either`` typing allows more than one -- not the
    full ancestor chain (a spanner-domain ``man`` object reports only
    ``['man']``, not ``['man', 'locatable', 'object']``; see
    ``StateView.object_types`` for why only the most specific type is
    exposed). An untyped object still resolves to the implicit PDDL root
    type ``"object"``, since pymimir's ``Domain.get_types()`` always
    declares it. A real ``either``-typed object (more than one declared
    base) has no single "the" type, so this raises rather than guessing.
    """

    bases = list(value.get_bases())
    if len(bases) != 1:
        raise ValueError(
            f"pymimir object {value.get_name()!r} declares {len(bases)} types "
            "via PDDL 'either' typing; a single per-object type id needs "
            "exactly one declared type per object"
        )
    return str(bases[0].get_name())


def _literal_key(literal: Any) -> LiteralKey:
    return LiteralKey(_atom_key(literal.get_atom()), bool(literal.get_polarity()))


class PymimirSnapshotReader:
    """Read canonical snapshots from one Pymimir wrapper problem."""

    backend_name = "pymimir"

    _SNAPSHOT_CACHE_LIMIT = 4096

    def __init__(self, problem: pymimir.Problem) -> None:
        if not isinstance(problem, pymimir.Problem):
            raise TypeError(
                f"PymimirSnapshotReader expects pymimir.Problem, got {type(problem)!r}"
            )
        self._problem = problem
        self._state_snapshot_cache: dict[Any, StateSnapshot] = {}
        self._state_snapshot_cacheable: bool | None = None

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
        types = (
            str(value.get_name())
            for value in domain.get_types()
            if str(value.get_name()) != _NUMBER_TYPE_NAME
        )
        return DomainSnapshot.canonical(
            name=domain.get_name(), predicates=predicates, actions=actions, types=types
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
        objects = list(self._problem.get_objects())
        return ProblemSnapshot.canonical(
            name=self._problem.get_name(),
            domain_name=self._problem.get_domain().get_name(),
            objects=(value.get_name() for value in objects),
            object_types=(_object_type_name(value) for value in objects),
            static_atoms=static_atoms,
            goals=goals,
        )

    def state_snapshot(self, state: object) -> StateSnapshot:
        if not isinstance(state, pymimir.State):
            raise TypeError(
                f"pymimir state snapshot expects pymimir.State, got {type(state)!r}"
            )
        return self._state_snapshot_memoized(state)

    def _state_snapshot_memoized(self, state: pymimir.State) -> StateSnapshot:
        # Planning states are immutable value objects, so the snapshot is a
        # pure function of content; memoize per reader (bounded, and
        # transparently disabled for unhashable states).
        if self._state_snapshot_cacheable is not False:
            try:
                cached = self._state_snapshot_cache.get(state)
            except TypeError:
                self._state_snapshot_cacheable = False
            else:
                if cached is not None:
                    return cached
                snapshot = self._compute_state_snapshot(state)
                cache = self._state_snapshot_cache
                if len(cache) >= self._SNAPSHOT_CACHE_LIMIT:
                    cache.clear()
                cache[state] = snapshot
                return snapshot
        return self._compute_state_snapshot(state)

    def _compute_state_snapshot(self, state: pymimir.State) -> StateSnapshot:
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
