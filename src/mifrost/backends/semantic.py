"""Canonical semantic identities shared by planning backends.

The records in this module deliberately contain no repository-local indices or
backend objects. They are suitable for parity checks and form the executable
specification for the compact native representation used by encoders.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol, runtime_checkable


class PredicateCategory(StrEnum):
    """Semantic predicate category used by Mimir and Tyr."""

    STATIC = "static"
    FLUENT = "fluent"
    DERIVED = "derived"


@dataclass(frozen=True, order=True, slots=True)
class PredicateKey:
    """Backend-independent predicate identity."""

    category: PredicateCategory
    name: str
    arity: int

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("predicate name must not be empty")
        if self.arity < 0:
            raise ValueError("predicate arity must be non-negative")


@dataclass(frozen=True, order=True, slots=True)
class ActionSchemaKey:
    """Backend-independent lifted action identity."""

    name: str
    arity: int

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("action name must not be empty")
        if self.arity < 0:
            raise ValueError("action arity must be non-negative")


@dataclass(frozen=True, order=True, slots=True)
class AtomKey:
    """Predicate plus its ordered object binding."""

    predicate: PredicateKey
    objects: tuple[str, ...]

    def __post_init__(self) -> None:
        if len(self.objects) != self.predicate.arity:
            raise ValueError(
                f"atom {self.predicate.name!r} expects {self.predicate.arity} "
                f"objects, got {len(self.objects)}"
            )
        if any(not name for name in self.objects):
            raise ValueError("atom object names must not be empty")


@dataclass(frozen=True, order=True, slots=True)
class LiteralKey:
    """A signed semantic atom."""

    atom: AtomKey
    polarity: bool


@dataclass(frozen=True, order=True, slots=True)
class GroundActionKey:
    """Lifted action identity plus its ordered object binding."""

    action: ActionSchemaKey
    objects: tuple[str, ...]

    def __post_init__(self) -> None:
        if len(self.objects) != self.action.arity:
            raise ValueError(
                f"action {self.action.name!r} expects {self.action.arity} "
                f"objects, got {len(self.objects)}"
            )
        if any(not name for name in self.objects):
            raise ValueError("action object names must not be empty")


def _predicate_sort_key(value: PredicateKey) -> tuple[str, str, int]:
    return value.category.value, value.name, value.arity


def _action_sort_key(value: ActionSchemaKey) -> tuple[str, int]:
    return value.name, value.arity


def _atom_sort_key(value: AtomKey) -> tuple[str, str, int, tuple[str, ...]]:
    predicate = value.predicate
    return predicate.category.value, predicate.name, predicate.arity, value.objects


def _literal_sort_key(
    value: LiteralKey,
) -> tuple[str, str, int, tuple[str, ...], bool]:
    return (*_atom_sort_key(value.atom), value.polarity)


@dataclass(frozen=True, slots=True)
class DomainSnapshot:
    """Canonical domain schema independent of backend iteration order."""

    name: str
    predicates: tuple[PredicateKey, ...]
    actions: tuple[ActionSchemaKey, ...]

    @classmethod
    def canonical(
        cls,
        *,
        name: str,
        predicates: Iterable[PredicateKey],
        actions: Iterable[ActionSchemaKey],
    ) -> DomainSnapshot:
        return cls(
            name=str(name),
            predicates=tuple(sorted(predicates, key=_predicate_sort_key)),
            actions=tuple(sorted(actions, key=_action_sort_key)),
        )


@dataclass(frozen=True, slots=True)
class ProblemSnapshot:
    """Canonical problem metadata needed by backend-neutral encoders."""

    name: str
    domain_name: str
    objects: tuple[str, ...]
    static_atoms: tuple[AtomKey, ...]
    goals: tuple[LiteralKey, ...]

    @classmethod
    def canonical(
        cls,
        *,
        name: str,
        domain_name: str,
        objects: Iterable[object],
        static_atoms: Iterable[AtomKey],
        goals: Iterable[LiteralKey],
    ) -> ProblemSnapshot:
        object_names = tuple(sorted(str(value) for value in objects))
        if len(set(object_names)) != len(object_names):
            raise ValueError("problem object names must be unique")
        return cls(
            name=str(name),
            domain_name=str(domain_name),
            objects=object_names,
            static_atoms=tuple(sorted(static_atoms, key=_atom_sort_key)),
            goals=tuple(sorted(goals, key=_literal_sort_key)),
        )


@dataclass(frozen=True, slots=True)
class StateSnapshot:
    """Canonical true propositional facts for one state."""

    static_atoms: tuple[AtomKey, ...]
    fluent_atoms: tuple[AtomKey, ...]
    derived_atoms: tuple[AtomKey, ...]

    @classmethod
    def canonical(
        cls,
        *,
        static_atoms: Iterable[AtomKey] = (),
        fluent_atoms: Iterable[AtomKey] = (),
        derived_atoms: Iterable[AtomKey] = (),
    ) -> StateSnapshot:
        return cls(
            static_atoms=tuple(sorted(static_atoms, key=_atom_sort_key)),
            fluent_atoms=tuple(sorted(fluent_atoms, key=_atom_sort_key)),
            derived_atoms=tuple(sorted(derived_atoms, key=_atom_sort_key)),
        )

    @property
    def atoms(self) -> tuple[AtomKey, ...]:
        """All true atoms, grouped in stable category order."""

        return self.static_atoms + self.fluent_atoms + self.derived_atoms


@runtime_checkable
class SnapshotReader(Protocol):
    """Per-problem adapter contract without global backend selection state."""

    backend_name: str

    def domain_snapshot(self) -> DomainSnapshot: ...

    def problem_snapshot(self) -> ProblemSnapshot: ...

    def state_snapshot(self, state: object) -> StateSnapshot: ...

    def action_key(self, action: object) -> GroundActionKey: ...
