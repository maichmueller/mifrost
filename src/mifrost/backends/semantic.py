"""Canonical semantic identities shared by planning backends.

The records in this module deliberately contain no repository-local indices or
backend objects. They are suitable for parity checks and form the executable
specification for the compact native representation used by encoders.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
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
    """Canonical domain schema independent of backend iteration order.

    ``types`` is the domain's declared object-type vocabulary (see
    :func:`ProblemSnapshot.canonical`'s ``object_types``), scoped to the
    *domain* rather than any one problem -- the same stability guarantee
    ``predicates``/``actions`` already have -- and ``None`` when the backend
    cannot resolve type declarations at all (see
    ``mifrost.backends.pytyr.PyTyrSnapshotReader.domain_snapshot``).

    ``type_bases`` records each declared type's *direct* parents, one entry
    per name in ``types``. It is the edge set of the PDDL type hierarchy,
    left untransitive on purpose: the closure is a derived quantity, and the
    consumer that needs it (``SparseAtomTypeSchema.ancestor_matrix``) also
    owns the type-id assignment the closure has to be indexed by. A root
    type maps to an empty tuple. ``None`` exactly when ``types`` is ``None``.
    """

    name: str
    predicates: tuple[PredicateKey, ...]
    actions: tuple[ActionSchemaKey, ...]
    types: tuple[str, ...] | None = None
    type_bases: Mapping[str, tuple[str, ...]] | None = None

    @classmethod
    def canonical(
        cls,
        *,
        name: str,
        predicates: Iterable[PredicateKey],
        actions: Iterable[ActionSchemaKey],
        types: Iterable[str] | None = None,
        type_bases: Mapping[str, Iterable[str]] | None = None,
    ) -> DomainSnapshot:
        resolved_types = (
            None if types is None else tuple(sorted(str(value) for value in types))
        )
        if type_bases is None:
            resolved_bases: Mapping[str, tuple[str, ...]] | None = None
        elif resolved_types is None:
            raise ValueError("type_bases requires types to be supplied as well")
        else:
            known = set(resolved_types)
            resolved_bases = MappingProxyType(
                {
                    name: tuple(sorted(str(base) for base in bases))
                    for name, bases in sorted(
                        (str(key), value) for key, value in type_bases.items()
                    )
                }
            )
            unknown = sorted(set(resolved_bases) - known)
            if unknown:
                raise ValueError(
                    f"type_bases names types absent from the vocabulary: {unknown}"
                )
            dangling = sorted(
                {base for bases in resolved_bases.values() for base in bases} - known
            )
            if dangling:
                raise ValueError(
                    f"type_bases refers to undeclared parent types: {dangling}"
                )
        return cls(
            name=str(name),
            predicates=tuple(sorted(predicates, key=_predicate_sort_key)),
            actions=tuple(sorted(actions, key=_action_sort_key)),
            types=resolved_types,
            type_bases=resolved_bases,
        )


@dataclass(frozen=True, slots=True)
class ProblemSnapshot:
    """Canonical problem metadata needed by backend-neutral encoders.

    ``object_types`` is the most specific declared PDDL type name for each
    entry of ``objects``, index-aligned with it (both are permuted by the
    same name-sort in :meth:`canonical`); ``None`` when the backend cannot
    resolve object types at all. See
    ``mifrost.encoders.custom.state_view.StateView.object_types`` for which
    backends support this and why.
    """

    name: str
    domain_name: str
    objects: tuple[str, ...]
    static_atoms: tuple[AtomKey, ...]
    goals: tuple[LiteralKey, ...]
    object_types: tuple[str, ...] | None = None

    @classmethod
    def canonical(
        cls,
        *,
        name: str,
        domain_name: str,
        objects: Iterable[object],
        static_atoms: Iterable[AtomKey],
        goals: Iterable[LiteralKey],
        object_types: Iterable[str] | None = None,
    ) -> ProblemSnapshot:
        object_list = [str(value) for value in objects]
        if object_types is None:
            type_list: list[str] | None = None
        else:
            type_list = [str(value) for value in object_types]
            if len(type_list) != len(object_list):
                raise ValueError(
                    "object_types must have exactly one entry per object: got "
                    f"{len(type_list)} types for {len(object_list)} objects"
                )
        order = sorted(range(len(object_list)), key=lambda index: object_list[index])
        object_names = tuple(object_list[index] for index in order)
        if len(set(object_names)) != len(object_names):
            raise ValueError("problem object names must be unique")
        resolved_types = (
            None if type_list is None else tuple(type_list[index] for index in order)
        )
        return cls(
            name=str(name),
            domain_name=str(domain_name),
            objects=object_names,
            static_atoms=tuple(sorted(static_atoms, key=_atom_sort_key)),
            goals=tuple(sorted(goals, key=_literal_sort_key)),
            object_types=resolved_types,
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
