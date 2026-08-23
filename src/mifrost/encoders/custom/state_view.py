"""Planner-neutral problem and state records for custom encoders.

`StateView` wraps one pymimir problem or one pytyr planning task and exposes
its schema through backend-free dataclasses (`Atom`, `Literal`,
`PredicateInfo`, `ActionInfo`). It builds on the existing snapshot layer
(`PymimirSnapshotReader`, `PyTyrSnapshotReader`) so two views of the same PDDL
files produce identical records.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from ...backends._derived_runtime import (
    _looks_like_pytyr_task,
    _normalize_backend,
)
from ...backends.semantic import AtomKey, GroundActionKey, LiteralKey

_CATEGORIES = frozenset({"static", "fluent", "derived"})


@dataclass(frozen=True)
class Atom:
    """A ground atom: predicate name plus its ordered object names."""

    predicate: str
    args: tuple[str, ...]

    @property
    def display(self) -> str:
        """Return the s-expression form, e.g. ``(on a b)`` or ``(handempty)``."""

        if self.args:
            return f"({self.predicate} {' '.join(self.args)})"
        return f"({self.predicate})"


@dataclass(frozen=True)
class Literal:
    """A signed ground atom."""

    atom: Atom
    positive: bool


@dataclass(frozen=True)
class PredicateInfo:
    """One domain predicate with its semantic category."""

    name: str
    arity: int
    category: str

    def __post_init__(self) -> None:
        if self.category not in _CATEGORIES:
            raise ValueError(
                f"predicate category must be one of {sorted(_CATEGORIES)}, "
                f"got {self.category!r}"
            )


@dataclass(frozen=True)
class ActionInfo:
    """One lifted action schema identity."""

    name: str
    arity: int


def _atom_from_key(key: AtomKey) -> Atom:
    return Atom(key.predicate.name, key.objects)


def _literal_from_key(key: LiteralKey) -> Literal:
    return Literal(_atom_from_key(key.atom), key.polarity)


def _action_atom_from_key(key: GroundActionKey) -> Atom:
    return Atom(key.action.name, key.objects)


class StateView:
    """Read-only planner-neutral view of one problem/task and its states."""

    def __init__(self, source: Any, *, backend: str | None = None) -> None:
        """Wrap a pymimir Problem or pytyr PlanningTask.

        The backend is auto-detected from the source type unless ``backend=``
        names one explicitly ('pymimir' or 'pytyr').
        """

        selected = _normalize_backend(backend)
        if selected is None:
            selected = "pytyr" if _looks_like_pytyr_task(source) else "pymimir"
        if selected == "pytyr":
            from ...backends.pytyr import PyTyrSnapshotReader

            self._reader = PyTyrSnapshotReader(source)
        else:
            from ...backends.pymimir import PymimirSnapshotReader

            self._reader = PymimirSnapshotReader(source)
        self.backend: str = selected
        domain = self._reader.domain_snapshot()
        problem = self._reader.problem_snapshot()
        self._problem_name: str = problem.name
        self._objects: list[str] = list(problem.objects)
        self._predicates: tuple[PredicateInfo, ...] = tuple(
            PredicateInfo(key.name, key.arity, key.category.value)
            for key in domain.predicates
        )
        self._action_schemas: tuple[ActionInfo, ...] = tuple(
            ActionInfo(key.name, key.arity) for key in domain.actions
        )
        self._static_facts: tuple[Atom, ...] = tuple(
            _atom_from_key(key) for key in problem.static_atoms
        )
        self._goal_literals: tuple[Literal, ...] = tuple(
            _literal_from_key(key) for key in problem.goals
        )

    @property
    def objects(self) -> list[str]:
        """All constant object names of the problem, sorted."""

        return list(self._objects)

    @property
    def object_types(self) -> list[str] | None:
        """Per-object type names, or ``None`` when unavailable.

        Neither the pymimir wrapper ``Object`` nor the pytyr ``Object``
        exposes type information today (both offer only ``get_index`` and
        ``get_name``), so no uniform cross-backend record exists and this
        property currently always returns ``None``.
        """

        return None

    @property
    def predicates(self) -> list[PredicateInfo]:
        """Domain predicates sorted by category, then name, then arity."""

        return list(self._predicates)

    @property
    def action_schemas(self) -> list[ActionInfo]:
        """Lifted action schemas sorted by name, then arity."""

        return list(self._action_schemas)

    @property
    def static_facts(self) -> tuple[Atom, ...]:
        """Static atoms that hold in every state of this problem."""

        return self._static_facts

    @property
    def problem_name(self) -> str:
        """The problem's declared name (best effort)."""

        return self._problem_name

    def state_facts(self, state: Any) -> tuple[Atom, ...]:
        """True fluent atoms followed by true derived atoms, stable order."""

        snapshot = self._reader.state_snapshot(state)
        return tuple(
            _atom_from_key(key)
            for key in (*snapshot.fluent_atoms, *snapshot.derived_atoms)
        )

    def goal_literals(self, state: Any) -> tuple[Literal, ...]:
        """The problem's default goal literals."""

        del state
        return self._goal_literals

    def neutral_actions(self, state: Any, actions: Iterable[Any]) -> tuple[Atom, ...]:
        """Map native ground actions to ``Atom(predicate=name, args=objects)``."""

        del state
        atoms: list[Atom] = []
        for action in actions:
            atom = self._action_from_native(action)
            if atom is None:
                raise TypeError(
                    "actions lane leaf is not a recognized "
                    f"{self.backend} ground action: {type(action)!r}"
                )
            atoms.append(atom)
        return tuple(atoms)

    def _action_from_native(self, action: Any) -> Atom | None:
        try:
            return _action_atom_from_key(self._reader.action_key(action))
        except TypeError:
            return None

    def neutral_literals(
        self, values: Iterable[Any], *, field: str = "goals"
    ) -> tuple[Literal, ...]:
        """Map native ground literals (or neutral records) to `Literal`.

        Already-neutral `Literal` records pass through unchanged, as do
        semantic `LiteralKey` values. Unknown leaf types raise TypeError
        naming the lane and the backend.
        """

        literals: list[Literal] = []
        for value in values:
            literal = self._literal_from_native(value)
            if literal is None:
                raise TypeError(
                    f"{field} lane leaf is not a recognized "
                    f"{self.backend} ground literal: {type(value)!r}"
                )
            literals.append(literal)
        return tuple(literals)

    def _literal_from_native(self, value: Any) -> Literal | None:
        if isinstance(value, Literal):
            return value
        if isinstance(value, LiteralKey):
            return _literal_from_key(value)
        try:
            if self.backend == "pytyr":
                from ...backends.pytyr import SemanticPlanningTaskAdapter

                return _literal_from_key(
                    SemanticPlanningTaskAdapter._literal_key(value)
                )
            from ...backends.pymimir import _literal_key

            return _literal_from_key(_literal_key(value))
        except (AttributeError, TypeError, ValueError):
            return None


__all__ = [
    "ActionInfo",
    "Atom",
    "Literal",
    "PredicateInfo",
    "StateView",
]
