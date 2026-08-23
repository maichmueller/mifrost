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

#: Bounded memo size for `StateView.state_facts`; states are immutable value
#: objects, so caching their fact tuples cannot change any output.
_STATE_FACTS_CACHE_LIMIT = 4096


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


@dataclass(frozen=True)
class Effect:
    """One conditional effect: guard literals plus the literals it applies."""

    condition: tuple[Literal, ...]
    literals: tuple[Literal, ...]


@dataclass(frozen=True)
class ActionStructure:
    """One lifted action schema with canonicalized precondition and effects.

    Parameters are renamed to positional canonical names (``?a0``, ``?a1``,
    ...) so pymimir and pytyr views of the same PDDL action produce
    identical records; literal arguments referencing a parameter use its
    canonical name, constant arguments keep their object names.

    Domains with ``:disjunctive-preconditions`` are flattened by both
    backends into duplicate-named schemas — one per disjunct, each with its
    single-disjunct precondition. Ordering of the returned tuple is
    canonicalized over the full record (name, arity, preconditions,
    effects), not just the name, so both backends agree even when their
    native expansion orders differ.
    """

    name: str
    arity: int
    parameters: tuple[str, ...]
    precondition: tuple[Literal, ...]
    effects: tuple[Effect, ...]


def _atom_from_key(key: AtomKey) -> Atom:
    return Atom(key.predicate.name, key.objects)


def _literal_from_key(key: LiteralKey) -> Literal:
    return Literal(_atom_from_key(key.atom), key.polarity)


def _action_atom_from_key(key: GroundActionKey) -> Atom:
    return Atom(key.action.name, key.objects)


def _canonical_literal(
    predicate: str, terms: Iterable[str], positive: bool, rename: dict[str, str]
) -> Literal:
    args = tuple(rename.get(term, term) for term in terms)
    return Literal(Atom(predicate, args), positive)


def _sorted_literals(literals: Iterable[Literal]) -> tuple[Literal, ...]:
    return tuple(sorted(literals, key=lambda literal: literal.atom.display))


def _sorted_effects(effects: Iterable[Effect]) -> tuple[Effect, ...]:
    return tuple(
        sorted(
            effects,
            key=lambda effect: (
                tuple(literal.atom.display for literal in effect.condition),
                tuple(literal.atom.display for literal in effect.literals),
            ),
        )
    )


def _structure_sort_key(structure: ActionStructure) -> tuple[Any, ...]:
    """Canonical total order over full action-structure records.

    Sorting by name alone is not a total order once disjunctive
    preconditions expand into duplicate-named schemas; the precondition and
    effect displays break those ties identically on both backends.
    """
    return (
        structure.name,
        structure.arity,
        tuple(literal.atom.display for literal in structure.precondition),
        tuple(
            (
                tuple(literal.atom.display for literal in effect.condition),
                tuple(literal.atom.display for literal in effect.literals),
            )
            for effect in structure.effects
        ),
    )


def _pymimir_action_structures(reader: Any) -> tuple[ActionStructure, ...]:
    domain = reader._problem.get_domain()
    structures: list[ActionStructure] = []
    for action in domain.get_actions():
        parameters = tuple(
            parameter.get_name() for parameter in action.get_parameters()
        )
        rename = {name: f"?a{index}" for index, name in enumerate(parameters)}

        def lift_pymimir_literal(literal: Any) -> Literal:
            atom = literal.get_atom()
            return _canonical_literal(
                atom.get_predicate().get_name(),
                (term.get_name() for term in atom.get_terms()),
                bool(literal.get_polarity()),
                rename,
            )

        precondition = _sorted_literals(
            lift_pymimir_literal(literal)
            for literal in action.get_precondition().get_literals()
        )
        effects = _sorted_effects(
            Effect(
                _sorted_literals(
                    lift_pymimir_literal(literal)
                    for literal in conditional.get_condition().get_literals()
                ),
                _sorted_literals(
                    lift_pymimir_literal(literal)
                    for literal in conditional.get_effect().get_literals()
                ),
            )
            for conditional in action.get_conditional_effect()
        )
        structures.append(
            ActionStructure(
                action.get_name(),
                int(action.get_arity()),
                tuple(f"?a{index}" for index in range(len(parameters))),
                precondition,
                effects,
            )
        )
    return tuple(sorted(structures, key=_structure_sort_key))


def _pytyr_action_structures(reader: Any) -> tuple[ActionStructure, ...]:
    domain = reader._planning_task.get_task().get_domain()
    structures: list[ActionStructure] = []
    for action in domain.get_actions():
        count = len(list(action.get_variables()))
        rename = {f"V{index}": f"?a{index}" for index in range(count)}

        def lift_pytyr_literal(literal: Any) -> Literal:
            atom = literal.get_atom()
            return _canonical_literal(
                atom.get_predicate().get_name(),
                (str(term) for term in atom.get_terms()),
                bool(literal.get_polarity()),
                rename,
            )

        def lift_conjunctive(condition: Any) -> tuple[Literal, ...]:
            literals: list[Literal] = []
            for getter in (
                condition.get_static_literals,
                condition.get_fluent_literals,
                condition.get_derived_literals,
            ):
                literals.extend(lift_pytyr_literal(literal) for literal in getter())
            return _sorted_literals(literals)

        def structure_effects(action: Any) -> tuple[Effect, ...]:
            effects: list[Effect] = []
            for conditional in action.get_effects():
                condition = lift_conjunctive(conditional.get_condition())
                literals = _sorted_literals(
                    lift_pytyr_literal(literal)
                    for literal in conditional.get_effect().get_literals()
                )
                # pytyr materializes an empty ``(and)`` effect as a
                # fully-empty ConjunctiveEffect; pymimir emits no effect at
                # all for those, so drop them to keep the views identical.
                # Conditional effects with a real condition but empty body
                # are kept on both sides.
                if not condition and not literals:
                    continue
                effects.append(Effect(condition, literals))
            return _sorted_effects(effects)

        precondition = lift_conjunctive(action.get_condition())
        effects = structure_effects(action)
        structures.append(
            ActionStructure(
                str(action.get_name()),
                int(action.get_original_arity()),
                tuple(f"?a{index}" for index in range(count)),
                precondition,
                effects,
            )
        )
    return tuple(sorted(structures, key=_structure_sort_key))


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
        self._action_structures: tuple[ActionStructure, ...] | None = None
        self._state_facts_cache: dict[Any, tuple[Atom, ...]] = {}
        self._state_facts_cacheable: bool | None = None

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
    def has_action_structures(self) -> bool:
        """Whether this backend exposes lifted precondition/effect data."""

        return True

    def action_structures(self) -> tuple[ActionStructure, ...]:
        """Lifted action schemas with canonicalized preconditions/effects.

        Both supported backends expose the full structure; parameters are
        canonically renamed positionally (see :class:`ActionStructure`).
        Collections are sorted canonically so the two backends agree
        byte-for-byte on the same PDDL files — including domains with
        ``:disjunctive-preconditions``, where both backends flatten each
        disjunct into a duplicate-named schema and ordering is
        canonicalized over full records rather than names alone.
        """

        if self._action_structures is None:
            if self.backend == "pytyr":
                self._action_structures = _pytyr_action_structures(self._reader)
            else:
                self._action_structures = _pymimir_action_structures(self._reader)
        return self._action_structures

    @property
    def static_facts(self) -> tuple[Atom, ...]:
        """Static atoms that hold in every state of this problem."""

        return self._static_facts

    @property
    def problem_name(self) -> str:
        """The problem's declared name (best effort)."""

        return self._problem_name

    def state_facts(self, state: Any) -> tuple[Atom, ...]:
        """True fluent atoms followed by true derived atoms, stable order.

        Results are memoized per state object: backend states are immutable
        value objects whose fact tuples are pure functions of their content,
        so repeated encodes of the same state reuse the first conversion.
        Unhashable states simply skip the memo.
        """

        if self._state_facts_cacheable is not False:
            try:
                cached = self._state_facts_cache.get(state)
            except TypeError:
                self._state_facts_cacheable = False
            else:
                if cached is not None:
                    return cached
                facts = self._compute_state_facts(state)
                cache = self._state_facts_cache
                if len(cache) >= _STATE_FACTS_CACHE_LIMIT:
                    cache.clear()
                cache[state] = facts
                return facts
        return self._compute_state_facts(state)

    def _compute_state_facts(self, state: Any) -> tuple[Atom, ...]:
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
    "ActionStructure",
    "Atom",
    "Effect",
    "Literal",
    "PredicateInfo",
    "StateView",
]
