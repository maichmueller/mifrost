"""Backend-neutral construction of semantic flat encoder inputs.

This module is the Phase 1 executable adapter. It intentionally favors a clear
semantic contract over hot-path performance; backend-specific native adapters
will construct the same compact records without Python object materialization.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from .. import _neutral_core as _core
from .semantic import (
    AtomKey,
    GroundActionKey,
    LiteralKey,
    PredicateCategory,
    SnapshotReader,
)


def _core_category(category: PredicateCategory) -> Any:
    values = _core.SemanticPredicateCategory
    if category is PredicateCategory.STATIC:
        return getattr(values, "static")
    if category is PredicateCategory.FLUENT:
        return values.fluent
    if category is PredicateCategory.DERIVED:
        return values.derived
    raise ValueError(f"unsupported predicate category: {category!r}")


class FlatSemanticAdapter:
    """Compile one problem reader into the existing semantic flat engine."""

    def __init__(self, reader: SnapshotReader, config: Any | None = None) -> None:
        if not isinstance(reader, SnapshotReader):
            raise TypeError(f"expected SnapshotReader, got {type(reader)!r}")
        self.reader = reader
        self.domain = reader.domain_snapshot()
        self.problem = reader.problem_snapshot()
        self._predicate_indices = {
            predicate: index for index, predicate in enumerate(self.domain.predicates)
        }
        self._action_indices = {
            action: index for index, action in enumerate(self.domain.actions)
        }
        self._object_indices = {
            name: index for index, name in enumerate(self.problem.objects)
        }
        predicates = [
            _core.SemanticPredicateSpec(
                _core_category(predicate.category),
                predicate.name,
                predicate.arity,
            )
            for predicate in self.domain.predicates
        ]
        actions = [
            _core.SemanticActionSpec(action.name, action.arity)
            for action in self.domain.actions
        ]
        if config is None:
            self.engine = _core.SemanticFlatRelationEncoderEngine(predicates, actions)
        else:
            self.engine = _core.SemanticFlatRelationEncoderEngine(
                predicates, actions, config
            )

    def _atom(self, atom: AtomKey) -> tuple[int, list[int]]:
        try:
            predicate_index = self._predicate_indices[atom.predicate]
        except KeyError as error:
            raise ValueError(
                f"atom predicate is outside the adapter domain: {atom.predicate!r}"
            ) from error
        try:
            objects = [self._object_indices[name] for name in atom.objects]
        except KeyError as error:
            raise ValueError(
                f"atom object is outside the adapter problem: {error.args[0]!r}"
            ) from error
        return predicate_index, objects

    def _literal(self, literal: LiteralKey) -> tuple[int, list[int], bool]:
        predicate, objects = self._atom(literal.atom)
        return predicate, objects, literal.polarity

    def _action(self, action: GroundActionKey) -> tuple[int, list[int]]:
        try:
            action_index = self._action_indices[action.action]
        except KeyError as error:
            raise ValueError(
                f"ground action is outside the adapter domain: {action.action!r}"
            ) from error
        try:
            objects = [self._object_indices[name] for name in action.objects]
        except KeyError as error:
            raise ValueError(
                f"action object is outside the adapter problem: {error.args[0]!r}"
            ) from error
        return action_index, objects

    def make_input(
        self,
        state: object,
        *,
        goals: Iterable[LiteralKey] | None = None,
        actions: Iterable[object] = (),
        subgoal_layers: Iterable[Iterable[LiteralKey]] = (),
        history: Iterable[tuple[int, Iterable[LiteralKey]]] = (),
        history_max_steps: int | None = None,
    ) -> Any:
        """Convert backend objects and semantic goal lanes into compact input."""

        snapshot = self.reader.state_snapshot(state)
        goal_values = self.problem.goals if goals is None else tuple(goals)
        return _core.SemanticFlatRelationInput.from_compact(
            objects=list(self.problem.objects),
            state_facts=[self._atom(atom) for atom in snapshot.atoms],
            goals=[self._literal(literal) for literal in goal_values],
            actions=[
                self._action(self.reader.action_key(action)) for action in actions
            ],
            subgoal_layers=[
                [self._literal(literal) for literal in layer]
                for layer in subgoal_layers
            ],
            history=[
                (int(delta), [self._literal(literal) for literal in literals])
                for delta, literals in history
            ],
            history_max_steps=history_max_steps,
        )

    def encode(
        self,
        state: object,
        *,
        goals: Iterable[LiteralKey] | None = None,
        actions: Iterable[object] = (),
        subgoal_layers: Iterable[Iterable[LiteralKey]] = (),
        history: Iterable[tuple[int, Iterable[LiteralKey]]] = (),
        history_max_steps: int | None = None,
    ) -> Any:
        """Encode one graph through the backend-neutral semantic engine."""

        return self.engine.encode(
            self.make_input(
                state,
                goals=goals,
                actions=actions,
                subgoal_layers=subgoal_layers,
                history=history,
                history_max_steps=history_max_steps,
            )
        )
