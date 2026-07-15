"""PyTyr implementation of the semantic snapshot contract."""

from __future__ import annotations

import importlib
from collections.abc import Iterable
from typing import Any

from pytyr.formalism.planning import PlanningTask

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


def _predicate_key(predicate: Any, category: PredicateCategory) -> PredicateKey:
    return PredicateKey(category, str(predicate.get_name()), int(predicate.get_arity()))


def _atom_key(atom: Any, category: PredicateCategory) -> AtomKey:
    predicate = atom.get_predicate()
    return AtomKey(
        _predicate_key(predicate, category),
        tuple(str(value.get_name()) for value in atom.get_objects()),
    )


def _literal_key(literal: Any, category: PredicateCategory) -> LiteralKey:
    return LiteralKey(
        _atom_key(literal.get_atom(), category), bool(literal.get_polarity())
    )


class PyTyrSnapshotReader:
    """Read canonical snapshots from one PyTyr planning task."""

    backend_name = "pytyr"

    def __init__(self, planning_task: PlanningTask) -> None:
        if not isinstance(planning_task, PlanningTask):
            raise TypeError(
                "PyTyrSnapshotReader expects pytyr PlanningTask, "
                f"got {type(planning_task)!r}"
            )
        self._planning_task = planning_task

    @property
    def _task(self) -> Any:
        return self._planning_task.get_task()

    def domain_snapshot(self) -> DomainSnapshot:
        domain = self._task.get_domain()
        predicates = [
            *(
                _predicate_key(predicate, PredicateCategory.STATIC)
                for predicate in domain.get_static_predicates()
            ),
            *(
                _predicate_key(predicate, PredicateCategory.FLUENT)
                for predicate in domain.get_fluent_predicates()
            ),
            *(
                _predicate_key(predicate, PredicateCategory.DERIVED)
                for predicate in domain.get_derived_predicates()
            ),
        ]
        actions = (
            ActionSchemaKey(str(action.get_name()), int(action.get_original_arity()))
            for action in domain.get_actions()
        )
        return DomainSnapshot.canonical(
            name=domain.get_name(), predicates=predicates, actions=actions
        )

    def problem_snapshot(self) -> ProblemSnapshot:
        task = self._task
        goal = task.get_goal()
        goals: list[LiteralKey] = []
        goals.extend(
            _literal_key(literal, PredicateCategory.STATIC)
            for literal in goal.get_static_facts()
        )
        goals.extend(
            LiteralKey(_atom_key(fact.get_atom(), PredicateCategory.FLUENT), True)
            for fact in goal.get_positive_facts()
            if fact.has_value()
        )
        goals.extend(
            LiteralKey(_atom_key(fact.get_atom(), PredicateCategory.FLUENT), False)
            for fact in goal.get_negative_facts()
            if fact.has_value()
        )
        goals.extend(
            _literal_key(literal, PredicateCategory.DERIVED)
            for literal in goal.get_derived_facts()
        )
        domain = task.get_domain()
        return ProblemSnapshot.canonical(
            name=task.get_name(),
            domain_name=domain.get_name(),
            objects=(value.get_name() for value in task.get_objects()),
            static_atoms=(
                _atom_key(atom, PredicateCategory.STATIC)
                for atom in task.get_static_atoms()
            ),
            goals=goals,
        )

    def state_snapshot(self, state: object) -> StateSnapshot:
        required = ("static_atoms", "fluent_facts", "derived_atoms")
        if any(not hasattr(state, name) for name in required):
            raise TypeError(
                "pytyr state snapshot expects a lifted or ground State, "
                f"got {type(state)!r}"
            )
        state_value: Any = state
        # An FDR fact without an atom means none of that variable's propositional
        # atoms is true. Omitting it is therefore the canonical STRIPS view.
        fluent_atoms = (
            _atom_key(fact.get_atom(), PredicateCategory.FLUENT)
            for fact in state_value.fluent_facts()
            if fact.has_value()
        )
        return StateSnapshot.canonical(
            static_atoms=(
                _atom_key(atom, PredicateCategory.STATIC)
                for atom in state_value.static_atoms()
            ),
            fluent_atoms=fluent_atoms,
            derived_atoms=(
                _atom_key(atom, PredicateCategory.DERIVED)
                for atom in state_value.derived_atoms()
            ),
        )

    def action_key(self, action: object) -> GroundActionKey:
        required = ("get_action", "get_objects")
        if any(not hasattr(action, name) for name in required):
            raise TypeError(
                f"pytyr action key expects GroundAction, got {type(action)!r}"
            )
        action_value: Any = action
        schema = action_value.get_action()
        return GroundActionKey(
            ActionSchemaKey(str(schema.get_name()), int(schema.get_original_arity())),
            tuple(str(value.get_name()) for value in action_value.get_objects()),
        )


class SemanticFlatRelationEncoder:
    """Native PyTyr conversion backed by the planner-neutral flat engine."""

    def __init__(self, planning_task: PlanningTask, config: Any | None = None) -> None:
        from mifrost import _neutral_core

        if config is None:
            config = _neutral_core.FlatRelationEncoderConfig()
        try:
            native_module = importlib.import_module("mifrost._pytyr_adapter")
        except ImportError as error:
            raise ModuleNotFoundError(
                "The native PyTyr adapter is unavailable. Install "
                "mifrost[pytyr], or rebuild with MIFROST_BUILD_BACKENDS=pytyr "
                "(or both)."
            ) from error

        config_capsule = _neutral_core._flat_relation_config_capsule(config)
        self._native = native_module._NativeSemanticFlatRelationEncoder(
            planning_task, config_capsule
        )
        self._engine = _neutral_core._consume_semantic_flat_engine_capsule(
            self._native._make_engine_capsule()
        )

    @property
    def engine(self) -> Any:
        return self._engine

    def make_input(self, state: object, actions: Iterable[object] = ()) -> Any:
        from mifrost import _neutral_core

        capsule = self._native._make_input_capsule(state, list(actions))
        return _neutral_core._consume_semantic_flat_input_capsule(capsule)

    def encode(self, state: object, actions: Iterable[object] = ()) -> Any:
        return self._engine.encode(self.make_input(state, actions))
