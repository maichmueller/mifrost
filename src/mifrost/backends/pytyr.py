"""PyTyr implementation of the semantic snapshot contract."""

from __future__ import annotations

import importlib
from collections.abc import Iterable
from typing import Any

from pytyr.formalism.planning import (
    DerivedGroundLiteral,
    FluentFDRFact,
    FluentGroundLiteral,
    PlanningTask,
    StaticGroundLiteral,
)

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


class SemanticPlanningTaskAdapter:
    """Native PyTyr conversion to compact backend-neutral semantic inputs."""

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
        domain = PyTyrSnapshotReader(planning_task).domain_snapshot()
        self._predicate_indices = {
            predicate: index for index, predicate in enumerate(domain.predicates)
        }
        object_names = sorted(
            str(value.get_name()) for value in planning_task.get_task().get_objects()
        )
        self._object_indices = {name: index for index, name in enumerate(object_names)}

    def make_color_engine(self, config: Any) -> Any:
        """Build a neutral Color engine from this task adapter's cached schema."""
        from mifrost import _neutral_core

        config_capsule = _neutral_core._semantic_color_config_capsule(config)
        engine_capsule = self._native._make_color_engine_capsule(config_capsule)
        return _neutral_core._consume_semantic_color_engine_capsule(engine_capsule)

    def make_hgraph_engine(self, config: Any) -> Any:
        """Build a neutral HGraph engine from this task adapter's cached schema."""
        from mifrost import _neutral_core

        config_capsule = _neutral_core._semantic_hgraph_config_capsule(config)
        engine_capsule = self._native._make_hgraph_engine_capsule(config_capsule)
        return _neutral_core._consume_semantic_hgraph_engine_capsule(engine_capsule)

    def make_successor_hgraph_engine(self, config: Any) -> Any:
        """Build a neutral successor HGraph engine from the cached task schema."""
        from mifrost import _neutral_core

        config_capsule = _neutral_core._semantic_successor_hgraph_config_capsule(config)
        engine_capsule = self._native._make_successor_hgraph_engine_capsule(
            config_capsule
        )
        return _neutral_core._consume_semantic_successor_hgraph_engine_capsule(
            engine_capsule
        )

    def make_horizon_hgraph_engine(self, config: Any) -> Any:
        """Build a neutral Horizon engine from the cached task schema."""
        from mifrost import _neutral_core

        config_capsule = _neutral_core._semantic_horizon_hgraph_config_capsule(config)
        engine_capsule = self._native._make_horizon_hgraph_engine_capsule(
            config_capsule
        )
        return _neutral_core._consume_semantic_horizon_hgraph_engine_capsule(
            engine_capsule
        )

    @staticmethod
    def _literal_key(value: object) -> LiteralKey:
        if isinstance(value, LiteralKey):
            return value
        if isinstance(value, StaticGroundLiteral):
            return _literal_key(value, PredicateCategory.STATIC)
        if isinstance(value, FluentGroundLiteral):
            return _literal_key(value, PredicateCategory.FLUENT)
        if isinstance(value, DerivedGroundLiteral):
            return _literal_key(value, PredicateCategory.DERIVED)
        if (
            isinstance(value, tuple)
            and len(value) == 2
            and isinstance(value[0], FluentFDRFact)
            and isinstance(value[1], bool)
        ):
            fact, polarity = value
            if not fact.has_value():
                raise ValueError("a PyTyr FDR no-value fact cannot form a literal")
            return LiteralKey(
                _atom_key(fact.get_atom(), PredicateCategory.FLUENT), polarity
            )
        if isinstance(value, FluentFDRFact):
            raise TypeError(
                "PyTyr FluentFDRFact goal inputs require an explicit polarity: "
                "pass (fact, True) or (fact, False)"
            )
        raise TypeError(
            "PyTyr flat literal inputs must be a ground literal, LiteralKey, or "
            f"(FluentFDRFact, polarity), got {type(value)!r}"
        )

    def _compact_literal(self, value: object) -> tuple[int, list[int], bool]:
        literal = self._literal_key(value)
        try:
            predicate = self._predicate_indices[literal.atom.predicate]
        except KeyError as error:
            raise ValueError(
                "literal predicate is outside the PyTyr adapter task: "
                f"{literal.atom.predicate!r}"
            ) from error
        try:
            arguments = [self._object_indices[name] for name in literal.atom.objects]
        except KeyError as error:
            raise ValueError(
                f"literal object is outside the PyTyr adapter task: {error.args[0]!r}"
            ) from error
        return predicate, arguments, literal.polarity

    def make_input(
        self,
        state: object,
        actions: Iterable[object] = (),
        *,
        goals: Iterable[object] | None = None,
        subgoal_layers: Iterable[Iterable[object]] = (),
        history: Iterable[tuple[int, Iterable[object]]] = (),
        history_max_steps: int | None = None,
    ) -> Any:
        from mifrost import _neutral_core

        compact_goals = (
            None if goals is None else [self._compact_literal(value) for value in goals]
        )
        compact_subgoals = [
            [self._compact_literal(value) for value in layer]
            for layer in subgoal_layers
        ]
        compact_history = [
            (int(delta), [self._compact_literal(value) for value in literals])
            for delta, literals in history
        ]
        capsule = self._native._make_input_capsule(
            state,
            list(actions),
            compact_goals,
            compact_subgoals,
            compact_history,
            history_max_steps,
        )
        return _neutral_core._consume_semantic_flat_input_capsule(capsule)

    def make_inputs(
        self,
        states: Iterable[object],
        actions: Iterable[Iterable[object]] = (),
        *,
        goals: Iterable[Iterable[object] | None] = (),
        subgoal_layers: Iterable[Iterable[Iterable[object]]] = (),
        history: Iterable[Iterable[tuple[int, Iterable[object]]]] = (),
        history_max_steps: int | None = None,
    ) -> list[Any]:
        """Convert a homogeneous state batch through one owned capsule."""
        from mifrost import _neutral_core

        compact_goals = [
            None
            if values is None
            else [self._compact_literal(value) for value in values]
            for values in goals
        ]
        compact_subgoals = [
            [[self._compact_literal(value) for value in layer] for layer in layers]
            for layers in subgoal_layers
        ]
        compact_history = [
            [
                (int(delta), [self._compact_literal(value) for value in literals])
                for delta, literals in entries
            ]
            for entries in history
        ]
        capsule = self._native._make_inputs_capsule(
            list(states),
            [list(values) for values in actions],
            compact_goals,
            compact_subgoals,
            compact_history,
            history_max_steps,
        )
        return list(_neutral_core._consume_semantic_flat_inputs_capsule(capsule))


class SemanticFlatRelationEncoder(SemanticPlanningTaskAdapter):
    """Native PyTyr conversion backed by the planner-neutral flat engine."""

    def __init__(self, planning_task: PlanningTask, config: Any | None = None) -> None:
        from mifrost import _neutral_core

        super().__init__(planning_task, config)
        self._engine = _neutral_core._consume_semantic_flat_engine_capsule(
            self._native._make_engine_capsule()
        )

    @property
    def engine(self) -> Any:
        return self._engine

    def encode(
        self,
        state: object,
        actions: Iterable[object] = (),
        *,
        goals: Iterable[object] | None = None,
        subgoal_layers: Iterable[Iterable[object]] = (),
        history: Iterable[tuple[int, Iterable[object]]] = (),
        history_max_steps: int | None = None,
    ) -> Any:
        return self._engine.encode(
            self.make_input(
                state,
                actions,
                goals=goals,
                subgoal_layers=subgoal_layers,
                history=history,
                history_max_steps=history_max_steps,
            )
        )
