"""PyTyr input reader for the planner-neutral Python ILG algorithm."""

from __future__ import annotations

from collections.abc import Iterable
from typing import cast, Literal

from pytyr.formalism.planning import PlanningTask

from ._ilg_runtime import (
    ILGAction,
    ILGAtom,
    ILGInput,
    ILGLiteral,
    semantic_display,
)
from .pytyr import PyTyrSnapshotReader, SemanticPlanningTaskAdapter
from .pytyr_flat import (
    _is_action,
    _is_state,
    _lane_values,
    _sequence,
    _subgoal_values,
)
from .semantic import AtomKey, GroundActionKey, LiteralKey


class PyTyrILGRuntime:
    backend_name: Literal["pytyr"] = "pytyr"

    def __init__(self, planning_task: object) -> None:
        if not isinstance(planning_task, PlanningTask):
            raise TypeError(
                "a PyTyr ILGEncoder expects a PlanningTask, "
                f"got {type(planning_task)!r}"
            )
        self._reader = PyTyrSnapshotReader(planning_task)
        domain = self._reader.domain_snapshot()
        self._problem = self._reader.problem_snapshot()
        self.predicate_arities = {
            predicate.name: predicate.arity for predicate in domain.predicates
        }
        self.action_feature_dim = (
            max((action.arity for action in domain.actions), default=0) + 1
        )

    @staticmethod
    def _atom(value: AtomKey) -> ILGAtom:
        return ILGAtom(
            value.predicate.name,
            value.objects,
            semantic_display(value.predicate.name, value.objects),
        )

    @classmethod
    def _literal(cls, value: object) -> ILGLiteral:
        literal = SemanticPlanningTaskAdapter._literal_key(value)
        return ILGLiteral(cls._atom(literal.atom), literal.polarity)

    @staticmethod
    def _action(value: GroundActionKey) -> ILGAction:
        return ILGAction(
            value.action.name,
            value.objects,
            semantic_display(value.action.name, value.objects),
        )

    @staticmethod
    def _is_literal(value: object) -> bool:
        if isinstance(value, LiteralKey):
            return True
        try:
            SemanticPlanningTaskAdapter._literal_key(value)
        except (TypeError, ValueError):
            return False
        return True

    def make_input(
        self,
        state: object,
        *,
        goals: object = None,
        actions: object = None,
        subgoal_layers: object = None,
    ) -> ILGInput:
        snapshot = self._reader.state_snapshot(state)
        goal_values: Iterable[object] = (
            self._problem.goals if goals is None else cast(Iterable[object], goals)
        )
        action_values: Iterable[object] = (
            () if actions is None else cast(Iterable[object], actions)
        )
        layer_values: Iterable[Iterable[object]] = (
            ()
            if subgoal_layers is None
            else cast(Iterable[Iterable[object]], subgoal_layers)
        )
        return ILGInput(
            objects=self._problem.objects,
            facts=tuple(self._atom(atom) for atom in snapshot.atoms),
            goals=tuple(self._literal(value) for value in goal_values),
            actions=tuple(
                self._action(self._reader.action_key(value)) for value in action_values
            ),
            subgoal_layers=tuple(
                tuple(self._literal(value) for value in layer) for layer in layer_values
            ),
        )

    def make_batch_inputs(
        self,
        states: object,
        *,
        goals: object = None,
        actions: object = None,
        subgoal_layers: object = None,
    ) -> list[ILGInput]:
        state_values = (
            [states] if _is_state(states) else _sequence(states, field="states")
        )
        if not all(_is_state(state) for state in state_values):
            raise TypeError("a PyTyr ILGEncoder batch can contain only PyTyr states")
        count = len(state_values)
        goal_values = _lane_values(
            goals,
            state_count=count,
            field="goals",
            leaf=self._is_literal,
        )
        action_values = _lane_values(
            actions,
            state_count=count,
            field="actions",
            leaf=_is_action,
        )
        subgoal_values = _subgoal_values(subgoal_layers, state_count=count)
        return [
            self.make_input(
                state,
                goals=goal_values[index],
                actions=action_values[index],
                subgoal_layers=subgoal_values[index],
            )
            for index, state in enumerate(state_values)
        ]


__all__ = ["PyTyrILGRuntime"]
