"""Pymimir input reader for the planner-neutral Python ILG algorithm."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Literal, cast

from .. import _core
from ..encoders._batch_contract import prepare_core_batch_inputs
from .pymimir_accessors import (
    action_objects,
    atom_objects,
    literal_atom,
    literal_polarity,
    object_name,
    predicate,
    predicate_arity,
    predicate_name,
)
from .pymimir_types import (
    ATOM_TYPES,
    WRAPPER_STATE_TYPES,
    is_action_input,
    is_goal_literal_input,
    is_state_input,
)
from ._ilg_runtime import ILGAction, ILGAtom, ILGInput, ILGLiteral
from .pymimir_common import _advanced_action, _advanced_literal, _advanced_state


def _gather_objects(items: Iterable[Any]) -> list[Any]:
    objects: list[Any] = []
    seen: set[str] = set()
    for item in items:
        if is_action_input(item):
            candidates = list(action_objects(_advanced_action(item)))
        elif is_goal_literal_input(item):
            candidates = list(atom_objects(literal_atom(_advanced_literal(item))))
        elif isinstance(item, ATOM_TYPES):
            candidates = list(atom_objects(item))
        else:
            raise TypeError(f"Unsupported object source type: {type(item)!r}")
        for candidate in candidates:
            name = object_name(candidate)
            if name not in seen:
                seen.add(name)
                objects.append(candidate)
    return objects


class PymimirILGRuntime:
    backend_name: Literal["pymimir"] = "pymimir"

    def __init__(self, domain: object) -> None:
        if not hasattr(domain, "get_predicates"):
            raise TypeError(f"Unsupported domain type: {type(domain)!r}")
        predicates = tuple(domain.get_predicates())
        self.predicate_arities = {
            predicate_name(value): predicate_arity(value) for value in predicates
        }
        actions = tuple(domain.get_actions()) if hasattr(domain, "get_actions") else ()
        self.action_feature_dim = (
            max((int(action.get_arity()) for action in actions), default=0) + 1
        )

    @staticmethod
    def _atom(value: object) -> ILGAtom:
        atom_value = cast(Any, value)
        arguments = tuple(object_name(item) for item in atom_objects(atom_value))
        return ILGAtom(
            predicate_name(predicate(atom_value)),
            arguments,
            str(atom_value),
        )

    @classmethod
    def _literal(cls, value: object) -> ILGLiteral:
        literal = _advanced_literal(value)
        return ILGLiteral(
            cls._atom(literal_atom(literal)),
            literal_polarity(literal),
        )

    @staticmethod
    def _action(value: object) -> ILGAction:
        action = _advanced_action(value)
        arguments = tuple(object_name(item) for item in action_objects(action))
        action_schema = action.get_action()
        return ILGAction(
            str(action_schema.get_name()),
            arguments,
            str(action),
        )

    def make_input(
        self,
        state: object,
        *,
        goals: object = None,
        actions: object = None,
        subgoal_layers: object = None,
    ) -> ILGInput:
        if isinstance(state, WRAPPER_STATE_TYPES):
            problem = state.get_problem()
            facts = list(state.get_atoms())
            if goals is None:
                goals = list(problem.get_goal_condition().get_literals())
            native_objects = list(problem.get_objects()) + list(
                problem.get_domain().get_constants()
            )
        elif is_state_input(state):
            advanced_state = _advanced_state(state)
            facts = list(advanced_state.get_fluent_atoms()) + list(
                advanced_state.get_derived_atoms()
            )
            if facts and all(isinstance(fact, int) for fact in facts):
                raise TypeError(
                    "ILGEncoder does not support advanced states that expose atom "
                    "indices only; pass wrapper states or an explicit iterable of atoms"
                )
            if goals is None:
                goals = []
            native_objects = _gather_objects(facts)
        else:
            try:
                facts = list(cast(Iterable[Any], state))
            except TypeError as error:
                raise TypeError(
                    f"Unsupported ILG state type: {type(state)!r}"
                ) from error
            if goals is None:
                goals = []
            native_objects = _gather_objects(facts)

        goal_values = [] if goals is None else list(cast(Iterable[Any], goals))
        action_values = [] if actions is None else list(cast(Iterable[Any], actions))
        layer_values = (
            []
            if subgoal_layers is None
            else [
                list(layer) for layer in cast(Iterable[Iterable[Any]], subgoal_layers)
            ]
        )
        if not native_objects:
            native_objects = _gather_objects([*facts, *goal_values, *action_values])
        return ILGInput(
            objects=tuple(object_name(value) for value in native_objects),
            facts=tuple(self._atom(value) for value in facts),
            goals=tuple(self._literal(value) for value in goal_values),
            actions=tuple(self._action(value) for value in action_values),
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
        state_list = (
            [states] if is_state_input(states) else list(cast(Iterable[Any], states))
        )
        inputs = prepare_core_batch_inputs(
            state_list,
            goals=goals,
            actions=actions,
            subgoal_layers=subgoal_layers,
        )
        _, goals_per_state, actions_per_state, subgoals_per_state = cast(
            Any, _core
        )._parse_ilg_batch_inputs(
            inputs.states,
            goals=inputs.goals,
            actions=inputs.actions,
            subgoal_layers=inputs.subgoal_layers,
        )
        return [
            self.make_input(
                state,
                goals=goals_per_state[index],
                actions=actions_per_state[index],
                subgoal_layers=subgoals_per_state[index],
            )
            for index, state in enumerate(state_list)
        ]


__all__ = ["PymimirILGRuntime"]
