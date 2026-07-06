from __future__ import annotations

import itertools
from typing import Iterable

import pymimir.advanced.formalism as af

from .types import (
    DomainInput,
    GoalLiteralInput,
    GroundActionInput,
    HistorySubgoalInput,
    StateInput,
    to_advanced_action,
    to_advanced_domain,
    to_advanced_literal,
    to_advanced_state,
)
from ._action_contract import parse_flat_actions

from .._core import GoalInputs


# The C++ boundary is strict/typed; wrappers are unwrapped explicitly here.
def _advanced_domain(domain: DomainInput):
    """Return the advanced pymimir domain object when wrapped."""
    return to_advanced_domain(domain)


def _advanced_state(state: StateInput):
    """Return the advanced pymimir state object when wrapped."""
    return to_advanced_state(state)


def _advanced_literal(literal: GoalLiteralInput):
    """Return the advanced pymimir literal object when wrapped."""
    return to_advanced_literal(literal)


def _advanced_action(action: GroundActionInput):
    """Return the advanced pymimir action object when wrapped."""
    return to_advanced_action(action)


def _split_goals(
    goals: Iterable[GoalLiteralInput],
    subgoal_layers: Iterable[Iterable[GoalLiteralInput]] | None,
) -> GoalInputs:
    """
    Build ``GoalInputs`` from goals and optional layered subgoals.

    Returns ``(goal_inputs, layer_count)`` where ``layer_count`` includes the
    primary goal layer plus optional subgoal layers.
    """
    # Keep Python-side unwrapping/normalization (registered conversion logic) but
    # delegate type splitting and level bookkeeping to the C++ GoalInputs container.
    goals = list(goals)

    if subgoal_layers is None:
        return GoalInputs(map(_advanced_literal, goals), 0)

    inputs = GoalInputs([])
    for depth, layer in enumerate(itertools.chain([goals], subgoal_layers)):
        if not layer:
            continue
        inputs.extend(map(_advanced_literal, layer), int(depth))

    return inputs


def _prepare_actions(
    actions: Iterable[GroundActionInput] | None,
) -> list[af.GroundAction]:
    """Convert flat action wrappers to advanced ``GroundAction`` objects.

    This path intentionally does not flatten nested action payloads. Callers
    must provide flat action sequences, or use ``HorizonEncoder`` for
    IW/lookahead DAG workflows.
    """
    return (
        []
        if actions is None
        else [_advanced_action(action) for action in parse_flat_actions(actions)]
    )


def _prepare_history_subgoals(
    history_subgoals: HistorySubgoalInput | None,
) -> list[
    tuple[
        int,
        list[af.StaticGroundLiteral | af.FluentGroundLiteral | af.DerivedGroundLiteral],
    ]
]:
    """Normalize history subgoals into advanced literal lists."""
    if history_subgoals is None:
        return []
    out: list[
        tuple[
            int,
            list[
                af.StaticGroundLiteral
                | af.FluentGroundLiteral
                | af.DerivedGroundLiteral
            ],
        ]
    ] = []
    for dt, literals in history_subgoals:
        adv_literals = [_advanced_literal(literal) for literal in literals]
        out.append((int(dt), adv_literals))
    return out
