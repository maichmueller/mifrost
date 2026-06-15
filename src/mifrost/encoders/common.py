from __future__ import annotations

import itertools
from collections.abc import Iterable as IterableABC
from collections.abc import Sequence as SequenceABC
from typing import Any, Callable, Iterable

import pymimir.advanced.formalism as af

from .types import (
    BatchParam,
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
    if actions is None:
        return []
    return [_advanced_action(action) for action in actions]


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


def _convert_batch_payload(
    value: Any,
    *,
    is_leaf: Callable[[object], bool],
    convert_leaf: Callable[[Any], Any],
) -> Any:
    if value is None:
        return None
    if isinstance(value, BatchParam):
        if value.kind == "none":
            return BatchParam.none()
        if value.kind == "shared":
            return BatchParam.shared(
                _convert_batch_payload(
                    value.value,
                    is_leaf=is_leaf,
                    convert_leaf=convert_leaf,
                )
            )
        if value.kind == "separate":
            if not isinstance(value.value, SequenceABC) or isinstance(
                value.value, (str, bytes, bytearray)
            ):
                raise TypeError("BatchParam(separate) value must be a sequence")
            return BatchParam.separate(
                (
                    _convert_batch_payload(
                        entry,
                        is_leaf=is_leaf,
                        convert_leaf=convert_leaf,
                    )
                    if entry is not None
                    else None
                )
                for entry in value.value
            )
        raise ValueError("BatchParam.kind must be 'shared', 'separate', or 'none'")
    if is_leaf(value):
        return convert_leaf(value)
    if isinstance(value, (str, bytes, bytearray)):
        return value
    if isinstance(value, tuple):
        return tuple(
            _convert_batch_payload(
                item,
                is_leaf=is_leaf,
                convert_leaf=convert_leaf,
            )
            for item in value
        )
    if isinstance(value, list):
        return [
            _convert_batch_payload(
                item,
                is_leaf=is_leaf,
                convert_leaf=convert_leaf,
            )
            for item in value
        ]
    if isinstance(value, SequenceABC):
        return [
            _convert_batch_payload(
                item,
                is_leaf=is_leaf,
                convert_leaf=convert_leaf,
            )
            for item in value
        ]
    if isinstance(value, IterableABC):
        return [
            _convert_batch_payload(
                item,
                is_leaf=is_leaf,
                convert_leaf=convert_leaf,
            )
            for item in value
        ]
    return value


def _encoding_dict_to_pyg(
    encoding: Any,
    *,
    as_batch: bool | None = None,
    include_metadata: bool = True,
) -> Any:
    """Compatibility wrapper for the conversion boundary module."""
    from .conversion import _encoding_dict_to_pyg as convert

    return convert(encoding, as_batch=as_batch, include_metadata=include_metadata)


def to_pyg(
    encoding: Any,
    *,
    as_batch: bool | None = None,
    include_metadata: bool = True,
) -> Any:
    """Compatibility wrapper for the conversion boundary module."""
    from .conversion import to_pyg as convert

    return convert(encoding, as_batch=as_batch, include_metadata=include_metadata)


def to_tensor_payload(encoding: Any) -> Any:
    """Compatibility wrapper for the conversion boundary module."""
    from .conversion import to_tensor_payload as convert

    return convert(encoding)


def encoding_to_tensors(encoding: Any) -> Any:
    """Compatibility wrapper for the conversion boundary module."""
    from .conversion import encoding_to_tensors as convert

    return convert(encoding)
