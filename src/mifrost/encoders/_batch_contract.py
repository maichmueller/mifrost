from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, TypeAlias, TypeVar, cast

from .. import _core
from .._core import TransitionDAG
from .types import (
    GoalLiteralInput,
    GroundActionInput,
    StateInput,
)

_PayloadT = TypeVar("_PayloadT")


@dataclass(frozen=True)
class ParsedBatchPlan(Generic[_PayloadT]):
    """Compact shared/per-state batch plan with indexed access."""

    state_count: int
    shared: _PayloadT | None = None
    per_state: list[_PayloadT | None] | None = None

    def __len__(self) -> int:
        return self.state_count

    def __getitem__(self, index: int) -> _PayloadT | None:
        if index < 0 or index >= self.state_count:
            raise IndexError("batch index out of range")
        if self.per_state is not None:
            return self.per_state[index]
        return self.shared


ParsedGoalsBatch: TypeAlias = ParsedBatchPlan[list[GoalLiteralInput]]
ParsedActionsBatch: TypeAlias = ParsedBatchPlan[list[GroundActionInput]]
ParsedSubgoalLayersBatch: TypeAlias = ParsedBatchPlan[list[list[GoalLiteralInput]]]
ParsedHistorySubgoalsBatch: TypeAlias = ParsedBatchPlan[
    list[tuple[int, list[GoalLiteralInput]]]
]


def _plan_from_core_tuple(
    payload: tuple[bool, object],
    *,
    state_count: int,
) -> ParsedBatchPlan:
    is_per_state, value = payload
    if bool(is_per_state):
        return ParsedBatchPlan(
            state_count=state_count,
            per_state=cast(list[object | None], list(value)),
        )
    return ParsedBatchPlan(state_count=state_count, shared=cast(object | None, value))


def parse_states_batch(states) -> list[StateInput]:
    return cast(list[StateInput], list(_core._parse_states_batch(states)))


def parse_goals_batch_param(
    goals,
    *,
    state_count: int,
) -> ParsedGoalsBatch:
    return cast(
        ParsedGoalsBatch,
        _plan_from_core_tuple(
            cast(
                tuple[bool, object], _core._parse_goals_batch_param(goals, state_count)
            ),
            state_count=state_count,
        ),
    )


def parse_actions_batch_param(
    actions,
    *,
    state_count: int,
) -> ParsedActionsBatch:
    return cast(
        ParsedActionsBatch,
        _plan_from_core_tuple(
            cast(
                tuple[bool, object],
                _core._parse_actions_batch_param(actions, state_count),
            ),
            state_count=state_count,
        ),
    )


def parse_subgoal_layers_batch_param(
    subgoal_layers,
    *,
    state_count: int,
) -> ParsedSubgoalLayersBatch:
    return cast(
        ParsedSubgoalLayersBatch,
        _plan_from_core_tuple(
            cast(
                tuple[bool, object],
                _core._parse_subgoal_layers_batch_param(subgoal_layers, state_count),
            ),
            state_count=state_count,
        ),
    )


def parse_history_subgoals_batch_param(
    history_subgoals,
    *,
    state_count: int,
) -> ParsedHistorySubgoalsBatch:
    return cast(
        ParsedHistorySubgoalsBatch,
        _plan_from_core_tuple(
            cast(
                tuple[bool, object],
                _core._parse_history_subgoals_batch_param(
                    history_subgoals, state_count
                ),
            ),
            state_count=state_count,
        ),
    )


def parse_successors_batch_param(
    successors,
    *,
    state_count: int,
) -> list[StateInput]:
    return cast(
        list[StateInput],
        list(_core._parse_successors_batch_param(successors, state_count)),
    )


def parse_dags_batch_param(
    dags,
    *,
    state_count: int,
) -> list[TransitionDAG | None]:
    return cast(
        list[TransitionDAG | None],
        list(_core._parse_dags_batch_param(dags, state_count)),
    )


def reject_unsupported_batch_field(
    encoder_name: str,
    field_name: str,
    value: object,
) -> None:
    if value is not None:
        raise TypeError(
            f"{encoder_name} does not accept '{field_name}' in encode_batch"
        )
