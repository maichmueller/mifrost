from __future__ import annotations

from collections.abc import Iterable as IterableABC
from collections.abc import Sequence as SequenceABC
from dataclasses import dataclass
from typing import Any, Callable, Generic, Protocol, TypeAlias, TypeVar, cast

from .. import _core
from .._core import TransitionDAG
from .types import (
    BatchParam,
    GoalLiteralInput,
    GroundActionInput,
    StateInput,
    is_action_input,
    is_goal_literal_input,
    is_state_input,
    to_advanced_action,
    to_advanced_literal,
    to_advanced_state,
)

_PayloadT = TypeVar("_PayloadT")


class _CoreBatchContract(Protocol):
    """Typed port for private native batch parsers omitted from generated stubs."""

    def _parse_states_batch(self, states: Any, /) -> Any: ...

    def _parse_goals_batch_param(self, goals: Any, state_count: int, /) -> Any: ...

    def _parse_actions_batch_param(self, actions: Any, state_count: int, /) -> Any: ...

    def _parse_subgoal_layers_batch_param(
        self, subgoal_layers: Any, state_count: int, /
    ) -> Any: ...

    def _parse_history_subgoals_batch_param(
        self, history_subgoals: Any, state_count: int, /
    ) -> Any: ...

    def _parse_successors_batch_param(
        self, successors: Any, state_count: int, /
    ) -> Any: ...

    def _parse_dags_batch_param(self, dags: Any, state_count: int, /) -> Any: ...


_CORE_BATCH = cast(_CoreBatchContract, _core)


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


@dataclass(frozen=True, slots=True)
class CoreBatchInputs:
    """Batch payloads adapted to the objects accepted by the native core.

    The original shared/per-state container shape, including ``BatchParam``,
    is preserved. Only domain leaves are adapted. Keeping that policy here
    prevents individual encoder facades from drifting as new input adapters
    are added.
    """

    states: Any
    goals: Any
    actions: Any
    subgoal_layers: Any
    history_subgoals: Any
    successors: Any


def _plan_from_core_tuple(
    payload: tuple[bool, object],
    *,
    state_count: int,
) -> ParsedBatchPlan:
    is_per_state, value = payload
    if bool(is_per_state):
        if not isinstance(value, IterableABC):
            raise TypeError("per-state batch payload must be iterable")
        return ParsedBatchPlan(
            state_count=state_count,
            per_state=cast(list[object | None], list(value)),
        )
    return ParsedBatchPlan(state_count=state_count, shared=cast(object | None, value))


def parse_states_batch(states) -> list[StateInput]:
    return cast(list[StateInput], list(_CORE_BATCH._parse_states_batch(states)))


def parse_goals_batch_param(
    goals,
    *,
    state_count: int,
) -> ParsedGoalsBatch:
    return cast(
        ParsedGoalsBatch,
        _plan_from_core_tuple(
            cast(
                tuple[bool, object],
                _CORE_BATCH._parse_goals_batch_param(goals, state_count),
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
                _CORE_BATCH._parse_actions_batch_param(actions, state_count),
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
                _CORE_BATCH._parse_subgoal_layers_batch_param(
                    subgoal_layers, state_count
                ),
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
                _CORE_BATCH._parse_history_subgoals_batch_param(
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
        list(_CORE_BATCH._parse_successors_batch_param(successors, state_count)),
    )


def parse_dags_batch_param(
    dags,
    *,
    state_count: int,
) -> list[TransitionDAG | None]:
    return cast(
        list[TransitionDAG | None],
        list(_CORE_BATCH._parse_dags_batch_param(dags, state_count)),
    )


def convert_batch_payload(
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
                convert_batch_payload(
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
                    convert_batch_payload(
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
            convert_batch_payload(
                item,
                is_leaf=is_leaf,
                convert_leaf=convert_leaf,
            )
            for item in value
        )
    if isinstance(value, list):
        return [
            convert_batch_payload(
                item,
                is_leaf=is_leaf,
                convert_leaf=convert_leaf,
            )
            for item in value
        ]
    if isinstance(value, SequenceABC):
        return [
            convert_batch_payload(
                item,
                is_leaf=is_leaf,
                convert_leaf=convert_leaf,
            )
            for item in value
        ]
    if isinstance(value, IterableABC):
        return [
            convert_batch_payload(
                item,
                is_leaf=is_leaf,
                convert_leaf=convert_leaf,
            )
            for item in value
        ]
    return value


def prepare_core_batch_inputs(
    states: Any,
    *,
    goals: Any = None,
    actions: Any = None,
    subgoal_layers: Any = None,
    history_subgoals: Any = None,
    successors: Any = None,
) -> CoreBatchInputs:
    """Adapt all standard encoder batch lanes at one native-core boundary.

    Encoder facades should call this once, then pass the returned fields to
    their native engine or parser. This is intentionally the only place that
    maps the standard state, literal, and action lanes to their advanced
    pymimir representations.
    """

    return CoreBatchInputs(
        states=convert_batch_payload(
            states,
            is_leaf=is_state_input,
            convert_leaf=to_advanced_state,
        ),
        goals=convert_batch_payload(
            goals,
            is_leaf=is_goal_literal_input,
            convert_leaf=to_advanced_literal,
        ),
        actions=convert_batch_payload(
            actions,
            is_leaf=is_action_input,
            convert_leaf=to_advanced_action,
        ),
        subgoal_layers=convert_batch_payload(
            subgoal_layers,
            is_leaf=is_goal_literal_input,
            convert_leaf=to_advanced_literal,
        ),
        history_subgoals=convert_batch_payload(
            history_subgoals,
            is_leaf=is_goal_literal_input,
            convert_leaf=to_advanced_literal,
        ),
        successors=convert_batch_payload(
            successors,
            is_leaf=is_state_input,
            convert_leaf=to_advanced_state,
        ),
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
