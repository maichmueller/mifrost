from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Iterable, Sequence
from numbers import Integral
from typing import Callable, Generic, TypeAlias, TypeVar, cast

from .._core import TransitionDAG
from ._action_contract import parse_actions_batch_plan
from .base import StateBatchInput
from .types import (
    GoalLiteralInput,
    GroundActionInput,
    HistorySubgoalInput,
    StateInput,
    is_goal_literal_input,
    is_state_input,
)


_STR_BYTES = (str, bytes, bytearray)

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


def _is_sequence_like_but_not_str_bytes(value: object) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, _STR_BYTES)


def _is_iterable_but_not_str_bytes(value: object) -> bool:
    return isinstance(value, Iterable) and not isinstance(value, _STR_BYTES)


def _goals_type_error(
    field: str,
    *,
    entry_idx: int | None = None,
    literal_idx: int | None = None,
    got: object | None = None,
) -> TypeError:
    location = ""
    if entry_idx is not None:
        location = f" entry at index {entry_idx}"
    if literal_idx is not None:
        location += f" literal at position {literal_idx}"
    got_text = f"{type(got)!r}" if got is not None else "invalid value"
    return TypeError(f"{field}{location} has invalid goal literal type: {got_text}")


def _parse_goal_literals_iterable(
    value: object,
    *,
    field: str,
    entry_idx: int | None = None,
) -> list[GoalLiteralInput]:
    if not _is_iterable_but_not_str_bytes(value):
        location = f" entry at index {entry_idx}" if entry_idx is not None else ""
        raise TypeError(
            f"{field}{location} must be an iterable of goal literals or None"
        )
    out = list(value)
    for literal_idx, literal in enumerate(out):
        if not is_goal_literal_input(literal):
            raise _goals_type_error(
                field,
                entry_idx=entry_idx,
                literal_idx=literal_idx,
                got=literal,
            )
    return cast(list[GoalLiteralInput], out)


def _parse_subgoal_layers_payload(
    value: object,
    *,
    field: str,
    entry_idx: int | None = None,
) -> list[list[GoalLiteralInput]]:
    if not _is_iterable_but_not_str_bytes(value):
        location = f" entry at index {entry_idx}" if entry_idx is not None else ""
        raise TypeError(
            f"{field}{location} must be an iterable of goal-literal layers or None"
        )
    layers_out: list[list[GoalLiteralInput]] = []
    for layer_idx, layer in enumerate(value):
        if is_goal_literal_input(layer):
            location = f" entry at index {entry_idx}" if entry_idx is not None else ""
            raise TypeError(
                f"{field}{location} layer at position {layer_idx} must be an iterable of goal literals"
            )
        layers_out.append(
            _parse_goal_literals_iterable(
                layer,
                field=field,
                entry_idx=entry_idx,
            )
        )
    return layers_out


def _parse_history_payload(
    value: object,
    *,
    field: str,
    entry_idx: int | None = None,
) -> list[tuple[int, list[GoalLiteralInput]]]:
    if not _is_iterable_but_not_str_bytes(value):
        location = f" entry at index {entry_idx}" if entry_idx is not None else ""
        raise TypeError(
            f"{field}{location} must be an iterable of (dt, goal-literals) tuples or None"
        )

    out: list[tuple[int, list[GoalLiteralInput]]] = []
    for history_idx, item in enumerate(value):
        if not (
            isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], Integral)
        ):
            location = f" entry at index {entry_idx}" if entry_idx is not None else ""
            raise TypeError(
                f"{field}{location} item at position {history_idx} must be a (dt, literals) tuple"
            )
        dt, literals = item
        literals_list = _parse_goal_literals_iterable(
            literals,
            field=field,
            entry_idx=entry_idx,
        )
        out.append((int(dt), literals_list))
    return out


def _parse_optional_shared_or_per_state(
    value: object | None,
    *,
    field: str,
    state_count: int,
    parse_shared: Callable[[object], _PayloadT],
    parse_entry: Callable[[object, int], _PayloadT],
    entry_indicates_per_state: Callable[[object], bool],
) -> ParsedBatchPlan[_PayloadT]:
    if value is None:
        return ParsedBatchPlan(state_count=state_count, shared=None)

    if not _is_sequence_like_but_not_str_bytes(value):
        return ParsedBatchPlan(state_count=state_count, shared=parse_shared(value))

    outer = list(value)
    if any(entry_indicates_per_state(entry) for entry in outer):
        if len(outer) != state_count:
            raise ValueError(f"{field} length must match states length")
        out: list[_PayloadT | None] = []
        for idx, entry in enumerate(outer):
            if entry is None:
                out.append(None)
                continue
            out.append(parse_entry(entry, idx))
        return ParsedBatchPlan(state_count=state_count, per_state=out)

    try:
        shared = parse_shared(outer)
    except TypeError:
        candidate_per_state = len(outer) == state_count and all(
            entry is None or _is_iterable_but_not_str_bytes(entry) for entry in outer
        )
        if not candidate_per_state:
            raise
        out: list[_PayloadT | None] = []
        for idx, entry in enumerate(outer):
            if entry is None:
                out.append(None)
                continue
            out.append(parse_entry(entry, idx))
        return ParsedBatchPlan(state_count=state_count, per_state=out)

    return ParsedBatchPlan(state_count=state_count, shared=shared)


def parse_states_batch(states: StateBatchInput) -> list[StateInput]:
    if is_state_input(states):
        return [states]
    if isinstance(states, _STR_BYTES):
        raise TypeError("encode_batch expects a state or an iterable of states")
    return list(states)


def parse_goals_batch_param(
    goals: Iterable[GoalLiteralInput]
    | Sequence[Iterable[GoalLiteralInput] | None]
    | None,
    *,
    state_count: int,
) -> ParsedGoalsBatch:
    return _parse_optional_shared_or_per_state(
        goals,
        field="goals",
        state_count=state_count,
        parse_shared=lambda value: _parse_goal_literals_iterable(value, field="goals"),
        parse_entry=lambda value, idx: _parse_goal_literals_iterable(
            value,
            field="goals",
            entry_idx=idx,
        ),
        entry_indicates_per_state=lambda entry: (
            entry is None
            or (
                _is_sequence_like_but_not_str_bytes(entry)
                and not is_goal_literal_input(entry)
            )
        ),
    )


def parse_actions_batch_param(
    actions: (
        Iterable[GroundActionInput]
        | Sequence[Iterable[GroundActionInput] | None]
        | None
    ),
    *,
    state_count: int,
) -> ParsedActionsBatch:
    plan = parse_actions_batch_plan(actions, state_count=state_count)
    if plan.per_state_actions is not None:
        return ParsedBatchPlan(
            state_count=state_count,
            per_state=plan.per_state_actions,
        )
    shared_actions = plan.shared_actions if plan.shared_actions is not None else []
    return ParsedBatchPlan(state_count=state_count, shared=shared_actions)


def parse_subgoal_layers_batch_param(
    subgoal_layers: (
        Iterable[Iterable[GoalLiteralInput]]
        | Sequence[Iterable[Iterable[GoalLiteralInput]] | None]
        | None
    ),
    *,
    state_count: int,
) -> ParsedSubgoalLayersBatch:
    def _entry_indicates_per_state(entry: object) -> bool:
        if entry is None:
            return True
        # Avoid consuming iterators/generators during mode detection.
        if not _is_sequence_like_but_not_str_bytes(entry):
            return False
        if len(entry) == 0:
            return False
        first = entry[0]
        return not is_goal_literal_input(first)

    return _parse_optional_shared_or_per_state(
        subgoal_layers,
        field="subgoal_layers",
        state_count=state_count,
        parse_shared=lambda value: _parse_subgoal_layers_payload(
            value,
            field="subgoal_layers",
        ),
        parse_entry=lambda value, idx: _parse_subgoal_layers_payload(
            value,
            field="subgoal_layers",
            entry_idx=idx,
        ),
        entry_indicates_per_state=_entry_indicates_per_state,
    )


def parse_history_subgoals_batch_param(
    history_subgoals: HistorySubgoalInput | Sequence[HistorySubgoalInput | None] | None,
    *,
    state_count: int,
) -> ParsedHistorySubgoalsBatch:
    def _is_history_item(value: object) -> bool:
        return (
            isinstance(value, tuple)
            and len(value) == 2
            and isinstance(value[0], Integral)
        )

    def _entry_indicates_per_state(entry: object) -> bool:
        if entry is None:
            return True
        if _is_history_item(entry):
            return False
        if not _is_sequence_like_but_not_str_bytes(entry):
            return False
        if len(entry) == 0:
            return False
        first = entry[0]
        return _is_history_item(first)

    return _parse_optional_shared_or_per_state(
        history_subgoals,
        field="history_subgoals",
        state_count=state_count,
        parse_shared=lambda value: _parse_history_payload(
            value,
            field="history_subgoals",
        ),
        parse_entry=lambda value, idx: _parse_history_payload(
            value,
            field="history_subgoals",
            entry_idx=idx,
        ),
        entry_indicates_per_state=_entry_indicates_per_state,
    )


def parse_successors_batch_param(
    successors: StateBatchInput | None,
    *,
    state_count: int,
) -> list[StateInput]:
    if successors is None:
        raise ValueError("successors must be provided for transition batch encoding")
    if is_state_input(successors):
        out = [successors]
    else:
        if isinstance(successors, _STR_BYTES):
            raise TypeError("successors must be a state or an iterable of states")
        out = list(successors)
    if len(out) != state_count:
        raise ValueError("successors length must match states length")
    return out


def parse_dags_batch_param(
    dags: TransitionDAG | Iterable[TransitionDAG | None] | None,
    *,
    state_count: int,
) -> list[TransitionDAG | None]:
    if dags is None:
        return [None for _ in range(state_count)]
    if isinstance(dags, TransitionDAG):
        return [dags for _ in range(state_count)]
    if not _is_iterable_but_not_str_bytes(dags):
        raise TypeError("dags must be a TransitionDAG, iterable of dags, or None")
    out = list(dags)
    if len(out) != state_count:
        raise ValueError("dags length must match states length")
    for idx, dag in enumerate(out):
        if dag is not None and not isinstance(dag, TransitionDAG):
            raise TypeError(
                f"dags entry at index {idx} has invalid type: {type(dag)!r}"
            )
    return out


def reject_unsupported_batch_field(
    encoder_name: str,
    field_name: str,
    value: object,
) -> None:
    if value is not None:
        raise TypeError(
            f"{encoder_name} does not accept '{field_name}' in encode_batch"
        )
