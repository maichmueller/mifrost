"""PyTyr runtime for the public flat relation encoder."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from types import MappingProxyType
from typing import Any, Literal, cast

from pytyr.formalism.planning import PlanningTask

from .pytyr import SemanticFlatRelationEncoder
from ..encoders._flat_validation import validate_subgoal_layers_state_payload


_STR_BYTES = (str, bytes, bytearray)


def _sequence(value: object, *, field: str) -> list[Any]:
    if isinstance(value, _STR_BYTES) or not isinstance(value, Iterable):
        raise TypeError(f"{field} must be an iterable")
    return list(value)


def _is_state(value: object) -> bool:
    return all(
        hasattr(value, member)
        for member in ("static_atoms", "fluent_facts", "derived_atoms")
    )


def _is_action(value: object) -> bool:
    return hasattr(value, "get_action") and hasattr(value, "get_objects")


def _is_history_entry(value: object) -> bool:
    return isinstance(value, tuple) and len(value) == 2 and isinstance(value[0], int)


def _batch_param(value: object) -> tuple[str, object]:
    kind = getattr(value, "kind", None)
    if kind in {"shared", "separate", "none"} and hasattr(value, "value"):
        return str(kind), getattr(value, "value")
    return "implicit", value


def _lane_values(
    value: object,
    *,
    state_count: int,
    field: str,
    leaf: Any,
) -> list[Any | None]:
    kind, payload = _batch_param(value)
    if kind == "none" or payload is None:
        return [None] * state_count
    if kind == "shared":
        return [payload] * state_count
    if kind == "separate":
        values = _sequence(payload, field=field)
        if len(values) != state_count:
            raise ValueError(f"{field} length must match states length")
        return values

    values = _sequence(payload, field=field)
    per_state = bool(values) and any(
        item is None or (not leaf(item) and isinstance(item, Sequence))
        for item in values
    )
    if per_state:
        if len(values) != state_count:
            raise ValueError(f"{field} length must match states length")
        return values
    return [values] * state_count


def _subgoal_values(value: object, *, state_count: int) -> list[Any | None]:
    kind, payload = _batch_param(value)
    if kind == "none" or payload is None:
        return [None] * state_count
    if kind == "shared":
        return [payload] * state_count
    values = _sequence(payload, field="subgoal_layers")
    if kind == "separate":
        if len(values) != state_count:
            raise ValueError("subgoal_layers length must match states length")
        return values
    per_state = any(
        item is None
        or (isinstance(item, Sequence) and bool(item) and isinstance(item[0], Sequence))
        for item in values
    )
    if per_state:
        if len(values) != state_count:
            raise ValueError("subgoal_layers length must match states length")
        return values
    return [values] * state_count


def _history_values(value: object, *, state_count: int) -> list[Any | None]:
    return _lane_values(
        value,
        state_count=state_count,
        field="history_subgoals",
        leaf=_is_history_entry,
    )


class _PyTyrFlatStream:
    def __init__(self, runtime: "PyTyrFlatRuntime", *, mutable: bool) -> None:
        self._runtime = runtime
        self._mutable = mutable
        self._reuse_removed = False
        self.reset()

    def append(self, state: object, **kwargs: Any) -> int:
        item = self._runtime._prepared(state, **kwargs)
        if self._reuse_removed and self._removed:
            stream_id = min(self._removed)
            self._removed.remove(stream_id)
            self._items[stream_id] = item
            return stream_id
        stream_id = self._next_id
        self._next_id += 1
        self._items[stream_id] = item
        return stream_id

    def update(self, stream_id: int, state: object, **kwargs: Any) -> None:
        if not self._mutable:
            raise NotImplementedError("update is not implemented for this stream")
        if stream_id not in self._items:
            raise KeyError(stream_id)
        self._items[stream_id] = self._runtime._prepared(state, **kwargs)

    def remove(self, stream_id: int) -> None:
        if not self._mutable:
            raise NotImplementedError("remove is not implemented for this stream")
        if stream_id not in self._items:
            raise KeyError(stream_id)
        del self._items[stream_id]
        self._removed.add(stream_id)

    def set_reuse_removed(self, value: bool) -> None:
        self._reuse_removed = bool(value)

    def flush(self) -> Any:
        # Each appended step was prepared immediately, so the stream holds no
        # borrowed Tyr state between calls; the handles are borrowed, not
        # consumed, so a flush may be repeated.
        values = [self._items[key] for key in sorted(self._items)]
        return self._runtime._direct.encode_prepared(values)

    def reset(self) -> None:
        self._items: dict[int, Any] = {}
        self._removed: set[int] = set()
        self._next_id = 0


class PyTyrFlatRuntime:
    """Adapt PyTyr task/state/action views to compact semantic inputs."""

    backend_name: Literal["pytyr"] = "pytyr"

    def __init__(self, planning_task: PlanningTask, config: Any) -> None:
        self._adapter = SemanticFlatRelationEncoder(planning_task, config)
        self.engine = self._adapter.engine
        self._direct = self._adapter.direct
        self._relation_dict = MappingProxyType(
            dict(
                zip(
                    self.engine.relation_names,
                    self.engine.relation_arities,
                    strict=True,
                )
            )
        )

    @property
    def relation_dict(self) -> Any:
        return self._relation_dict

    def _input(
        self,
        state: object,
        *,
        goals: object = None,
        actions: object = None,
        subgoal_layers: object = None,
        history_subgoals: object = None,
        history_max_steps: int | None = None,
    ) -> Any:
        if not _is_state(state):
            raise TypeError(
                "a PyTyr FlatRelationEncoder expects a lifted or ground PyTyr "
                f"state, got {type(state)!r}"
            )
        return self._adapter.make_input(
            state,
            cast(Iterable[object], () if actions is None else actions),
            goals=cast(Iterable[object] | None, goals),
            subgoal_layers=cast(
                Iterable[Iterable[object]],
                () if subgoal_layers is None else subgoal_layers,
            ),
            history=cast(
                Iterable[tuple[int, Iterable[object]]],
                () if history_subgoals is None else history_subgoals,
            ),
            history_max_steps=history_max_steps,
        )

    def _prepared(self, state: object, **kwargs: Any) -> Any:
        """Prepare one appended step without building an owning input."""
        if not _is_state(state):
            raise TypeError(
                "a PyTyr FlatRelationEncoder expects a lifted or ground PyTyr "
                f"state, got {type(state)!r}"
            )
        return self._direct.prepare(
            state,
            cast(
                Iterable[object],
                () if kwargs.get("actions") is None else kwargs["actions"],
            ),
            goals=cast(Iterable[object] | None, kwargs.get("goals")),
            subgoal_layers=cast(
                Iterable[Iterable[object]],
                ()
                if kwargs.get("subgoal_layers") is None
                else kwargs["subgoal_layers"],
            ),
            history=cast(
                Iterable[tuple[int, Iterable[object]]],
                ()
                if kwargs.get("history_subgoals") is None
                else kwargs["history_subgoals"],
            ),
            history_max_steps=kwargs.get("history_max_steps"),
        )

    def encode(self, state: object, **kwargs: Any) -> Any:
        if not _is_state(state):
            raise TypeError(
                "a PyTyr FlatRelationEncoder expects a lifted or ground PyTyr "
                f"state, got {type(state)!r}"
            )
        # Direct-View encode inside the PyTyr module; no owning semantic input.
        return self._adapter.encode(
            state,
            cast(
                Iterable[object],
                () if kwargs.get("actions") is None else kwargs["actions"],
            ),
            goals=cast(Iterable[object] | None, kwargs.get("goals")),
            subgoal_layers=cast(
                Iterable[Iterable[object]],
                ()
                if kwargs.get("subgoal_layers") is None
                else kwargs["subgoal_layers"],
            ),
            history=cast(
                Iterable[tuple[int, Iterable[object]]],
                ()
                if kwargs.get("history_subgoals") is None
                else kwargs["history_subgoals"],
            ),
            history_max_steps=kwargs.get("history_max_steps"),
        )

    def encode_batch(
        self,
        states: object,
        *,
        goals: object = None,
        actions: object = None,
        subgoal_layers: object = None,
        history_subgoals: object = None,
        history_max_steps: int | None = None,
    ) -> Any:
        state_values = (
            [states] if _is_state(states) else _sequence(states, field="states")
        )
        if not all(_is_state(state) for state in state_values):
            raise TypeError(
                "a PyTyr FlatRelationEncoder batch can contain only PyTyr states"
            )
        count = len(state_values)

        def is_literal(value: object) -> bool:
            try:
                self._adapter._literal_key(value)
            except (TypeError, ValueError):
                return False
            return True

        goal_values = _lane_values(
            goals, state_count=count, field="goals", leaf=is_literal
        )
        action_values = _lane_values(
            actions, state_count=count, field="actions", leaf=_is_action
        )
        subgoal_values = _subgoal_values(subgoal_layers, state_count=count)
        for index, value in enumerate(subgoal_values):
            validate_subgoal_layers_state_payload(
                value,
                state_index=index,
                max_goal_level=int(self.engine.config.max_goal_level),
            )
        history_values = _history_values(history_subgoals, state_count=count)
        # Direct-View batch: the states are prepared and encoded in one native
        # crossing, so no owning semantic input exists at any point.
        return self._direct.encode_batch(
            state_values,
            [() if values is None else values for values in action_values],
            goals=goal_values,
            subgoal_layers=[
                () if values is None else values for values in subgoal_values
            ],
            history=[() if values is None else values for values in history_values],
            history_max_steps=history_max_steps,
        )

    def append_into_builder(self, state: object, builder: Any, **kwargs: Any) -> None:
        # The caller owns a `BatchBuilder` registered in the core module's
        # nanobind registry, which the PyTyr module cannot accept. This is the
        # one flat path that still crosses as an owned input, and the reason is
        # the ABI split itself, not the encoding.
        self.engine.encode(self._input(state, **kwargs), builder)

    def make_stream(self, *, mutable: bool) -> Any:
        return _PyTyrFlatStream(self, mutable=mutable)


__all__ = ["PyTyrFlatRuntime"]
