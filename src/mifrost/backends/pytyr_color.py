"""PyTyr runtime for the public homogeneous Color encoder."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Literal, cast

from .pytyr import SemanticPlanningTaskAdapter
from .pytyr_flat import _is_state, _lane_values, _sequence, _subgoal_values
from ..encoders._flat_validation import validate_subgoal_layers_state_payload


_ACTION_ERROR = "ColorEncoderEngine does not support action encoding"
_BATCH_ACTION_ERROR = "Color batch encoding does not support explicit action payloads"


class _PyTyrColorStream:
    def __init__(self, runtime: "PyTyrColorRuntime") -> None:
        self._runtime = runtime
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
        if stream_id not in self._items:
            raise KeyError(stream_id)
        self._items[stream_id] = self._runtime._prepared(state, **kwargs)

    def remove(self, stream_id: int) -> None:
        if stream_id not in self._items:
            raise KeyError(stream_id)
        del self._items[stream_id]
        self._removed.add(stream_id)

    def set_reuse_removed(self, value: bool) -> None:
        self._reuse_removed = bool(value)

    def flush(self) -> Any:
        # Each appended step was prepared immediately, so the stream holds no
        # borrowed Tyr state between calls.
        values = [self._items[key] for key in sorted(self._items)]
        return self._runtime._direct.encode_prepared(values)

    def reset(self) -> None:
        self._items: dict[int, Any] = {}
        self._removed: set[int] = set()
        self._next_id = 0


class PyTyrColorRuntime:
    """Adapt PyTyr task/state views to compact semantic Color inputs."""

    backend_name: Literal["pytyr"] = "pytyr"

    def __init__(self, planning_task: object, config: Any) -> None:
        self._adapter = SemanticPlanningTaskAdapter(planning_task)
        self.engine = self._adapter.make_color_engine(config)
        self._direct = self._adapter.make_direct_color_encoder(config)

    def _checked_lanes(
        self,
        state: object,
        *,
        goals: object = None,
        actions: object = None,
        subgoal_layers: object = None,
    ) -> tuple[Any, list[Any]]:
        if not _is_state(state):
            raise TypeError(
                "a PyTyr ColorEncoder expects a lifted or ground PyTyr state, "
                f"got {type(state)!r}"
            )
        action_values = [] if actions is None else list(cast(Iterable[Any], actions))
        if action_values:
            raise ValueError(_ACTION_ERROR)
        layers = (
            []
            if subgoal_layers is None
            else list(cast(Iterable[Iterable[Any]], subgoal_layers))
        )
        validate_subgoal_layers_state_payload(layers, state_index=0, max_goal_level=3)
        return goals, layers

    def _input(self, state: object, **kwargs: Any) -> Any:
        goals, layers = self._checked_lanes(state, **kwargs)
        return self._adapter.make_input(
            state,
            goals=cast(Iterable[object] | None, goals),
            subgoal_layers=layers,
        )

    def _prepared(self, state: object, **kwargs: Any) -> Any:
        goals, layers = self._checked_lanes(state, **kwargs)
        return self._direct.prepare(
            state,
            goals=cast(Iterable[object] | None, goals),
            subgoal_layers=layers,
        )

    def encode_one(self, state: object, **kwargs: Any) -> Any:
        goals, layers = self._checked_lanes(state, **kwargs)
        # Direct-View encode inside the PyTyr module; no owning semantic input.
        return self._direct.encode(
            state,
            goals=cast(Iterable[object] | None, goals),
            subgoal_layers=layers,
        )

    def encode_batch(
        self,
        states: object,
        *,
        goals: object = None,
        actions: object = None,
        subgoal_layers: object = None,
    ) -> Any:
        state_values = (
            [states] if _is_state(states) else _sequence(states, field="states")
        )
        if not all(_is_state(state) for state in state_values):
            raise TypeError("a PyTyr ColorEncoder batch can contain only PyTyr states")
        count = len(state_values)

        action_values = _lane_values(
            actions, state_count=count, field="actions", leaf=lambda _value: True
        )
        if any(values for values in action_values):
            raise ValueError(_BATCH_ACTION_ERROR)

        def is_literal(value: object) -> bool:
            try:
                self._adapter._literal_key(value)
            except (TypeError, ValueError):
                return False
            return True

        goal_values = _lane_values(
            goals, state_count=count, field="goals", leaf=is_literal
        )
        subgoal_values = _subgoal_values(subgoal_layers, state_count=count)
        for index, value in enumerate(subgoal_values):
            validate_subgoal_layers_state_payload(
                value, state_index=index, max_goal_level=3
            )
        # Direct-View batch: prepared and encoded in one native crossing.
        return self._direct.encode_batch(
            state_values,
            goals=goal_values,
            subgoal_layers=[
                () if values is None else values for values in subgoal_values
            ],
        )

    def make_stream(self) -> Any:
        return _PyTyrColorStream(self)


__all__ = ["PyTyrColorRuntime"]
