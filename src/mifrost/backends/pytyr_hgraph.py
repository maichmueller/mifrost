"""PyTyr runtime for the public heterogeneous graph encoder."""

from __future__ import annotations

from collections.abc import Iterable
from types import MappingProxyType
from typing import Any, Literal, cast

from pytyr.formalism.planning import GroundAction

from .pytyr import SemanticPlanningTaskAdapter
from .pytyr_flat import (
    _history_values,
    _is_state,
    _lane_values,
    _sequence,
    _subgoal_values,
)
from ..encoders._flat_validation import validate_subgoal_layers_state_payload


def _is_pytyr_action(value: object) -> bool:
    return isinstance(value, GroundAction)


class _PyTyrHGraphStream:
    def __init__(self, runtime: "PyTyrHGraphRuntime", *, mutable: bool) -> None:
        self._runtime = runtime
        self._mutable = mutable
        self._reuse_removed = False
        self.reset()

    def append(self, state: object, **kwargs: Any) -> int:
        item = self._runtime._input(state, **kwargs)
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
        self._items[stream_id] = self._runtime._input(state, **kwargs)

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
        return self._runtime.engine.encode_batch(
            [self._items[key] for key in sorted(self._items)]
        )

    def reset(self) -> None:
        self._items: dict[int, Any] = {}
        self._removed: set[int] = set()
        self._next_id = 0


class PyTyrHGraphRuntime:
    """Adapt PyTyr task/state/action views to compact semantic HGraph inputs."""

    backend_name: Literal["pytyr"] = "pytyr"

    def __init__(self, planning_task: object, config: Any) -> None:
        self._adapter = SemanticPlanningTaskAdapter(planning_task)
        self.engine = self._adapter.make_hgraph_engine(config)
        self._relation_dict = MappingProxyType(dict(self.engine.relation_arities))

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
                "a PyTyr HGraphEncoder expects a lifted or ground PyTyr state, "
                f"got {type(state)!r}"
            )
        action_values = list(cast(Iterable[object], () if actions is None else actions))
        if not all(_is_pytyr_action(action) for action in action_values):
            raise TypeError("a PyTyr HGraphEncoder accepts only PyTyr ground actions")
        layers = (
            []
            if subgoal_layers is None
            else list(cast(Iterable[Iterable[object]], subgoal_layers))
        )
        validate_subgoal_layers_state_payload(
            layers,
            state_index=0,
            max_goal_level=int(self.engine.config.max_goal_level),
        )
        return self._adapter.make_input(
            state,
            action_values,
            goals=cast(Iterable[object] | None, goals),
            subgoal_layers=layers,
            history=cast(
                Iterable[tuple[int, Iterable[object]]],
                () if history_subgoals is None else history_subgoals,
            ),
            history_max_steps=history_max_steps,
        )

    def append_into_builder(self, state: object, builder: Any, **kwargs: Any) -> None:
        self.engine.encode(self._input(state, **kwargs), builder)

    def encode_one(self, state: object, **kwargs: Any) -> Any:
        return self.engine.encode(self._input(state, **kwargs))

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
            raise TypeError("a PyTyr HGraphEncoder batch can contain only PyTyr states")
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
            actions, state_count=count, field="actions", leaf=_is_pytyr_action
        )
        for values in action_values:
            if values is not None and not all(
                _is_pytyr_action(value) for value in values
            ):
                raise TypeError(
                    "a PyTyr HGraphEncoder batch accepts only PyTyr ground actions"
                )
        subgoal_values = _subgoal_values(subgoal_layers, state_count=count)
        for index, values in enumerate(subgoal_values):
            validate_subgoal_layers_state_payload(
                values,
                state_index=index,
                max_goal_level=int(self.engine.config.max_goal_level),
            )
        history_values = _history_values(history_subgoals, state_count=count)
        inputs = self._adapter.make_inputs(
            state_values,
            [() if values is None else values for values in action_values],
            goals=goal_values,
            subgoal_layers=[
                () if values is None else values for values in subgoal_values
            ],
            history=[() if values is None else values for values in history_values],
            history_max_steps=history_max_steps,
        )
        return self.engine.encode_batch(inputs)

    def update_relations(self, relation_dict: Any) -> None:
        del relation_dict
        raise NotImplementedError(
            "update_relations is not implemented for the PyTyr HGraph backend"
        )

    def make_stream(self, *, mutable: bool) -> Any:
        return _PyTyrHGraphStream(self, mutable=mutable)


__all__ = ["PyTyrHGraphRuntime"]
