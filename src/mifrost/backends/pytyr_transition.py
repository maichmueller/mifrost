"""PyTyr runtime for public transition HGraph encoders."""

from __future__ import annotations

from collections.abc import Iterable
from types import MappingProxyType
from typing import Any, Literal, cast

from ._relation_updates import relation_arities
from .pytyr import SemanticPlanningTaskAdapter
from .pytyr_flat import (
    _batch_param,
    _is_state,
    _lane_values,
    _sequence,
    _subgoal_values,
)
from ..encoders._flat_validation import validate_subgoal_layers_state_payload


class _PyTyrTransitionStream:
    def __init__(self, runtime: "PyTyrTransitionRuntime") -> None:
        self._runtime = runtime
        self._reuse_removed = False
        self.reset()

    def append(self, current: object, successor: object, **kwargs: Any) -> int:
        item = self._runtime._prepared(current, successor, **kwargs)
        if self._reuse_removed and self._removed:
            stream_id = min(self._removed)
            self._removed.remove(stream_id)
            self._items[stream_id] = item
            return stream_id
        stream_id = self._next_id
        self._next_id += 1
        self._items[stream_id] = item
        return stream_id

    def update(
        self,
        stream_id: int,
        current: object,
        successor: object,
        **kwargs: Any,
    ) -> None:
        if stream_id not in self._items:
            raise KeyError(stream_id)
        self._items[stream_id] = self._runtime._prepared(current, successor, **kwargs)

    def remove(self, stream_id: int) -> None:
        if stream_id not in self._items:
            raise KeyError(stream_id)
        del self._items[stream_id]
        self._removed.add(stream_id)

    def set_reuse_removed(self, value: bool) -> None:
        self._reuse_removed = bool(value)

    def flush(self) -> Any:
        # Each appended transition was prepared immediately, so the stream holds
        # no borrowed Tyr state between calls.
        pairs = [self._items[key] for key in sorted(self._items)]
        return self._runtime._direct.encode_prepared(pairs)

    def reset(self) -> None:
        self._items: dict[int, tuple[Any, Any]] = {}
        self._removed: set[int] = set()
        self._next_id = 0


class PyTyrTransitionRuntime:
    """Convert aligned PyTyr states into compact neutral successor inputs."""

    backend_name: Literal["pytyr"] = "pytyr"

    def __init__(self, planning_task: object, config: Any) -> None:
        self._adapter = SemanticPlanningTaskAdapter(planning_task)
        self.engine = self._adapter.make_successor_hgraph_engine(config)
        self._direct = self._adapter.make_direct_successor_encoder(config)
        self._relation_dict = MappingProxyType(dict(self.engine.relation_arities))

    @property
    def relation_dict(self) -> Any:
        return self._relation_dict

    def _validate_state(self, value: object, *, lane: str) -> None:
        if not _is_state(value):
            raise TypeError(
                f"a PyTyr transition encoder expects a PyTyr {lane} state, "
                f"got {type(value)!r}"
            )

    def _checked_lanes(
        self,
        current: object,
        successor: object,
        *,
        goals: object = None,
        subgoal_layers: object = None,
    ) -> tuple[Any, list[Any]]:
        self._validate_state(current, lane="current")
        self._validate_state(successor, lane="successor")
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
        return goals, layers

    def _inputs(
        self,
        current: object,
        successor: object,
        **kwargs: Any,
    ) -> tuple[Any, Any]:
        goals, layers = self._checked_lanes(current, successor, **kwargs)
        current_input = self._adapter.make_input(
            current,
            goals=cast(Any, goals),
            subgoal_layers=layers,
        )
        successor_input = self._adapter.make_input(successor)
        return current_input, successor_input

    def _prepared(
        self,
        current: object,
        successor: object,
        **kwargs: Any,
    ) -> tuple[Any, Any]:
        goals, layers = self._checked_lanes(current, successor, **kwargs)
        return self._direct.prepare(
            current,
            successor,
            goals=cast(Any, goals),
            subgoal_layers=layers,
        )

    def append_into_builder(
        self,
        current: object,
        successor: object,
        builder: Any,
        **kwargs: Any,
    ) -> None:
        # The caller owns a core-module `BatchBuilder`, which the PyTyr module
        # cannot accept across the nanobind ABI split; this path stays on the
        # owned-input route for that reason alone.
        current_input, successor_input = self._inputs(current, successor, **kwargs)
        self.engine.encode(current_input, successor_input, builder)

    def encode(self, current: object, successor: object, **kwargs: Any) -> Any:
        goals, layers = self._checked_lanes(current, successor, **kwargs)
        return self._direct.encode(
            current, successor, goals=cast(Any, goals), subgoal_layers=layers
        )

    @staticmethod
    def _successor_values(value: object, *, state_count: int) -> list[Any]:
        kind, payload = _batch_param(value)
        if kind == "shared" or _is_state(payload):
            values = [payload] * state_count
        else:
            values = _sequence(payload, field="successors")
            if len(values) != state_count:
                raise ValueError("successors length must match states length")
        if not all(_is_state(state) for state in values):
            bad_index = next(
                index for index, state in enumerate(values) if not _is_state(state)
            )
            raise TypeError(
                f"successors entry at index {bad_index} has invalid type "
                f"{type(values[bad_index])!r}"
            )
        return values

    def encode_batch(
        self,
        states: object,
        *,
        successors: object,
        goals: object = None,
        subgoal_layers: object = None,
    ) -> Any:
        current_values = (
            [states] if _is_state(states) else _sequence(states, field="states")
        )
        if not all(_is_state(state) for state in current_values):
            raise TypeError(
                "a PyTyr transition encoder batch can contain only PyTyr current states"
            )
        count = len(current_values)
        successor_values = self._successor_values(successors, state_count=count)

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
        for index, values in enumerate(subgoal_values):
            validate_subgoal_layers_state_payload(
                values,
                state_index=index,
                max_goal_level=int(self.engine.config.max_goal_level),
            )
        # Direct-View batch: both lanes are prepared and encoded in one native
        # crossing, so no owning semantic input exists at any point.
        return self._direct.encode_batch(
            current_values,
            successor_values,
            goals=goal_values,
            subgoal_layers=[
                () if values is None else values for values in subgoal_values
            ],
        )

    def update_relations(self, relation_dict: Any) -> None:
        relations = relation_arities(relation_dict)
        # Both engines: the direct encoder owns its own instance, so updating
        # only the compatibility engine would leave every encode on the stale
        # arity table.
        self.engine.update_relations(relations)
        self._direct.update_relations(relations)
        self._relation_dict = MappingProxyType(dict(relations))

    def make_stream(self) -> Any:
        return _PyTyrTransitionStream(self)


__all__ = ["PyTyrTransitionRuntime"]
