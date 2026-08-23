"""PyTyr runtime for the public derived-graph encoder family."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any, Literal, TypeGuard, cast

from pytyr.formalism.planning import PlanningTask

from .pytyr_flat import (
    _history_values,
    _is_action,
    _is_state,
    _lane_values,
    _sequence,
    _subgoal_values,
)
from .pytyr import SemanticPlanningTaskAdapter


_NODE_UNIVERSE_OPTIONS = ("objects_and_atoms", "objects_only")
_ATOM_EXPANSION_OPTIONS = ("star", "clique", "chain", "star_first")
_STR_BYTES = (str, bytes, bytearray)


def _config_option(value: object, options: tuple[str, ...], field: str) -> str:
    """Map a string, enum-like, or integer config value to its native name."""
    if isinstance(value, str) and value in options:
        return value
    name = getattr(value, "name", None)
    if isinstance(name, str) and name in options:
        return name
    try:
        return options[int(cast("int", value))]
    except (TypeError, ValueError, IndexError) as error:
        raise ValueError(f"Unknown {field} {value!r}") from error


def _native_config(config: Any) -> Any:
    """Return the neutral native config for any supported config shape."""
    from mifrost import _neutral_core

    if isinstance(config, _neutral_core.SemanticDerivedGraphEncoderConfig):
        return config
    values = {
        field: getattr(config, field)
        for field in (
            "node_universe",
            "atom_expansion",
            "include_reverse_edges",
            "export_node_names",
            "include_line_graph",
            "line_graph_max_degree",
            "include_hyperedge_incidence",
            "include_tuple_tensors",
            "include_spd",
            "spd_max_hops",
        )
        if hasattr(config, field)
    }
    if "node_universe" in values:
        values["node_universe"] = _config_option(
            values["node_universe"], _NODE_UNIVERSE_OPTIONS, "node_universe"
        )
    if "atom_expansion" in values:
        values["atom_expansion"] = _config_option(
            values["atom_expansion"], _ATOM_EXPANSION_OPTIONS, "atom_expansion"
        )
    return _neutral_core.SemanticDerivedGraphEncoderConfig(**values)


def _is_sequence_like(value: object) -> TypeGuard[Sequence[Any]]:
    return isinstance(value, Sequence) and not isinstance(value, _STR_BYTES)


def _check_layer_shapes(layers: object, *, state_index: int) -> None:
    """Validate subgoal layer nesting; the layer count itself is uncapped."""
    if layers is None:
        return
    if not _is_sequence_like(layers):
        raise TypeError(
            f"subgoal_layers entry at state index {state_index} must be an "
            "iterable of goal-literal layers or None"
        )
    for layer_index, layer in enumerate(layers):
        if not _is_sequence_like(layer):
            raise TypeError(
                f"subgoal_layers entry at state index {state_index} layer at "
                f"position {layer_index} must be an iterable of goal literals"
            )


class _PyTyrDerivedStream:
    def __init__(self, runtime: "PyTyrDerivedRuntime") -> None:
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


class PyTyrDerivedRuntime:
    """Adapt PyTyr task/state views to derived-graph encoder inputs."""

    backend_name: Literal["pytyr"] = "pytyr"

    def __init__(self, planning_task: PlanningTask, config: Any) -> None:
        """Build the neutral engine and direct encoder for one planning task."""
        self._adapter = SemanticPlanningTaskAdapter(planning_task)
        native_config = _native_config(config)
        self.engine = self._adapter.make_derived_engine(native_config)
        self._direct = self._adapter.make_direct_derived_encoder(native_config)

    def _check_state(self, state: object) -> None:
        if not _is_state(state):
            raise TypeError(
                "a PyTyr derived-graph encoder expects a lifted or ground "
                f"PyTyr state, got {type(state)!r}"
            )

    def _lanes(
        self,
        goals: object,
        actions: object,
        subgoal_layers: object,
        history_subgoals: object,
    ) -> tuple[Any, Any, Any, Any]:
        """Normalize the four optional lanes of one state encoding."""
        _check_layer_shapes(subgoal_layers, state_index=0)
        return (
            cast("Iterable[object] | None", goals),
            cast("Iterable[object]", () if actions is None else actions),
            cast(
                "Iterable[Iterable[object]]",
                () if subgoal_layers is None else subgoal_layers,
            ),
            cast(
                "Iterable[tuple[int, Iterable[object]]]",
                () if history_subgoals is None else history_subgoals,
            ),
        )

    def _prepared(self, state: object, **kwargs: Any) -> Any:
        self._check_state(state)
        goals, actions, layers, history = self._lanes(
            kwargs.get("goals"),
            kwargs.get("actions"),
            kwargs.get("subgoal_layers"),
            kwargs.get("history_subgoals"),
        )
        return self._direct.prepare(
            state,
            actions,
            goals=goals,
            subgoal_layers=layers,
            history=history,
            history_max_steps=kwargs.get("history_max_steps"),
        )

    def encode(
        self,
        state: object,
        *,
        goals: object = None,
        actions: object = None,
        subgoal_layers: object = None,
        history_subgoals: object = None,
        history_max_steps: int | None = None,
    ) -> Any:
        """Encode one state into a derived graph."""
        self._check_state(state)
        compact_goals, actions, layers, history = self._lanes(
            goals, actions, subgoal_layers, history_subgoals
        )
        # Direct-View encode inside the PyTyr module; no owning semantic input.
        return self._direct.encode(
            state,
            actions,
            goals=compact_goals,
            subgoal_layers=layers,
            history=history,
            history_max_steps=history_max_steps,
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
        """Encode a batch of states into one derived-graph batch encoding."""
        state_values = (
            [states] if _is_state(states) else _sequence(states, field="states")
        )
        if not all(_is_state(state) for state in state_values):
            raise TypeError("a PyTyr derived-graph batch can contain only PyTyr states")
        count = len(state_values)

        def is_literal(value: object) -> bool:
            try:
                self._adapter._literal_key(value)
            except (TypeError, ValueError):
                return False
            return True

        action_values = _lane_values(
            actions, state_count=count, field="actions", leaf=_is_action
        )
        history_values = _history_values(history_subgoals, state_count=count)
        goal_values = _lane_values(
            goals, state_count=count, field="goals", leaf=is_literal
        )
        subgoal_values = _subgoal_values(subgoal_layers, state_count=count)
        for index, value in enumerate(subgoal_values):
            _check_layer_shapes(value, state_index=index)
        # Direct-View batch: prepared and encoded in one native crossing.
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

    def make_stream(self) -> Any:
        """Create a pure-Python streaming encoder over this runtime."""
        return _PyTyrDerivedStream(self)


__all__ = ["PyTyrDerivedRuntime"]
