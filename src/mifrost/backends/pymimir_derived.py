"""Pymimir runtime for the public derived-graph encoder family."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any, Literal, cast

from .._neutral_core import (
    SemanticDerivedGraphEncoderConfig,
    SemanticDerivedGraphEncoderEngine,
)
from .flat import FlatSemanticAdapter
from .pymimir import PymimirSnapshotReader, _literal_key
from .pymimir_types import (
    DOMAIN_TYPES,
    is_action_input,
    is_goal_literal_input,
    is_state_input,
)
from .semantic import LiteralKey

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
    if isinstance(config, SemanticDerivedGraphEncoderConfig):
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
    return SemanticDerivedGraphEncoderConfig(**values)


def _semantic_literal(value: object) -> LiteralKey:
    """Accept semantic literal keys and native Pymimir goal literals alike."""
    if isinstance(value, LiteralKey):
        return value
    return _literal_key(value)


def _state_values(states: object) -> list[Any]:
    """Normalize one state or a state iterable into a list."""
    if is_state_input(states):
        return [states]
    if isinstance(states, _STR_BYTES) or not isinstance(states, Iterable):
        raise TypeError("states must be a state or an iterable of states")
    return list(cast("Iterable[Any]", states))


def _batch_lane(value: object) -> tuple[str, object]:
    kind = getattr(value, "kind", None)
    if kind in {"shared", "separate", "none"} and hasattr(value, "value"):
        return str(kind), getattr(value, "value")
    return "implicit", value


def _sequence_lane(
    value: object,
    *,
    state_count: int,
    field: str,
    leaf: Any,
) -> list[Any | None]:
    kind, payload = _batch_lane(value)
    if kind == "none" or payload is None:
        return [None] * state_count
    if kind == "shared":
        return [payload] * state_count
    if isinstance(payload, _STR_BYTES) or not isinstance(payload, Iterable):
        raise TypeError(f"{field} must be an iterable")
    values = list(cast("Iterable[Any]", payload))
    if kind == "separate":
        if len(values) != state_count:
            raise ValueError(f"{field} length must match states length")
        return values
    per_state = bool(values) and any(
        item is None or (not leaf(item) and isinstance(item, Sequence))
        for item in values
    )
    if per_state:
        if len(values) != state_count:
            raise ValueError(f"{field} length must match states length")
        return values
    return [values] * state_count


def _subgoal_lane(value: object, *, state_count: int) -> list[Any | None]:
    kind, payload = _batch_lane(value)
    if kind == "none" or payload is None:
        return [None] * state_count
    if kind == "shared":
        return [payload] * state_count
    if isinstance(payload, _STR_BYTES) or not isinstance(payload, Iterable):
        raise TypeError("subgoal_layers must be an iterable")
    values = list(cast("Iterable[Any]", payload))
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


def _history_entry(value: object) -> bool:
    return isinstance(value, tuple) and len(value) == 2 and isinstance(value[0], int)


class PymimirDerivedRuntime:
    """Encode Pymimir wrapper states as derived graphs via the flat reader."""

    backend_name: Literal["pymimir"] = "pymimir"

    def __init__(self, problem: object, config: Any) -> None:
        """Build the flat adapter and neutral derived-graph engine."""
        if isinstance(problem, DOMAIN_TYPES):
            raise TypeError(
                "Derived-graph encoders need a pymimir Problem, not a Domain: "
                "object tables, static facts, and goal literals are problem-scoped"
            )
        self._adapter = FlatSemanticAdapter(PymimirSnapshotReader(problem))
        self.engine = SemanticDerivedGraphEncoderEngine(
            self._adapter.engine.predicates, _native_config(config)
        )

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
        """Build one semantic flat input, defaulting goals to the problem's."""
        return self._adapter.make_input(
            state,
            goals=None
            if goals is None
            else [_semantic_literal(value) for value in cast("Iterable[Any]", goals)],
            actions=() if actions is None else cast("Iterable[Any]", actions),
            subgoal_layers=(
                ()
                if subgoal_layers is None
                else [
                    [_semantic_literal(value) for value in layer]
                    for layer in cast("Iterable[Iterable[Any]]", subgoal_layers)
                ]
            ),
            history=(
                ()
                if history_subgoals is None
                else [
                    (int(delta), [_semantic_literal(value) for value in literals])
                    for delta, literals in cast("Iterable[Any]", history_subgoals)
                ]
            ),
            history_max_steps=history_max_steps,
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
        """Encode one state; goals default to the problem's own goals."""
        return self.engine.encode(
            self._input(
                state,
                goals=goals,
                actions=actions,
                subgoal_layers=subgoal_layers,
                history_subgoals=history_subgoals,
                history_max_steps=history_max_steps,
            )
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
        state_values = _state_values(states)
        count = len(state_values)
        goal_values = _sequence_lane(
            goals,
            state_count=count,
            field="goals",
            leaf=is_goal_literal_input,
        )
        action_values = _sequence_lane(
            actions,
            state_count=count,
            field="actions",
            leaf=is_action_input,
        )
        layer_values = _subgoal_lane(subgoal_layers, state_count=count)
        history_values = _sequence_lane(
            history_subgoals,
            state_count=count,
            field="history_subgoals",
            leaf=_history_entry,
        )
        inputs = [
            self._input(
                state,
                goals=goal_values[index],
                actions=action_values[index],
                subgoal_layers=layer_values[index],
                history_subgoals=history_values[index],
                history_max_steps=history_max_steps,
            )
            for index, state in enumerate(state_values)
        ]
        return self.engine.encode_batch(inputs)

    def make_stream(self) -> "_PymimirDerivedStream":
        """Create a pure-Python streaming encoder over this runtime."""
        return _PymimirDerivedStream(self)


class _PymimirDerivedStream:
    """Pure-Python append/update/remove buffer over prepared semantic inputs."""

    def __init__(self, runtime: PymimirDerivedRuntime) -> None:
        self._runtime = runtime
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
        if stream_id not in self._items:
            raise KeyError(stream_id)
        self._items[stream_id] = self._runtime._input(state, **kwargs)

    def remove(self, stream_id: int) -> None:
        if stream_id not in self._items:
            raise KeyError(stream_id)
        del self._items[stream_id]
        self._removed.add(stream_id)

    def set_reuse_removed(self, value: bool) -> None:
        self._reuse_removed = bool(value)

    def flush(self) -> Any:
        values = [self._items[key] for key in sorted(self._items)]
        return self._runtime.engine.encode_batch(values)

    def reset(self) -> None:
        self._items: dict[int, Any] = {}
        self._removed: set[int] = set()
        self._next_id = 0


__all__ = ["PymimirDerivedRuntime"]
