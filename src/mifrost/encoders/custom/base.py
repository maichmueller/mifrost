"""Custom-encoder facade: StateView + GraphWriter behind EncoderBase.

Subclasses of `CustomGraphEncoder` implement exactly one method,
`encode_state`, receiving already-neutralized lanes (tuples of `Literal` /
`Atom` records). Batching, streaming, kwargs validation and PyG conversion
come from the shared encoder machinery.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from ..base import EncoderBase, StreamEncoderBase
from .state_view import Atom, Literal, StateView
from .writer import GraphWriter

_STANDARD_LANES = frozenset(
    {
        "goals",
        "actions",
        "subgoal_layers",
        "history_subgoals",
        "history_max_steps",
    }
)
_STR_BYTES = (str, bytes, bytearray)


def _batch_param(value: object) -> tuple[str, object]:
    kind = getattr(value, "kind", None)
    if kind in {"shared", "separate", "none"} and hasattr(value, "value"):
        return str(kind), getattr(value, "value")
    return "implicit", value


def _sequence(value: object, *, field: str) -> list[Any]:
    if isinstance(value, _STR_BYTES) or not isinstance(value, Iterable):
        raise TypeError(f"{field} must be an iterable")
    return list(value)


def _is_history_entry(value: object) -> bool:
    return (
        isinstance(value, tuple)
        and len(value) == 2
        and isinstance(value[0], int)
        and isinstance(value[1], Iterable)
    )


class CustomGraphEncoder(EncoderBase[Any]):
    """Base class for pure-Python custom graph encoders.

    Subclasses set `accepted_kwargs` for extra lanes beyond the standard five
    and override `encode_state` to draw one graph into a fresh `GraphWriter`.
    """

    accepted_kwargs: frozenset[str] = frozenset()

    def __init__(
        self,
        source: Any,
        *,
        graph_kind: str = "homo",
        export_node_names: bool = True,
        backend: str | None = None,
    ) -> None:
        """Build one `StateView` over a pymimir problem or pytyr task."""

        self.view = StateView(source, backend=backend)
        self.backend = self.view.backend
        self.graph_kind = graph_kind
        self.export_node_names = export_node_names

    def encode_state(
        self,
        out: GraphWriter,
        state: Any,
        *,
        goals: tuple[Literal, ...] | None = None,
        actions: tuple[Atom, ...] | None = None,
        subgoal_layers: tuple[tuple[Literal, ...], ...] | None = None,
        history_subgoals: tuple[tuple[int, tuple[Literal, ...]], ...] | None = None,
        history_max_steps: int | None = None,
        **kwargs: Any,
    ) -> None:
        """Draw one state's graph into ``out``; the only required override."""

        raise NotImplementedError(
            f"{type(self).__name__} must implement encode_state()"
        )

    def stream(self) -> CustomStream:
        """Create a pure-Python streaming variant of this encoder."""

        return CustomStream(self)

    def _accepted_kwargs(self) -> set[str]:
        return set(_STANDARD_LANES | set(self.accepted_kwargs))

    def _new_writer(self) -> GraphWriter:
        return GraphWriter(
            self.view,
            graph_kind=self.graph_kind,
            export_node_names=self.export_node_names,
        )

    def _neutralize_lanes(
        self,
        state: Any,
        *,
        goals: Any,
        actions: Any,
        subgoal_layers: Any,
        history_subgoals: Any,
    ) -> dict[str, Any]:
        """Convert native lane leaves into neutral records via the view."""

        view = self.view
        return {
            "goals": (
                None if goals is None else view.neutral_literals(goals, field="goals")
            ),
            "actions": (
                None if actions is None else view.neutral_actions(state, actions)
            ),
            "subgoal_layers": (
                None
                if subgoal_layers is None
                else tuple(
                    view.neutral_literals(layer, field="subgoal_layers")
                    for layer in subgoal_layers
                )
            ),
            "history_subgoals": (
                None
                if history_subgoals is None
                else tuple(
                    (
                        int(delta),
                        view.neutral_literals(literals, field="history_subgoals"),
                    )
                    for delta, literals in history_subgoals
                )
            ),
        }

    def _encode(
        self,
        state: Any,
        *,
        goals: Any = None,
        actions: Any = None,
        subgoal_layers: Any = None,
        **kwargs: Any,
    ) -> Any:
        history_subgoals = kwargs.pop("history_subgoals", None)
        history_max_steps = kwargs.pop("history_max_steps", None)
        out = self._new_writer()
        lanes = self._neutralize_lanes(
            state,
            goals=goals,
            actions=actions,
            subgoal_layers=subgoal_layers,
            history_subgoals=history_subgoals,
        )
        self.encode_state(
            out, state, **lanes, history_max_steps=history_max_steps, **kwargs
        )
        return out.finish()

    def _encode_batch(
        self,
        states: Any,
        *,
        goals: Any = None,
        actions: Any = None,
        subgoal_layers: Any = None,
        **kwargs: Any,
    ) -> Any:
        from ... import batch_encodings

        history_subgoals = kwargs.pop("history_subgoals", None)
        history_max_steps = kwargs.pop("history_max_steps", None)
        state_values = self._state_values(states)
        count = len(state_values)
        goal_lanes = self._split_lane(
            goals, count=count, field="goals", leaf=self._is_goal_leaf
        )
        action_lanes = self._split_lane(
            actions, count=count, field="actions", leaf=self._is_action_leaf
        )
        layer_lanes = self._split_layers(subgoal_layers, count=count)
        history_lanes = self._split_lane(
            history_subgoals,
            count=count,
            field="history_subgoals",
            leaf=_is_history_entry,
        )
        encodings = []
        for index, state in enumerate(state_values):
            encodings.append(
                self._encode(
                    state,
                    goals=goal_lanes[index],
                    actions=action_lanes[index],
                    subgoal_layers=layer_lanes[index],
                    history_subgoals=history_lanes[index],
                    history_max_steps=history_max_steps,
                    **kwargs,
                )
            )
        return batch_encodings(encodings)

    def _state_values(self, states: Any) -> list[Any]:
        """Normalize one state or a state iterable into a list."""

        if _looks_like_state(states):
            return [states]
        values = _sequence(states, field="states")
        if not all(_looks_like_state(value) for value in values):
            raise TypeError("states must be planning states or iterables of them")
        return values

    def _is_goal_leaf(self, value: object) -> bool:
        if self.view._literal_from_native(value) is not None:
            return True
        return not isinstance(value, Sequence)

    def _is_action_leaf(self, value: object) -> bool:
        if self.view._action_from_native(value) is not None:
            return True
        return not isinstance(value, Sequence)

    def _split_lane(
        self,
        value: object,
        *,
        count: int,
        field: str,
        leaf: Any,
    ) -> list[Any | None]:
        """Split a batch lane into per-state values (shared/separate/implicit)."""

        kind, payload = _batch_param(value)
        if kind == "none" or payload is None:
            return [None] * count
        if kind == "shared":
            return [payload] * count
        if kind == "separate":
            values = _sequence(payload, field=field)
            if len(values) != count:
                raise ValueError(f"{field} length must match states length")
            return values
        values = _sequence(payload, field=field)
        per_state = bool(values) and any(
            item is None or (not leaf(item) and isinstance(item, Sequence))
            for item in values
        )
        if per_state:
            if len(values) != count:
                raise ValueError(f"{field} length must match states length")
            return values
        return [values] * count

    def _split_layers(self, value: object, *, count: int) -> list[Any | None]:
        """Split a subgoal-layers lane where leaves are literal layers."""

        kind, payload = _batch_param(value)
        if kind == "none" or payload is None:
            return [None] * count
        if kind == "shared":
            return [payload] * count
        if kind == "separate":
            values = _sequence(payload, field="subgoal_layers")
            if len(values) != count:
                raise ValueError("subgoal_layers length must match states length")
            return values
        values = _sequence(payload, field="subgoal_layers")
        per_state = any(
            item is None
            or (
                isinstance(item, Sequence)
                and bool(item)
                and isinstance(item[0], Sequence)
            )
            for item in values
        )
        if per_state:
            if len(values) != count:
                raise ValueError("subgoal_layers length must match states length")
            return values
        return [values] * count


@dataclass
class CustomStream(StreamEncoderBase[Any]):
    """Pure-Python stream storing ``(state, kwargs)`` recipes until flush.

    Recipes hold references to their states, so appended states must outlive
    the stream: nothing is encoded until `flush` re-runs every recipe through
    its encoder in id order.
    """

    encoder: CustomGraphEncoder

    def __post_init__(self) -> None:
        self._reuse_removed = False
        self.reset()

    def append(self, state: Any, **kwargs: Any) -> int:
        """Store one encoding recipe and return its stream id."""

        if self._reuse_removed and self._removed:
            stream_id = min(self._removed)
            self._removed.remove(stream_id)
        else:
            stream_id = self._next_id
            self._next_id += 1
        self._items[stream_id] = (state, dict(kwargs))
        return stream_id

    def update(self, stream_id: int, state: Any, **kwargs: Any) -> None:
        """Replace the recipe stored at ``stream_id``."""

        if stream_id not in self._items:
            raise KeyError(stream_id)
        self._items[stream_id] = (state, dict(kwargs))

    def remove(self, stream_id: int) -> None:
        """Drop one recipe; its slot is reusable when reuse is enabled."""

        if stream_id not in self._items:
            raise KeyError(stream_id)
        del self._items[stream_id]
        self._removed.add(stream_id)

    def set_reuse_removed(self, value: bool) -> None:
        """Enable or disable removed-slot reuse (replace-in-order semantics)."""

        self._reuse_removed = bool(value)

    def flush(self) -> Any:
        """Re-encode every stored recipe in id order as one batch encoding."""

        if not self._items:
            raise ValueError("cannot flush an empty CustomStream")
        from ... import batch_encodings

        encodings = []
        for key in sorted(self._items):
            recipe_state, recipe_kwargs = self._items[key]
            encodings.append(self.encoder.encode(recipe_state, **recipe_kwargs))
        result = batch_encodings(encodings)
        self.reset()
        return result

    def reset(self) -> None:
        """Clear every stored recipe."""

        self._items: dict[int, tuple[Any, dict[str, Any]]] = {}
        self._removed: set[int] = set()
        self._next_id = 0

    def _reset_builder(self) -> None:
        self.reset()


def _looks_like_state(value: object) -> bool:
    """Structural check covering pytyr states plus pymimir wrapper states."""

    if all(
        hasattr(value, name)
        for name in ("static_atoms", "fluent_facts", "derived_atoms")
    ):
        return True
    try:
        import pymimir
    except ImportError:
        return False
    return isinstance(value, pymimir.State)


__all__ = ["CustomGraphEncoder", "CustomStream"]
