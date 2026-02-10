from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import numpy as np
from torch_geometric.data import HeteroData

from .._core import BatchBuilder
from .base import EncoderBase, StreamEncoderBase
from .common import _encoding_dict_to_pyg


@dataclass
class ExampleConstantStreamEncoder(StreamEncoderBase[HeteroData]):
    """Minimal stream encoder example built fully in Python."""

    _default_value: float = 1.0

    def __post_init__(self) -> None:
        """Initialize an empty hetero builder for streaming."""
        self._reset_builder()

    def append(self, state: Any, *, value: float | None = None, **_: Any) -> int:
        """Append one single-node graph with a constant feature value."""
        stream_id = getattr(self, "_next_stream_id", 0)
        self._next_stream_id = stream_id + 1
        node_value = self._default_value if value is None else float(value)
        x = np.asarray([[node_value]], dtype=np.float32)
        self._builder.add_node_features("node", "x", x)
        self._builder.set_node_names("node", [str(state)])
        self._builder.set_object_names([str(state)])
        self._builder.next_graph()
        return stream_id

    def _reset_builder(self) -> None:
        """Reset stream accumulation state."""
        self._builder = BatchBuilder()
        self._builder.set_graph_kind("hetero")

    def _dict_to_pyg(
        self,
        encoding_dict: Mapping[str, Any] | object,
        *,
        as_batch: bool,
        include_metadata: bool = True,
    ) -> HeteroData:
        return _encoding_dict_to_pyg(
            encoding_dict, as_batch=as_batch, include_metadata=include_metadata
        )


class ExampleConstantEncoder(EncoderBase[HeteroData]):
    """
    Minimal pure-Python encoder example.

    This class reuses BatchBuilder and the shared EncoderBase API without any
    custom C++ engine code.
    """

    def __init__(self, *, default_value: float = 1.0) -> None:
        """Create the example encoder with a default scalar feature."""
        self._default_value = float(default_value)

    def _accepted_kwargs(self) -> set[str]:
        """Allow overriding the scalar value via ``value=...``."""
        return {"value"}

    def _encode(
        self,
        state: Any,
        *,
        goals: Iterable[Any] | None = None,
        actions: Iterable[Any] | None = None,
        subgoal_layers: Iterable[Iterable[Any]] | None = None,
        value: float | None = None,
        **_: Any,
    ) -> Mapping[str, Any]:
        """Encode one input into a one-node hetero graph payload."""
        del goals, actions, subgoal_layers
        node_value = self._default_value if value is None else float(value)
        builder = BatchBuilder()
        builder.set_graph_kind("hetero")
        x = np.asarray([[node_value]], dtype=np.float32)
        builder.add_node_features("node", "x", x)
        builder.set_node_names("node", [str(state)])
        builder.set_object_names([str(state)])
        return builder.build_batch_encoding()

    def _encode_batch(
        self,
        states: Iterable[Any] | Any,
        *,
        goals: Iterable[Any] | None = None,
        actions: Iterable[Any] | None = None,
        subgoal_layers: Iterable[Iterable[Any]] | None = None,
        value: float | None = None,
        **_: Any,
    ) -> Mapping[str, Any]:
        """Encode one or many inputs into one-node-per-sample payloads."""
        del goals, actions, subgoal_layers
        if hasattr(states, "__iter__") and not isinstance(states, (str, bytes)):
            state_list = list(states)
        else:
            state_list = [states]

        builder = BatchBuilder()
        builder.set_graph_kind("hetero")
        node_value = self._default_value if value is None else float(value)
        x = np.asarray([[node_value]], dtype=np.float32)
        for state in state_list:
            builder.add_node_features("node", "x", x)
            builder.set_node_names("node", [str(state)])
            builder.set_object_names([str(state)])
            builder.next_graph()
        return builder.build_batch_encoding()

    def stream(self) -> ExampleConstantStreamEncoder:
        """Create a streaming variant of this example encoder."""
        return ExampleConstantStreamEncoder(self._default_value)
