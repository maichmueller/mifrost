from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import networkx as nx
import torch
from torch_geometric.data import Batch, Data

from .._core import (
    BatchBuilder,
    ColorEncoderConfig,
    ColorEncoderEngine,
    ColorStreamEncoder as _ColorStreamEncoder,
    GoalInputs,
)
from .base import (
    ActionBatchInput,
    EncoderBase,
    GoalBatchInput,
    StateBatchInput,
    StreamEncoderBase,
    SubgoalLayersInput,
)
from .common import _advanced_domain, _advanced_state, _split_goals
from .types import (
    STATE_TYPES,
    DomainInput,
    GoalLiteralInput,
    StateInput,
    default_goals_from_state,
)


def _parts_to_pyg_homo(
    parts: Mapping[str, Any],
    *,
    as_batch: bool | None = None,
    include_metadata: bool = True,
    undirected: bool = True,
) -> Data:
    """
    Convert color-encoder parts into homogeneous PyG ``Data``/``Batch``.

    This adapter expects the standard parts schema and uses a single node type.
    """
    raw_tensors: Mapping[str, Any] = parts.get("tensors", {})
    schema_obj = parts.get("schema")
    if schema_obj is None:
        raise ValueError("parts schema missing; rebuild the extension to emit schema")
    if hasattr(schema_obj, "to_dict"):
        schema_obj = schema_obj.to_dict()
    if not isinstance(schema_obj, Mapping):
        raise TypeError(f"parts schema must be a mapping, got {type(schema_obj)}")
    schema: Mapping[str, Any] = schema_obj

    node_names_map: Mapping[str, list[str]] = parts.get("node_names", {})
    num_graphs = int(parts.get("num_graphs", 0))

    if as_batch is None:
        as_batch = num_graphs > 1

    data: Data
    if as_batch:
        data = Batch(_base_cls=Data)
    else:
        data = Data()

    tensors: dict[str, Any] = {}
    tensors_torch: dict[str, torch.Tensor] = {}
    for key_obj, value in raw_tensors.items():
        key = key_obj if isinstance(key_obj, str) else str(key_obj)
        tensors[key] = value

    def get_tensor(key: str, value: Any | None = None) -> torch.Tensor:
        cached = tensors_torch.get(key)
        if cached is not None:
            return cached
        if value is None:
            value = tensors[key]
        tensor = torch.as_tensor(value)
        tensors_torch[key] = tensor
        return tensor

    node_types = schema.get("node_types", [])
    node_type = node_types[0] if node_types else "node"

    for entry in schema.get("node_tensors", []):
        key = entry["key"]
        if key not in tensors:
            raise KeyError(f"Schema references missing tensor key: {key}")
        if entry["node_type"] != node_type:
            continue
        attr = entry["attr"]
        data[attr] = get_tensor(key, tensors[key])

    edge_parts: dict[str, torch.Tensor] = {}
    edge_attr: torch.Tensor | None = None
    for entry in schema.get("edge_tensors", []):
        key = entry["key"]
        if key not in tensors:
            raise KeyError(f"Schema references missing tensor key: {key}")
        attr = entry["attr"]
        if attr == "edge_index":
            part = str(entry.get("part", ""))
            if part == "":
                raise ValueError(f"Missing edge_index part for key: {key}")
            edge_parts[part] = get_tensor(key, tensors[key])
        elif attr == "edge_attr":
            edge_attr = get_tensor(key, tensors[key])

    if "0" in edge_parts and "1" in edge_parts:
        edge_index = torch.stack((edge_parts["0"], edge_parts["1"]), dim=0)
        if undirected and edge_index.numel() > 0:
            src = edge_index[0]
            dst = edge_index[1]
            mask = src != dst
            rev = torch.stack((dst[mask], src[mask]), dim=0)
            edge_index = torch.cat((edge_index, rev), dim=1)
            if edge_attr is not None:
                edge_attr = torch.cat((edge_attr, edge_attr[mask]), dim=0)
        data.edge_index = edge_index

    if edge_attr is not None:
        data.edge_attr = edge_attr

    if include_metadata:
        names = node_names_map.get(node_type, [])
        data.node_names = names if isinstance(names, list) else list(names)

    if as_batch and num_graphs > 0:
        data._num_graphs = num_graphs
        batch_key = f"{node_type}/batch"
        if batch_key in tensors:
            data.batch = get_tensor(batch_key, tensors[batch_key]).long()

    if getattr(data, "x", None) is None:
        if hasattr(data, "num_nodes") and data.num_nodes:
            pass
        elif hasattr(data, "node_names"):
            data.num_nodes = len(data.node_names)

    return data


def _add_color_extension(parts: Mapping[str, Any]) -> None:
    """Ensure the schema extensions include color encoding metadata."""
    schema_obj = parts.get("schema")
    if schema_obj is None:
        return
    if hasattr(schema_obj, "to_dict"):
        schema_obj = schema_obj.to_dict()
    if not isinstance(schema_obj, Mapping):
        return
    schema = dict(schema_obj)
    flags = schema.get("flags", {})
    edge_features = bool(flags.get("edge_features", False))
    extensions = dict(schema.get("extensions", {}))
    extensions["color_encoding"] = "edge" if edge_features else "node"
    schema["extensions"] = extensions
    parts["schema"] = schema


@dataclass
class ColorEncoderStream(StreamEncoderBase[Data]):
    """Streaming wrapper for ``ColorEncoder``."""

    _encoder: "ColorEncoder"

    def __post_init__(self) -> None:
        """Initialize an empty homo builder for streaming."""
        self._stream = _ColorStreamEncoder(self._encoder.engine)
        self._reset_builder()

    def append(
        self,
        state: StateInput,
        *,
        goals: Iterable[GoalLiteralInput] | None = None,
        subgoal_layers: Iterable[Iterable[GoalLiteralInput]] | None = None,
    ) -> int:
        """Append one state encoding to the color stream."""
        adv_state = _advanced_state(state)
        if goals is None and subgoal_layers is None:
            return self._coerce_stream_id(self._stream.append(adv_state))
        else:
            if goals is None:
                goals = default_goals_from_state(state)
            inputs, _ = _split_goals(goals, subgoal_layers)
            return self._coerce_stream_id(self._stream.append(adv_state, inputs))

    def remove(self, stream_id: int) -> None:
        self._stream.remove(stream_id)

    def update(
        self,
        stream_id: int,
        state: StateInput,
        *,
        goals: Iterable[GoalLiteralInput] | None = None,
        subgoal_layers: Iterable[Iterable[GoalLiteralInput]] | None = None,
    ) -> None:
        adv_state = _advanced_state(state)
        if goals is None and subgoal_layers is None:
            self._stream.update(stream_id, adv_state)
            return
        if goals is None:
            goals = default_goals_from_state(state)
        inputs, _ = _split_goals(goals, subgoal_layers)
        self._stream.update(stream_id, adv_state, inputs)

    def _reset_builder(self) -> None:
        """Reset stream accumulation state."""
        self._stream.reset()

    def _flush_batch_encoding_py_impl(self) -> Mapping[str, object]:
        parts = self._stream.flush_batch_encoding_py()
        _add_color_extension(parts)
        return parts

    def _parts_to_pyg(
        self,
        parts: Mapping[str, object],
        *,
        as_batch: bool,
        include_metadata: bool = True,
    ) -> Data:
        return _parts_to_pyg_homo(
            parts, as_batch=as_batch, include_metadata=include_metadata
        )


class ColorEncoder(EncoderBase[Data]):
    """
    Homogeneous color encoder backed by ``ColorEncoderEngine``.

    Produces compact ``Data``/``Batch`` outputs with integer-like node/edge
    attributes suitable for color-based graph models.
    """

    def __init__(
        self,
        domain: DomainInput,
        *,
        edge_features: bool = False,
        enable_global_predicate_nodes: bool = False,
    ) -> None:
        """Create a color encoder for one domain."""
        config = ColorEncoderConfig()
        config.edge_features = edge_features
        config.enable_global_predicate_nodes = enable_global_predicate_nodes
        self._engine = ColorEncoderEngine(_advanced_domain(domain), config)
        self.edge_features = edge_features
        self.predicate_nodes_enabled = enable_global_predicate_nodes

    @property
    def engine(self) -> ColorEncoderEngine:
        """Expose the underlying C++ color engine."""
        return self._engine

    def encode_parts(
        self,
        state: StateInput,
        *,
        goals: GoalBatchInput = None,
        actions: ActionBatchInput = None,
        subgoal_layers: SubgoalLayersInput = None,
    ) -> Mapping[str, object]:
        """Encode one state into homogeneous parts."""
        adv_state = _advanced_state(state)
        if goals is None and subgoal_layers is None:
            return self._engine.encode(adv_state)
        if goals is None:
            goals = default_goals_from_state(state)
        inputs, _ = _split_goals(goals, subgoal_layers)
        return self._engine.encode(adv_state, inputs)

    def encode(
        self,
        state: StateInput,
        *,
        goals: GoalBatchInput = None,
        subgoal_layers: SubgoalLayersInput = None,
        include_metadata: bool = True,
        **kwargs: object,
    ) -> Data:
        """Encode one state into ``Data``."""
        return super().encode(
            state,
            goals=goals,
            subgoal_layers=subgoal_layers,
            include_metadata=include_metadata,
            **kwargs,
        )

    def encode_batch_parts(
        self,
        states: StateBatchInput,
        *,
        goals: GoalBatchInput = None,
        actions: ActionBatchInput = None,
        subgoal_layers: SubgoalLayersInput = None,
    ) -> Mapping[str, object]:
        """Encode one or many states into homogeneous batch parts."""
        if isinstance(states, STATE_TYPES):
            state_list = [states]
        else:
            if isinstance(states, (str, bytes)):
                raise TypeError("encode_batch expects a state or an iterable of states")
            state_list = list(states)

        builder = BatchBuilder()
        builder.set_graph_kind("homo")
        shared_inputs: GoalInputs | None = None
        if goals is not None:
            shared_inputs, _ = _split_goals(goals, subgoal_layers)

        for state in state_list:
            adv_state = _advanced_state(state)
            if goals is None and subgoal_layers is None:
                self._engine.encode(adv_state, builder)
            else:
                if goals is None:
                    goals_for_state = default_goals_from_state(state)
                    inputs, _ = _split_goals(goals_for_state, subgoal_layers)
                else:
                    inputs = shared_inputs
                self._engine.encode(adv_state, inputs, builder)
            builder.next_graph()
        return builder.build_batch_encoding_py()

    def encode_batch(
        self,
        states: StateBatchInput,
        *,
        goals: GoalBatchInput = None,
        subgoal_layers: SubgoalLayersInput = None,
        include_metadata: bool = True,
        **kwargs: object,
    ) -> Data:
        """Encode one or many states into batched ``Data``."""
        return super().encode_batch(
            states,
            goals=goals,
            subgoal_layers=subgoal_layers,
            include_metadata=include_metadata,
            **kwargs,
        )

    def _parts_to_pyg(
        self,
        parts: Mapping[str, object],
        *,
        as_batch: bool,
        include_metadata: bool = True,
    ) -> Data:
        return _parts_to_pyg_homo(
            parts, as_batch=as_batch, include_metadata=include_metadata
        )

    def stream(self) -> ColorEncoderStream:
        """Create a streaming encoder sharing this encoder's C++ engine."""
        return ColorEncoderStream(self)

    def to_networkx(self, data: Data) -> nx.Graph:
        """Convert a color-encoded PyG graph into a NetworkX graph."""
        graph = nx.Graph()
        node_names = getattr(data, "node_names", None)
        if not node_names:
            if hasattr(data, "x") and data.x is not None:
                count = data.x.shape[0]
            else:
                count = getattr(data, "num_nodes", 0) or 0
            node_names = [str(i) for i in range(count)]

        for i, name in enumerate(node_names):
            val = data.x[i] if hasattr(data, "x") and data.x is not None else 0
            attrs = {"type": val.item() if torch.is_tensor(val) else val}
            if hasattr(data, "goal_level") and data.goal_level is not None:
                gval = data.goal_level[i]
                attrs["goal_level"] = gval.item() if torch.is_tensor(gval) else gval
            graph.add_node(name, **attrs)

        if hasattr(data, "edge_index"):
            for i in range(data.edge_index.shape[1]):
                u_idx = data.edge_index[0, i].item()
                v_idx = data.edge_index[1, i].item()
                u_name = node_names[u_idx]
                v_name = node_names[v_idx]
                attrs = {}
                if hasattr(data, "edge_attr") and data.edge_attr is not None:
                    val = data.edge_attr[i]
                    attrs["type"] = (
                        int(val.item()) if torch.is_tensor(val) else int(val)
                    )
                graph.add_edge(u_name, v_name, **attrs)

        graph.graph["encoder_hash"] = getattr(data, "encoder_hash", None)
        return graph


__all__ = ["ColorEncoder", "ColorEncoderStream"]
