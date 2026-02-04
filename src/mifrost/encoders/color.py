from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import networkx as nx
import torch
from torch_geometric.data import Batch, Data

from .._core import BatchBuilder, ColorEncoderConfig, ColorEncoderEngine, GoalInputs
from .base import EncoderBase, StreamEncoderBase
from .common import _advanced_domain, _advanced_state, _split_goals


def _parts_to_pyg_homo(
    parts: Mapping[str, Any],
    *,
    as_batch: bool | None = None,
    include_metadata: bool = True,
    undirected: bool = True,
) -> Data:
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


@dataclass
class ColorEncoderStream(StreamEncoderBase[Data]):
    _encoder: "ColorEncoder"

    def __post_init__(self) -> None:
        self._reset_builder()

    def append(
        self,
        state: Any,
        *,
        goals: Iterable[Any] | None = None,
        subgoal_layers: Iterable[Iterable[Any]] | None = None,
    ) -> None:
        adv_state = _advanced_state(state)
        if goals is None and subgoal_layers is None:
            self._encoder.engine.encode(adv_state, self._builder)
        else:
            if goals is None:
                if hasattr(state, "get_problem"):
                    goals = list(
                        state.get_problem().get_goal_condition().get_literals()
                    )
                else:
                    raise ValueError(
                        "goals must be provided when passing an advanced state"
                    )
            inputs, _ = _split_goals(goals, subgoal_layers)
            self._encoder.engine.encode(adv_state, inputs, self._builder)
        self._builder.next_graph()

    def _reset_builder(self) -> None:
        self._builder = BatchBuilder()
        self._builder.set_graph_kind("homo")

    def _parts_to_pyg(
        self,
        parts: Mapping[str, Any],
        *,
        as_batch: bool,
        include_metadata: bool = True,
    ) -> Data:
        return _parts_to_pyg_homo(
            parts, as_batch=as_batch, include_metadata=include_metadata
        )


class ColorEncoder(EncoderBase[Data]):
    def __init__(
        self,
        domain: Any,
        *,
        edge_features: bool = False,
        enable_global_predicate_nodes: bool = False,
    ) -> None:
        config = ColorEncoderConfig()
        config.edge_features = edge_features
        config.enable_global_predicate_nodes = enable_global_predicate_nodes
        self._engine = ColorEncoderEngine(_advanced_domain(domain), config)
        self.edge_features = edge_features
        self.predicate_nodes_enabled = enable_global_predicate_nodes

    @property
    def engine(self) -> ColorEncoderEngine:
        return self._engine

    def encode_parts(
        self,
        state: Any,
        *,
        goals: Iterable[Any] | None = None,
        actions: Iterable[Any] | None = None,
        subgoal_layers: Iterable[Iterable[Any]] | None = None,
    ) -> Mapping[str, Any]:
        adv_state = _advanced_state(state)
        if goals is None and subgoal_layers is None:
            return self._engine.encode(adv_state)
        if goals is None:
            if hasattr(state, "get_problem"):
                goals = list(state.get_problem().get_goal_condition().get_literals())
            else:
                raise ValueError(
                    "goals must be provided when passing an advanced state"
                )
        inputs, _ = _split_goals(goals, subgoal_layers)
        return self._engine.encode(adv_state, inputs)

    def encode(
        self,
        state: Any,
        *,
        goals: Iterable[Any] | None = None,
        subgoal_layers: Iterable[Iterable[Any]] | None = None,
        include_metadata: bool = True,
        **kwargs: Any,
    ) -> Data:
        return super().encode(
            state,
            goals=goals,
            subgoal_layers=subgoal_layers,
            include_metadata=include_metadata,
            **kwargs,
        )

    def encode_batch_parts(
        self,
        states: Iterable[Any] | Any,
        *,
        goals: Iterable[Any] | None = None,
        actions: Iterable[Any] | None = None,
        subgoal_layers: Iterable[Iterable[Any]] | None = None,
    ) -> Mapping[str, Any]:
        is_state_like = hasattr(states, "get_problem") or hasattr(
            states, "_advanced_state"
        )
        if is_state_like:
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
                    if hasattr(state, "get_problem"):
                        goals_for_state = list(
                            state.get_problem().get_goal_condition().get_literals()
                        )
                        inputs, _ = _split_goals(goals_for_state, subgoal_layers)
                    else:
                        raise ValueError(
                            "goals must be provided when passing an advanced state"
                        )
                else:
                    inputs = shared_inputs
                self._engine.encode(adv_state, inputs, builder)
            builder.next_graph()
        return builder.build_parts()

    def encode_batch(
        self,
        states: Iterable[Any] | Any,
        *,
        goals: Iterable[Any] | None = None,
        subgoal_layers: Iterable[Iterable[Any]] | None = None,
        include_metadata: bool = True,
        **kwargs: Any,
    ) -> Data:
        return super().encode_batch(
            states,
            goals=goals,
            subgoal_layers=subgoal_layers,
            include_metadata=include_metadata,
            **kwargs,
        )

    def _parts_to_pyg(
        self,
        parts: Mapping[str, Any],
        *,
        as_batch: bool,
        include_metadata: bool = True,
    ) -> Data:
        return _parts_to_pyg_homo(
            parts, as_batch=as_batch, include_metadata=include_metadata
        )

    def stream(self) -> ColorEncoderStream:
        return ColorEncoderStream(self)

    def to_networkx(self, data: Data) -> nx.Graph:
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
