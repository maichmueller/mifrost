from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Any, Sequence

import torch
from torch_geometric.data import Batch, Data


def _normalize_str_tuple(values: object | None) -> tuple[str, ...]:
    if values is None:
        return ()
    if isinstance(values, list) and values and isinstance(values[0], (list, tuple)):
        first = tuple(str(value) for value in values[0])
        if all(tuple(str(v) for v in entry) == first for entry in values[1:]):
            return first
    if isinstance(values, tuple):
        return tuple(str(value) for value in values)
    if isinstance(values, str):
        return (values,)
    return tuple(str(value) for value in values)


def _normalize_int_tuple(values: object | None) -> tuple[int, ...]:
    if values is None:
        return ()
    if torch.is_tensor(values):
        return tuple(int(v) for v in values.view(-1).tolist())
    if isinstance(values, list) and values and isinstance(values[0], (list, tuple)):
        first = tuple(int(value) for value in values[0])
        if all(tuple(int(v) for v in entry) == first for entry in values[1:]):
            return first
    return tuple(int(value) for value in values)


def _copy_data_attrs(dst: Data, src: Data) -> None:
    for key in src.keys():
        dst[key] = src[key]
    if hasattr(src, "_num_graphs"):
        dst._num_graphs = int(src._num_graphs)


def _normalize_optional_int(value: object | None) -> int | None:
    if value is None:
        return None
    if torch.is_tensor(value):
        flat = value.view(-1).tolist()
        if not flat:
            return None
        first = int(flat[0])
        if all(int(entry) == first for entry in flat[1:]):
            return first
        raise ValueError(
            "FlatRelationData schema_fingerprint must be constant across the batch"
        )
    if isinstance(value, list):
        if not value:
            return None
        first = int(value[0])
        if all(int(entry) == first for entry in value[1:]):
            return first
        raise ValueError(
            "FlatRelationData schema_fingerprint must be constant across the batch"
        )
    return int(value)


@dataclass(frozen=True)
class FlatRelationSchema:
    names: tuple[str, ...]
    arities: tuple[int, ...]
    sources: tuple[str, ...] = ()
    fingerprint: int | None = None

    @cached_property
    def name_to_id(self) -> dict[str, int]:
        return {name: idx for idx, name in enumerate(self.names)}


class FlatRelationData(Data):
    """Packed flat relation carrier with PyG-compatible batching behavior."""

    def __inc__(self, key: str, value: Any, *args, **kwargs) -> Any:
        if key in {"relation_args", "object_indices", "action_indices"}:
            return int(getattr(self, "num_nodes", 0))
        return super().__inc__(key, value, *args, **kwargs)

    @cached_property
    def schema(self) -> FlatRelationSchema:
        return FlatRelationSchema(
            names=_normalize_str_tuple(getattr(self, "relation_names", ()) or ()),
            arities=_normalize_int_tuple(getattr(self, "relation_arities", ()) or ()),
            sources=_normalize_str_tuple(getattr(self, "relation_sources", ()) or ()),
            fingerprint=_normalize_optional_int(
                getattr(self, "schema_fingerprint", None)
            ),
        )

    @property
    def flattened_relations(self) -> dict[str, torch.Tensor]:
        return self.flattened_relations_view()

    def relation_instance_counts_total(self) -> torch.Tensor:
        counts = getattr(self, "relation_counts", None)
        if counts is None:
            return torch.zeros((len(self.schema.names),), dtype=torch.long)
        counts = counts.long()
        if counts.dim() == 1:
            return counts
        return counts.sum(dim=0)

    def relation_slot_offsets(self, graph_index: int | None = None) -> torch.Tensor:
        counts = self._relation_counts_for(graph_index)
        arities = self._relation_arities_tensor(counts.device)
        slot_counts = counts * arities
        prefix = torch.zeros(
            (slot_counts.numel() + 1,), dtype=torch.long, device=slot_counts.device
        )
        if slot_counts.numel() > 0:
            prefix[1:] = torch.cumsum(slot_counts, dim=0)
        return prefix

    def flattened_relations_view(
        self, graph_index: int | None = None
    ) -> dict[str, torch.Tensor]:
        relation_args = getattr(self, "relation_args", None)
        if relation_args is None:
            return {
                name: torch.empty((0, arity), dtype=torch.long)
                for name, arity in zip(self.schema.names, self.schema.arities)
            }

        relation_args = relation_args.long().view(-1)
        counts = self._relation_counts_for(graph_index)
        arities = self.schema.arities
        start = self._relation_arg_start(graph_index)
        out: dict[str, torch.Tensor] = {}
        for relation_idx, name in enumerate(self.schema.names):
            arity = arities[relation_idx]
            instances = int(counts[relation_idx].item()) if counts.numel() > 0 else 0
            slots = instances * arity
            chunk = relation_args[start : start + slots]
            if arity > 0:
                out[name] = chunk.view(instances, arity)
            else:
                out[name] = chunk.new_empty((instances, 0))
            start += slots
        return out

    def graph_node_names(self, graph_index: int = 0) -> list[str]:
        node_names = getattr(self, "node_names", None)
        if node_names is None:
            start, end = self.graph_node_range(graph_index)
            return [f"entity:{idx}" for idx in range(start, end)]
        if (
            self.num_graphs > 1
            and node_names
            and isinstance(node_names[0], (list, tuple))
        ):
            return [str(name) for name in node_names[graph_index]]
        return [str(name) for name in node_names]

    def graph_object_names(self, graph_index: int = 0) -> list[str]:
        object_names = getattr(self, "object_names", None)
        if object_names is None:
            return self.graph_node_names(graph_index)
        if (
            self.num_graphs > 1
            and object_names
            and isinstance(object_names[0], (list, tuple))
        ):
            return [str(name) for name in object_names[graph_index]]
        return [str(name) for name in object_names]

    def graph_node_range(self, graph_index: int = 0) -> tuple[int, int]:
        node_sizes = getattr(self, "node_sizes", None)
        if node_sizes is None:
            return (0, int(getattr(self, "num_nodes", 0)))
        node_sizes = node_sizes.long().view(-1)
        if graph_index < 0 or graph_index >= len(node_sizes):
            raise IndexError(
                f"graph_index {graph_index} out of range for {len(node_sizes)} graphs"
            )
        start = int(node_sizes[:graph_index].sum().item()) if graph_index > 0 else 0
        end = start + int(node_sizes[graph_index].item())
        return start, end

    @property
    def num_graphs(self) -> int:
        if hasattr(self, "_num_graphs"):
            return int(self._num_graphs)
        node_sizes = getattr(self, "node_sizes", None)
        if node_sizes is not None and torch.is_tensor(node_sizes):
            return int(node_sizes.view(-1).numel())
        return 1

    def _relation_counts_for(self, graph_index: int | None) -> torch.Tensor:
        counts = getattr(self, "relation_counts", None)
        if counts is None:
            return torch.zeros((len(self.schema.names),), dtype=torch.long)
        counts = counts.long()
        if counts.dim() == 1:
            return counts
        if graph_index is None:
            return counts.sum(dim=0)
        if graph_index < 0 or graph_index >= counts.size(0):
            raise IndexError(
                f"graph_index {graph_index} out of range for {counts.size(0)} graphs"
            )
        return counts[graph_index]

    def _relation_arities_tensor(
        self, device: torch.device | None = None
    ) -> torch.Tensor:
        return torch.tensor(
            list(self.schema.arities),
            dtype=torch.long,
            device=device or torch.device("cpu"),
        )

    def _relation_arg_start(self, graph_index: int | None) -> int:
        if graph_index is None or self.num_graphs <= 1:
            return 0
        counts = getattr(self, "relation_counts", None)
        if counts is None:
            return 0
        counts = counts.long()
        if graph_index <= 0:
            return 0
        arities = self._relation_arities_tensor(counts.device)
        prior_counts = counts[:graph_index].sum(dim=0)
        return int((prior_counts * arities).sum().item())


def flat_relation_data_from_pyg(
    data: Data,
    *,
    schema_fingerprint: int | None = None,
) -> FlatRelationData | Batch:
    if isinstance(data, Batch):
        out = Batch(_base_cls=FlatRelationData)
    else:
        out = FlatRelationData()
    _copy_data_attrs(out, data)

    if hasattr(out, "__dict__"):
        out.__dict__.pop("schema", None)

    relation_names = getattr(out, "relation_names", None)
    if relation_names is not None:
        out.relation_names = _normalize_str_tuple(relation_names)
    relation_sources = getattr(out, "relation_sources", None)
    if relation_sources is not None:
        out.relation_sources = _normalize_str_tuple(relation_sources)
    relation_arities = getattr(out, "relation_arities", None)
    if relation_arities is not None:
        out.relation_arities = _normalize_int_tuple(relation_arities)
    if schema_fingerprint is not None:
        out.schema_fingerprint = str(int(schema_fingerprint))
    return out


__all__ = ["FlatRelationData", "FlatRelationSchema", "flat_relation_data_from_pyg"]
