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


def _split_names_by_sizes(
    values: object | None,
    sizes: object | None,
) -> object | None:
    if not isinstance(values, list) or not torch.is_tensor(sizes):
        return values
    offsets = sizes.long().view(-1).tolist()
    if values and isinstance(values[0], (list, tuple)):
        if len(values) == len(offsets):
            return [[str(name) for name in entry] for entry in values]
        if len(values) == 1:
            values = list(values[0])
        else:
            return values
    split: list[list[str]] = []
    start = 0
    for size in offsets:
        stop = start + int(size)
        split.append([str(name) for name in values[start:stop]])
        start = stop
    return split


def _sizes_from_ptr(ptr: object | None) -> torch.Tensor | None:
    if not torch.is_tensor(ptr):
        return None
    ptr = ptr.long().view(-1)
    if ptr.numel() <= 1:
        return torch.zeros((0,), dtype=torch.long, device=ptr.device)
    return ptr[1:] - ptr[:-1]


def _normalize_shared_str_list(values: object | None) -> object | None:
    if values is None:
        return None
    if isinstance(values, list) and values and isinstance(values[0], (list, tuple)):
        first = [str(value) for value in values[0]]
        if all([str(value) for value in entry] == first for entry in values[1:]):
            return first
        return [[str(value) for value in entry] for entry in values]
    if isinstance(values, (list, tuple)):
        return [str(value) for value in values]
    return values


def _normalize_shared_scalar(value: object | None) -> object | None:
    if value is None or isinstance(value, str):
        return value
    if isinstance(value, list):
        if not value:
            return None
        first = value[0]
        if all(entry == first for entry in value[1:]):
            return first
    return value


def normalize_flat_relation_batch_metadata(
    data: FlatRelationData | Batch,
) -> FlatRelationData | Batch:
    num_graphs = int(getattr(data, "num_graphs", getattr(data, "_num_graphs", 1)))
    target_sizes = getattr(data, "target_sizes", None)
    if target_sizes is None:
        target_sizes = _sizes_from_ptr(getattr(data, "target_positions_ptr", None))
        if target_sizes is not None:
            data.target_sizes = target_sizes
    if num_graphs > 1:
        data.node_names = _split_names_by_sizes(
            getattr(data, "node_names", None),
            getattr(data, "node_sizes", None),
        )
        data.object_names = _split_names_by_sizes(
            getattr(data, "object_names", None),
            getattr(data, "object_sizes", None),
        )
        data.target_names = _split_names_by_sizes(
            getattr(data, "target_names", None),
            target_sizes,
        )
    data.target_groups = _normalize_shared_str_list(
        getattr(data, "target_groups", None)
    )
    data.target_entity_groups = _normalize_shared_str_list(
        getattr(data, "target_entity_groups", None)
    )
    data.target_symbol_prefix = _normalize_shared_scalar(
        getattr(data, "target_symbol_prefix", None)
    )
    data.lgan_tn_edge_pos = _normalize_shared_scalar(
        getattr(data, "lgan_tn_edge_pos", None)
    )
    data.lgan_nn_edge_pos = _normalize_shared_scalar(
        getattr(data, "lgan_nn_edge_pos", None)
    )
    data.lgan_rr_edge_pos = _normalize_shared_scalar(
        getattr(data, "lgan_rr_edge_pos", None)
    )
    for attr in (
        "target_positions_ptr",
        "target_indices_ptr",
        "target_candidate_ids_ptr",
        "target_depths_ptr",
        "target_group_ids_ptr",
    ):
        if hasattr(data, attr):
            delattr(data, attr)
    return data


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
        if key in {
            "relation_args",
            "object_indices",
            "history_entity_indices",
            "target_entity_indices",
            "target_positions",
            "lgan_tn_entity_indices",
            "lgan_nn_entity_indices",
        }:
            return int(getattr(self, "num_nodes", 0))
        if key in {
            "lgan_tn_relation_indices",
            "lgan_nn_relation_indices",
            "lgan_rr_src_relation_indices",
            "lgan_rr_dst_relation_indices",
        }:
            return self._relation_instance_offset()
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
        if graph_index is None and self.num_graphs > 1:
            counts = getattr(self, "relation_counts", None)
            if counts is None:
                return {
                    name: relation_args.new_empty((0, arity))
                    for name, arity in zip(self.schema.names, self.schema.arities)
                }
            counts = counts.long()
            arities = self.schema.arities
            chunks: dict[str, list[torch.Tensor]] = {
                name: [] for name in self.schema.names
            }
            start = 0
            for row in counts:
                for relation_idx, name in enumerate(self.schema.names):
                    arity = arities[relation_idx]
                    instances = int(row[relation_idx].item()) if row.numel() > 0 else 0
                    slots = instances * arity
                    chunk = relation_args[start : start + slots]
                    if arity > 0:
                        chunks[name].append(chunk.view(instances, arity))
                    else:
                        chunks[name].append(chunk.new_empty((instances, 0)))
                    start += slots
            out: dict[str, torch.Tensor] = {}
            for relation_name, arity in zip(self.schema.names, self.schema.arities):
                parts = chunks[relation_name]
                out[relation_name] = (
                    torch.cat(parts, dim=0)
                    if parts
                    else relation_args.new_empty((0, arity))
                )
            return out

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
        if node_names and isinstance(node_names[0], (list, tuple)):
            return [str(name) for name in node_names[graph_index]]
        return [str(name) for name in node_names]

    def graph_object_names(self, graph_index: int = 0) -> list[str]:
        object_names = getattr(self, "object_names", None)
        if object_names is None:
            return self.graph_node_names(graph_index)
        if object_names and isinstance(object_names[0], (list, tuple)):
            return [str(name) for name in object_names[graph_index]]
        return [str(name) for name in object_names]

    def graph_object_indices(self, graph_index: int = 0) -> torch.Tensor:
        return self._graph_cat_field_slice(
            field_name="object_indices",
            size_field_name="object_sizes",
            graph_index=graph_index,
        )

    def graph_history_entity_indices(self, graph_index: int = 0) -> torch.Tensor:
        return self._graph_cat_field_slice(
            field_name="history_entity_indices",
            size_field_name="history_entity_sizes",
            graph_index=graph_index,
        )

    def graph_history_entity_dt(self, graph_index: int = 0) -> torch.Tensor:
        return self._graph_cat_field_slice(
            field_name="history_entity_dt",
            size_field_name="history_entity_sizes",
            graph_index=graph_index,
        )

    def graph_history_entity_names(self, graph_index: int = 0) -> list[str]:
        history_entity_indices = self.graph_history_entity_indices(graph_index)
        if history_entity_indices.numel() == 0:
            return []
        start, _end = self.graph_node_range(graph_index)
        local_names = self.graph_node_names(graph_index)
        return [
            str(local_names[int(global_idx.item()) - start])
            for global_idx in history_entity_indices
        ]

    def graph_target_entity_indices(
        self, graph_index: int = 0, group: str | int | None = None
    ) -> torch.Tensor:
        indices = self._graph_cat_field_slice(
            field_name="target_entity_indices",
            size_field_name="target_entity_sizes",
            graph_index=graph_index,
        )
        if group is None or indices.numel() == 0:
            return indices
        group_ids = self.graph_target_entity_group_ids(graph_index)
        mask = self._group_mask(
            group_ids, group=group, attr_name="target_entity_groups"
        )
        return indices[mask]

    def graph_target_entity_group_ids(self, graph_index: int = 0) -> torch.Tensor:
        return self._graph_cat_field_slice(
            field_name="target_entity_group_ids",
            size_field_name="target_entity_sizes",
            graph_index=graph_index,
        )

    def graph_target_entity_names(
        self, graph_index: int = 0, group: str | int | None = None
    ) -> list[str]:
        target_entity_indices = self.graph_target_entity_indices(
            graph_index, group=group
        )
        if target_entity_indices.numel() == 0:
            return []
        start, _end = self.graph_node_range(graph_index)
        local_names = self.graph_node_names(graph_index)
        return [
            str(local_names[int(global_idx.item()) - start])
            for global_idx in target_entity_indices
        ]

    def graph_target_positions(self, graph_index: int = 0) -> torch.Tensor:
        return self._graph_cat_field_slice(
            field_name="target_positions",
            size_field_name="target_sizes",
            graph_index=graph_index,
        )

    def graph_target_indices(self, graph_index: int = 0) -> torch.Tensor:
        return self._graph_cat_field_slice(
            field_name="target_indices",
            size_field_name="target_sizes",
            graph_index=graph_index,
        )

    def graph_target_candidate_ids(self, graph_index: int = 0) -> torch.Tensor:
        return self._graph_cat_field_slice(
            field_name="target_candidate_ids",
            size_field_name="target_sizes",
            graph_index=graph_index,
        )

    def graph_target_depths(self, graph_index: int = 0) -> torch.Tensor:
        return self._graph_cat_field_slice(
            field_name="target_depths",
            size_field_name="target_sizes",
            graph_index=graph_index,
        )

    def graph_target_group_ids(self, graph_index: int = 0) -> torch.Tensor:
        return self._graph_cat_field_slice(
            field_name="target_group_ids",
            size_field_name="target_sizes",
            graph_index=graph_index,
        )

    def graph_target_names(self, graph_index: int = 0) -> list[str]:
        target_names = getattr(self, "target_names", None)
        if target_names is None:
            return []
        if target_names and isinstance(target_names[0], (list, tuple)):
            return [str(name) for name in target_names[graph_index]]
        return [str(name) for name in target_names]

    def graph_relation_instance_range(self, graph_index: int = 0) -> tuple[int, int]:
        relation_instance_sizes = getattr(self, "relation_instance_sizes", None)
        if relation_instance_sizes is None:
            counts = getattr(self, "relation_counts", None)
            if counts is None:
                return (0, 0)
            counts = counts.long()
            if counts.dim() == 1:
                total = int(counts.sum().item())
                return (0, total)
            if graph_index < 0 or graph_index >= counts.size(0):
                raise IndexError(
                    f"graph_index {graph_index} out of range for {counts.size(0)} graphs"
                )
            sizes = counts.sum(dim=1)
        else:
            sizes = relation_instance_sizes.long().view(-1)
            if graph_index < 0 or graph_index >= len(sizes):
                raise IndexError(
                    f"graph_index {graph_index} out of range for {len(sizes)} graphs"
                )
        start = int(sizes[:graph_index].sum().item()) if graph_index > 0 else 0
        end = start + int(sizes[graph_index].item())
        return start, end

    def graph_lgan_tn_edges(self, graph_index: int = 0) -> torch.Tensor:
        return self._graph_edge_slice(
            src_field_name="lgan_tn_relation_indices",
            dst_field_name="lgan_tn_entity_indices",
            size_field_name="lgan_tn_sizes",
            graph_index=graph_index,
        )

    def graph_lgan_nn_edges(self, graph_index: int = 0) -> torch.Tensor:
        return self._graph_edge_slice(
            src_field_name="lgan_nn_relation_indices",
            dst_field_name="lgan_nn_entity_indices",
            size_field_name="lgan_nn_sizes",
            graph_index=graph_index,
        )

    def graph_lgan_rr_edges(self, graph_index: int = 0) -> torch.Tensor:
        return self._graph_edge_slice(
            src_field_name="lgan_rr_src_relation_indices",
            dst_field_name="lgan_rr_dst_relation_indices",
            size_field_name="lgan_rr_sizes",
            graph_index=graph_index,
        )

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

    def _graph_cat_field_slice(
        self,
        *,
        field_name: str,
        size_field_name: str,
        graph_index: int,
    ) -> torch.Tensor:
        values = getattr(self, field_name, None)
        if values is None:
            return torch.empty((0,), dtype=torch.long)
        values = values.long().view(-1)
        sizes = getattr(self, size_field_name, None)
        if sizes is None:
            return values
        sizes = sizes.long().view(-1)
        if graph_index < 0 or graph_index >= len(sizes):
            raise IndexError(
                f"graph_index {graph_index} out of range for {len(sizes)} graphs"
            )
        start = int(sizes[:graph_index].sum().item()) if graph_index > 0 else 0
        end = start + int(sizes[graph_index].item())
        return values[start:end]

    def _graph_edge_slice(
        self,
        *,
        src_field_name: str,
        dst_field_name: str,
        size_field_name: str,
        graph_index: int,
    ) -> torch.Tensor:
        src = self._graph_cat_field_slice(
            field_name=src_field_name,
            size_field_name=size_field_name,
            graph_index=graph_index,
        )
        dst = self._graph_cat_field_slice(
            field_name=dst_field_name,
            size_field_name=size_field_name,
            graph_index=graph_index,
        )
        if src.numel() == 0 or dst.numel() == 0:
            device = src.device if src.numel() else dst.device
            return torch.empty((2, 0), dtype=torch.long, device=device)
        return torch.stack((src, dst), dim=0)

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

    def _group_mask(
        self,
        group_ids: torch.Tensor,
        *,
        group: str | int,
        attr_name: str,
    ) -> torch.Tensor:
        target_group_id = self._group_id(group=group, attr_name=attr_name)
        return group_ids.long().view(-1) == target_group_id

    def _group_id(self, *, group: str | int, attr_name: str) -> int:
        if isinstance(group, int):
            return group
        group_names = getattr(self, attr_name, None)
        if group_names is None:
            raise ValueError(f"{attr_name} metadata is not available")
        names = [str(value) for value in group_names]
        if group not in names:
            raise ValueError(
                f"Unknown {attr_name} group {group!r}; expected one of {names!r}"
            )
        return names.index(group)

    def _relation_instance_offset(self) -> int:
        relation_instance_sizes = getattr(self, "relation_instance_sizes", None)
        if relation_instance_sizes is None:
            return 0
        if torch.is_tensor(relation_instance_sizes):
            return int(relation_instance_sizes.long().sum().item())
        return int(sum(int(value) for value in relation_instance_sizes))


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
    return normalize_flat_relation_batch_metadata(out)


__all__ = [
    "FlatRelationData",
    "FlatRelationSchema",
    "flat_relation_data_from_pyg",
    "normalize_flat_relation_batch_metadata",
]
