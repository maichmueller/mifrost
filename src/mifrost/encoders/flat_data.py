from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Any, cast

import torch
from torch_geometric.data import Batch, Data

_RELATION_ARGS_GRAPH_MAJOR = "graph_major"
_RELATION_ARGS_RELATION_MAJOR = "relation_major"


def _normalize_str_tuple(values: Any | None) -> tuple[str, ...]:
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


def _normalize_int_tuple(values: Any | None) -> tuple[int, ...]:
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


def _normalize_optional_int(value: Any | None) -> int | None:
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
    values: Any | None,
    sizes: Any | None,
) -> Any | None:
    if not isinstance(values, list) or not torch.is_tensor(sizes):
        return values
    sizes = cast(torch.Tensor, sizes)
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


def _sizes_from_ptr(ptr: Any | None) -> torch.Tensor | None:
    if not torch.is_tensor(ptr):
        return None
    ptr = cast(torch.Tensor, ptr)
    ptr = ptr.long().view(-1)
    if ptr.numel() <= 1:
        return torch.zeros((0,), dtype=torch.long, device=ptr.device)
    return ptr[1:] - ptr[:-1]


def _normalize_shared_str_list(values: Any | None) -> Any | None:
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


def _normalize_shared_scalar(value: Any | None) -> Any | None:
    if value is None or isinstance(value, str):
        return value
    if isinstance(value, list):
        if not value:
            return None
        first = value[0]
        if all(entry == first for entry in value[1:]):
            return first
    return value


def _normalize_shared_bool(value: Any | None) -> bool | None:
    if value is None:
        return None
    if torch.is_tensor(value):
        flat = [bool(entry) for entry in value.view(-1).tolist()]
        if not flat:
            return None
        first = flat[0]
        if all(entry == first for entry in flat[1:]):
            return first
        raise ValueError(
            "FlatRelationData include_lgan_edges must be constant across the batch"
        )
    if isinstance(value, list):
        if not value:
            return None
        flat = [bool(entry) for entry in value]
        first = flat[0]
        if all(entry == first for entry in flat[1:]):
            return first
        raise ValueError(
            "FlatRelationData include_lgan_edges must be constant across the batch"
        )
    return bool(value)


def normalize_flat_relation_batch_metadata(
    data: FlatRelationData | Batch,
) -> FlatRelationData | Batch:
    """Normalize flat metadata after native or PyG batching.

    This makes shared schema labels and per-graph name lists easier to use from
    Python. It does not change the stored flat relation tensors.
    """
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
    data.target_sources = _normalize_shared_str_list(
        getattr(data, "target_sources", None)
    )
    data.target_entity_groups = _normalize_shared_str_list(
        getattr(data, "target_entity_groups", None)
    )
    data.entity_role_names = _normalize_shared_str_list(
        getattr(data, "entity_role_names", None)
    )
    data.lgan_anchor_sources = _normalize_shared_str_list(
        getattr(data, "lgan_anchor_sources", None)
    )
    data.include_lgan_edges = _normalize_shared_bool(
        getattr(data, "include_lgan_edges", None)
    )
    data.use_predicate_virtual_nodes = _normalize_shared_bool(
        getattr(data, "use_predicate_virtual_nodes", None)
    )
    data.entity_node_type = _normalize_shared_scalar(
        getattr(data, "entity_node_type", None)
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
    data.relation_args_layout = _normalize_shared_scalar(
        getattr(data, "relation_args_layout", None)
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
    """Small immutable view of the flat relation schema."""

    names: tuple[str, ...]
    arities: tuple[int, ...]
    logical_arities: tuple[int, ...] = ()
    encoded_arities: tuple[int, ...] = ()
    sources: tuple[str, ...] = ()
    slot_role_names: tuple[str, ...] = ()
    slot_role_ids: tuple[int, ...] = ()
    slot_role_offsets: tuple[int, ...] = ()
    fingerprint: int | None = None

    @cached_property
    def name_to_id(self) -> dict[str, int]:
        """Map each relation name to its fixed schema index."""
        return {name: idx for idx, name in enumerate(self.names)}

    @cached_property
    def slot_roles(self) -> tuple[tuple[str, ...], ...]:
        """Decode per-relation slot-role labels from flattened schema metadata."""
        if not self.slot_role_ids or not self.slot_role_offsets:
            return tuple(tuple() for _ in self.names)
        decoded: list[tuple[str, ...]] = []
        for relation_idx in range(len(self.names)):
            start = self.slot_role_offsets[relation_idx]
            end = self.slot_role_offsets[relation_idx + 1]
            decoded.append(
                tuple(
                    self.slot_role_names[role_id]
                    if 0 <= role_id < len(self.slot_role_names)
                    else str(role_id)
                    for role_id in self.slot_role_ids[start:end]
                )
            )
        return tuple(decoded)


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
        """Return the normalized flat relation schema for this batch."""
        return FlatRelationSchema(
            names=_normalize_str_tuple(getattr(self, "relation_names", ()) or ()),
            arities=_normalize_int_tuple(getattr(self, "relation_arities", ()) or ()),
            logical_arities=_normalize_int_tuple(
                getattr(self, "relation_logical_arities", None)
                or getattr(self, "relation_arities", ())
            ),
            encoded_arities=_normalize_int_tuple(
                getattr(self, "relation_encoded_arities", None)
                or getattr(self, "relation_arities", ())
            ),
            sources=_normalize_str_tuple(getattr(self, "relation_sources", ()) or ()),
            slot_role_names=_normalize_str_tuple(
                getattr(self, "slot_role_names", ()) or ()
            ),
            slot_role_ids=_normalize_int_tuple(
                getattr(self, "relation_slot_roles", ()) or ()
            ),
            slot_role_offsets=_normalize_int_tuple(
                getattr(self, "relation_slot_role_offsets", ()) or ()
            ),
            fingerprint=_normalize_optional_int(
                getattr(self, "schema_fingerprint", None)
            ),
        )

    @property
    def flattened_relations(self) -> dict[str, torch.Tensor]:
        """Return the flat relations grouped by relation name."""
        return self.flattened_relations_view()

    def relation_instance_counts_total(self) -> torch.Tensor:
        """Return total instance counts per relation across the whole batch."""
        counts = getattr(self, "relation_counts", None)
        if counts is None:
            return torch.zeros((len(self.schema.names),), dtype=torch.long)
        counts = counts.long()
        if counts.dim() == 1:
            return counts
        return counts.sum(dim=0)

    def relation_slot_offsets(self, graph_index: int | None = None) -> torch.Tensor:
        """Return slot offsets into `relation_args` for one graph or the whole batch."""
        if self._uses_relation_major_args() and graph_index is not None:
            raise ValueError(
                "relation_slot_offsets(graph_index=...) is not representable for "
                "relation-major relation_args because one graph is not contiguous"
            )
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
        """Slice `relation_args` into per-relation matrices.

        With `graph_index=None`, the result covers the whole batch. With a graph
        index, the result is limited to one graph.
        """
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
            if self._uses_relation_major_args():
                relation_major_graph_out: dict[str, torch.Tensor] = {}
                counts_matrix = self._relation_counts_matrix(counts.device)
                start = 0
                for relation_idx, relation_name in enumerate(self.schema.names):
                    arity = arities[relation_idx]
                    instances = int(counts_matrix[:, relation_idx].sum().item())
                    slots = instances * arity
                    chunk = relation_args[start : start + slots]
                    relation_major_graph_out[relation_name] = (
                        chunk.view(instances, arity)
                        if arity > 0
                        else chunk.new_empty((instances, 0))
                    )
                    start += slots
                return relation_major_graph_out
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
            graph_major_graph_out: dict[str, torch.Tensor] = {}
            for relation_name, arity in zip(self.schema.names, self.schema.arities):
                parts = chunks[relation_name]
                graph_major_graph_out[relation_name] = (
                    torch.cat(parts, dim=0)
                    if parts
                    else relation_args.new_empty((0, arity))
                )
            return graph_major_graph_out

        counts = self._relation_counts_for(graph_index)
        arities = self.schema.arities
        if self._uses_relation_major_args() and graph_index is not None:
            starts = self._relation_major_slot_offsets_for_graph(graph_index)
            relation_major_out: dict[str, torch.Tensor] = {}
            for relation_idx, name in enumerate(self.schema.names):
                arity = arities[relation_idx]
                instances = (
                    int(counts[relation_idx].item()) if counts.numel() > 0 else 0
                )
                slots = instances * arity
                start = int(starts[relation_idx].item())
                chunk = relation_args[start : start + slots]
                if arity > 0:
                    relation_major_out[name] = chunk.view(instances, arity)
                else:
                    relation_major_out[name] = chunk.new_empty((instances, 0))
            return relation_major_out
        start = self._relation_arg_start(graph_index)
        graph_major_out: dict[str, torch.Tensor] = {}
        for relation_idx, name in enumerate(self.schema.names):
            arity = arities[relation_idx]
            instances = int(counts[relation_idx].item()) if counts.numel() > 0 else 0
            slots = instances * arity
            chunk = relation_args[start : start + slots]
            if arity > 0:
                graph_major_out[name] = chunk.view(instances, arity)
            else:
                graph_major_out[name] = chunk.new_empty((instances, 0))
            start += slots
        return graph_major_out

    def graph_node_names(self, graph_index: int = 0) -> list[str]:
        """Return entity-row names for one graph."""
        node_names = getattr(self, "node_names", None)
        if node_names is None:
            start, end = self.graph_node_range(graph_index)
            return [f"entity:{idx}" for idx in range(start, end)]
        if node_names and isinstance(node_names[0], (list, tuple)):
            return [str(name) for name in node_names[graph_index]]
        return [str(name) for name in node_names]

    def graph_object_names(self, graph_index: int = 0) -> list[str]:
        """Return object names for one graph."""
        object_names = getattr(self, "object_names", None)
        if object_names is None:
            return self.graph_node_names(graph_index)
        if object_names and isinstance(object_names[0], (list, tuple)):
            return [str(name) for name in object_names[graph_index]]
        return [str(name) for name in object_names]

    def graph_object_indices(self, graph_index: int = 0) -> torch.Tensor:
        """Return global entity rows that correspond to objects."""
        return self._graph_cat_field_slice(
            field_name="object_indices",
            size_field_name="object_sizes",
            graph_index=graph_index,
        )

    def graph_history_entity_indices(self, graph_index: int = 0) -> torch.Tensor:
        """Return global entity rows used as history carriers."""
        return self._graph_cat_field_slice(
            field_name="history_entity_indices",
            size_field_name="history_entity_sizes",
            graph_index=graph_index,
        )

    def graph_history_entity_dt(self, graph_index: int = 0) -> torch.Tensor:
        """Return the `dt` values for history carrier rows."""
        return self._graph_cat_field_slice(
            field_name="history_entity_dt",
            size_field_name="history_entity_sizes",
            graph_index=graph_index,
        )

    def graph_history_entity_names(self, graph_index: int = 0) -> list[str]:
        """Return names for the history carrier rows of one graph."""
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
        """Return candidate-carrier rows for one graph.

        Use `group` to limit the result to one source, such as `goal`,
        `subgoal`, `action`, or `history`.
        """
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
        """Return source-group ids for target-entity rows."""
        return self._graph_cat_field_slice(
            field_name="target_entity_group_ids",
            size_field_name="target_entity_sizes",
            graph_index=graph_index,
        )

    def graph_target_entity_names(
        self, graph_index: int = 0, group: str | int | None = None
    ) -> list[str]:
        """Return names for target-entity rows, optionally filtered by group."""
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
        """Return entity-row positions for prediction targets."""
        return self._graph_cat_field_slice(
            field_name="target_positions",
            size_field_name="target_sizes",
            graph_index=graph_index,
        )

    def graph_target_indices(self, graph_index: int = 0) -> torch.Tensor:
        """Return per-graph target indices in encounter order."""
        return self._graph_cat_field_slice(
            field_name="target_indices",
            size_field_name="target_sizes",
            graph_index=graph_index,
        )

    def graph_target_candidate_ids(self, graph_index: int = 0) -> torch.Tensor:
        """Return stable candidate ids for prediction targets."""
        return self._graph_cat_field_slice(
            field_name="target_candidate_ids",
            size_field_name="target_sizes",
            graph_index=graph_index,
        )

    def graph_target_depths(self, graph_index: int = 0) -> torch.Tensor:
        """Return target depths when the encoder emitted them."""
        return self._graph_cat_field_slice(
            field_name="target_depths",
            size_field_name="target_sizes",
            graph_index=graph_index,
        )

    def graph_target_group_ids(self, graph_index: int = 0) -> torch.Tensor:
        """Return source-group ids for prediction targets."""
        return self._graph_cat_field_slice(
            field_name="target_group_ids",
            size_field_name="target_sizes",
            graph_index=graph_index,
        )

    def graph_target_names(self, graph_index: int = 0) -> list[str]:
        """Return display names for prediction targets."""
        target_names = getattr(self, "target_names", None)
        if target_names is None:
            return []
        if target_names and isinstance(target_names[0], (list, tuple)):
            return [str(name) for name in target_names[graph_index]]
        return [str(name) for name in target_names]

    def graph_relation_instance_range(self, graph_index: int = 0) -> tuple[int, int]:
        """Return the global relation-instance index range for one graph."""
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
        """Return packed TN LGAN edges for one graph."""
        return self._graph_edge_slice(
            src_field_name="lgan_tn_relation_indices",
            dst_field_name="lgan_tn_entity_indices",
            size_field_name="lgan_tn_sizes",
            graph_index=graph_index,
        )

    def graph_lgan_nn_edges(self, graph_index: int = 0) -> torch.Tensor:
        """Return packed NN LGAN edges for one graph."""
        return self._graph_edge_slice(
            src_field_name="lgan_nn_relation_indices",
            dst_field_name="lgan_nn_entity_indices",
            size_field_name="lgan_nn_sizes",
            graph_index=graph_index,
        )

    def graph_lgan_rr_edges(self, graph_index: int = 0) -> torch.Tensor:
        """Return packed RR LGAN edges for one graph."""
        return self._graph_edge_slice(
            src_field_name="lgan_rr_src_relation_indices",
            dst_field_name="lgan_rr_dst_relation_indices",
            size_field_name="lgan_rr_sizes",
            graph_index=graph_index,
        )

    def graph_node_range(self, graph_index: int = 0) -> tuple[int, int]:
        """Return the global entity-row range for one graph."""
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

    def graph_entity_role_ids(self, graph_index: int = 0) -> torch.Tensor:
        """Return per-entity role ids for one graph."""
        start, end = self.graph_node_range(graph_index)
        entity_role_ids = getattr(self, "entity_role_ids", None)
        if entity_role_ids is None:
            return torch.empty((0,), dtype=torch.long)
        return entity_role_ids.long().view(-1)[start:end]

    def graph_entity_roles(self, graph_index: int = 0) -> list[str]:
        """Return decoded per-entity role labels for one graph."""
        role_names = [
            str(value) for value in getattr(self, "entity_role_names", []) or []
        ]
        out: list[str] = []
        for role_id in self.graph_entity_role_ids(graph_index).tolist():
            if 0 <= role_id < len(role_names):
                out.append(role_names[role_id])
            else:
                out.append(str(role_id))
        return out

    @property
    def num_graphs(self) -> int:
        """Return how many graphs are stored in this flat carrier."""
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
            relation_count = len(self.schema.names)
            if self.num_graphs > 1 and relation_count > 0:
                expected = self.num_graphs * relation_count
                if counts.numel() == expected:
                    counts_matrix = counts.view(self.num_graphs, relation_count)
                    if graph_index is None:
                        return counts_matrix.sum(dim=0)
                    if graph_index < 0 or graph_index >= counts_matrix.size(0):
                        raise IndexError(
                            f"graph_index {graph_index} out of range for "
                            f"{counts_matrix.size(0)} graphs"
                        )
                    return counts_matrix[graph_index]
            return counts
        if graph_index is None:
            return counts.sum(dim=0)
        if graph_index < 0 or graph_index >= counts.size(0):
            raise IndexError(
                f"graph_index {graph_index} out of range for {counts.size(0)} graphs"
            )
        return counts[graph_index]

    def _relation_counts_matrix(
        self, device: torch.device | None = None
    ) -> torch.Tensor:
        counts = getattr(self, "relation_counts", None)
        relation_count = len(self.schema.names)
        if counts is None:
            return torch.zeros(
                (self.num_graphs, relation_count),
                dtype=torch.long,
                device=device or torch.device("cpu"),
            )
        counts = counts.long()
        if device is not None:
            counts = counts.to(device=device)
        if counts.dim() == 1:
            if self.num_graphs > 1 and relation_count > 0:
                expected = self.num_graphs * relation_count
                if counts.numel() == expected:
                    return counts.view(self.num_graphs, relation_count)
            return counts.view(1, -1)
        return counts

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
        counts = self._relation_counts_matrix()
        if graph_index <= 0:
            return 0
        arities = self._relation_arities_tensor(counts.device)
        prior_counts = counts[:graph_index].sum(dim=0)
        return int((prior_counts * arities).sum().item())

    def _relation_args_layout(self) -> str:
        layout = getattr(self, "relation_args_layout", None)
        if layout is None:
            return _RELATION_ARGS_GRAPH_MAJOR
        value = str(layout)
        if value not in {_RELATION_ARGS_GRAPH_MAJOR, _RELATION_ARGS_RELATION_MAJOR}:
            raise ValueError(
                "Unknown FlatRelationData relation_args_layout "
                f"{value!r}; expected 'graph_major' or 'relation_major'"
            )
        return value

    def _uses_relation_major_args(self) -> bool:
        return self._relation_args_layout() == _RELATION_ARGS_RELATION_MAJOR

    def _relation_major_slot_offsets_for_graph(self, graph_index: int) -> torch.Tensor:
        counts = self._relation_counts_matrix()
        if graph_index < 0 or graph_index >= counts.size(0):
            raise IndexError(
                f"graph_index {graph_index} out of range for {counts.size(0)} graphs"
            )
        arities = self._relation_arities_tensor(counts.device)
        slot_counts = counts * arities
        relation_totals = slot_counts.sum(dim=0)
        relation_starts = torch.zeros(
            (relation_totals.numel() + 1,),
            dtype=torch.long,
            device=counts.device,
        )
        if relation_totals.numel() > 0:
            relation_starts[1:] = torch.cumsum(relation_totals, dim=0)
        prior_graph_slots = (
            slot_counts[:graph_index].sum(dim=0)
            if graph_index > 0
            else torch.zeros_like(relation_totals)
        )
        return relation_starts[:-1] + prior_graph_slots

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
    """Cast a PyG `Data` or `Batch` into `FlatRelationData`.

    This keeps existing tensors and only normalizes the flat schema and shared
    metadata fields.
    """
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
    relation_logical_arities = getattr(out, "relation_logical_arities", None)
    if relation_logical_arities is not None:
        out.relation_logical_arities = _normalize_int_tuple(relation_logical_arities)
    relation_encoded_arities = getattr(out, "relation_encoded_arities", None)
    if relation_encoded_arities is not None:
        out.relation_encoded_arities = _normalize_int_tuple(relation_encoded_arities)
    relation_slot_roles = getattr(out, "relation_slot_roles", None)
    if relation_slot_roles is not None:
        out.relation_slot_roles = _normalize_int_tuple(relation_slot_roles)
    relation_slot_role_offsets = getattr(out, "relation_slot_role_offsets", None)
    if relation_slot_role_offsets is not None:
        out.relation_slot_role_offsets = _normalize_int_tuple(
            relation_slot_role_offsets
        )
    slot_role_names = getattr(out, "slot_role_names", None)
    if slot_role_names is not None:
        out.slot_role_names = _normalize_str_tuple(slot_role_names)
    if schema_fingerprint is not None:
        out.schema_fingerprint = str(int(schema_fingerprint))
    return normalize_flat_relation_batch_metadata(out)


__all__ = [
    "FlatRelationData",
    "FlatRelationSchema",
    "flat_relation_data_from_pyg",
    "normalize_flat_relation_batch_metadata",
]
