from __future__ import annotations

from functools import cached_property
from typing import Any

import torch
from torch_geometric.data import Batch, Data


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


def normalize_derived_graph_batch_metadata(
    data: DerivedGraphData | Batch,
) -> DerivedGraphData | Batch:
    """Normalize shared vocabulary metadata after native or PyG batching.

    PyG collects non-tensor attributes into a per-graph list during batching;
    this collapses identical vocabulary lists back into one shared list. It
    does not change any stored derived graph tensors.
    """
    for attr in (
        "vocab_roles",
        "vocab_predicates",
        "vocab_edge_kinds",
        "channel_names",
        "edge_channel_names",
    ):
        setattr(data, attr, _normalize_shared_str_list(getattr(data, attr, None)))
    return data


class DerivedGraphData(Data):
    """Integer-id homogeneous derived graph carrier for vanilla GNN layers.

    Core channels are ``x_ids`` ([N, F]), ``edge_index`` ([2, E]) and
    ``edge_attr_ids`` ([E, Fe]). Optional strategy extras are
    ``hyperedge_index`` / ``hyperedge_attr_ids``, ``line_edge_index`` /
    ``line_edge_attr_ids``, the CSR-style tuple channels ``tuple_args`` /
    ``tuple_ptr`` / ``tuple_rel_ids`` and sparse pairwise distances
    ``spd_src`` / ``spd_dst`` / ``spd_dist``. Shared string vocabularies are
    carried as plain python metadata attributes.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if "num_nodes" in self._store:
            return
        x_ids = self._store.get("x_ids")
        if isinstance(x_ids, torch.Tensor) and x_ids.dim() > 0:
            self._store.num_nodes = int(x_ids.size(0))

    @property
    def num_nodes(self) -> int:
        """Return the number of entity nodes carried by this graph."""
        store = getattr(self, "_store", None)
        if store is not None and "num_nodes" in store:
            stored = store["num_nodes"]
            if stored is not None:
                return int(stored)
        x_ids = getattr(self, "x_ids", None)
        if torch.is_tensor(x_ids) and x_ids.dim() > 0:
            return int(x_ids.size(0))
        inferred = super().num_nodes
        return int(inferred) if inferred is not None else 0

    @num_nodes.setter
    def num_nodes(self, value: int | None) -> None:
        setattr(self._store, "num_nodes", value)

    def __inc__(self, key: str, value: Any, *args, **kwargs) -> Any:
        if key in {"tuple_args", "spd_src", "spd_dst"}:
            return self.num_nodes
        if key == "hyperedge_index":
            return torch.tensor(
                ((self.num_nodes,), (self._num_hyperedges(),)),
                dtype=torch.long,
                device=value.device if torch.is_tensor(value) else None,
            )
        if key == "hyperedge_attr_ids":
            return self._num_hyperedges()
        if key in {"tuple_ptr", "tuple_rel_ids", "tuple_role_ids"}:
            return self._num_tuples()
        return super().__inc__(key, value, *args, **kwargs)

    @cached_property
    def schema_summary(self) -> dict[str, int]:
        """Return basic size statistics for the stored derived graph."""
        edge_index = getattr(self, "edge_index", None)
        spd_src = getattr(self, "spd_src", None)
        edges = (
            int(edge_index.size(-1))
            if torch.is_tensor(edge_index) and edge_index.dim() >= 1
            else 0
        )
        spd_pairs = int(spd_src.view(-1).numel()) if torch.is_tensor(spd_src) else 0
        return {
            "nodes": self.num_nodes,
            "edges": edges,
            "hyperedges": self._num_hyperedges(),
            "tuples": self._num_tuples(),
            "spd_pairs": spd_pairs,
        }

    def padded_tuple_matrix(
        self, fill_value: int = -1
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return dense per-tuple argument ids plus a validity mask.

        The result is built from the CSR-style ``tuple_args`` / ``tuple_ptr``
        pair. Rows shorter than the maximum arity are right-padded with
        ``fill_value`` and marked as invalid in the boolean mask.
        """
        empty_args = torch.empty((0, 0), dtype=torch.long)
        empty_mask = torch.empty((0, 0), dtype=torch.bool)
        tuple_args = getattr(self, "tuple_args", None)
        tuple_ptr = getattr(self, "tuple_ptr", None)
        if not torch.is_tensor(tuple_args) or not torch.is_tensor(tuple_ptr):
            return empty_args, empty_mask
        args_flat = tuple_args.long().view(-1)
        ptr = tuple_ptr.long().view(-1)
        num_tuples = max(ptr.numel() - 1, 0)
        if num_tuples == 0:
            return empty_args, empty_mask
        sizes = ptr[1:] - ptr[:-1]
        width = int(sizes.max().item())
        out = args_flat.new_full((num_tuples, width), fill_value)
        mask = torch.zeros((num_tuples, width), dtype=torch.bool)
        for row in range(num_tuples):
            start = int(ptr[row])
            end = int(ptr[row + 1])
            out[row, : end - start] = args_flat[start:end]
            mask[row, : end - start] = True
        return out, mask

    def _num_hyperedges(self) -> int:
        stored = getattr(self, "num_hyperedges", None)
        if stored is not None:
            if torch.is_tensor(stored):
                flat = stored.view(-1).tolist()
                if len(flat) > 1:
                    return int(sum(int(entry) for entry in flat))
                return int(flat[0]) if flat else 0
            if isinstance(stored, (list, tuple)):
                return int(sum(int(entry) for entry in stored))
            return int(stored)
        hyperedge_attr_ids = getattr(self, "hyperedge_attr_ids", None)
        if torch.is_tensor(hyperedge_attr_ids) and hyperedge_attr_ids.dim() > 0:
            return int(hyperedge_attr_ids.size(0))
        hyperedge_index = getattr(self, "hyperedge_index", None)
        if (
            torch.is_tensor(hyperedge_index)
            and hyperedge_index.dim() == 2
            and hyperedge_index.size(1) > 0
        ):
            return int(hyperedge_index[1].max().item()) + 1
        return 0

    def _num_tuples(self) -> int:
        tuple_rel_ids = getattr(self, "tuple_rel_ids", None)
        if torch.is_tensor(tuple_rel_ids) and tuple_rel_ids.dim() > 0:
            return int(tuple_rel_ids.size(0))
        tuple_ptr = getattr(self, "tuple_ptr", None)
        if torch.is_tensor(tuple_ptr):
            return max(tuple_ptr.numel() - 1, 0)
        return 0


__all__ = [
    "DerivedGraphData",
    "normalize_derived_graph_batch_metadata",
]
