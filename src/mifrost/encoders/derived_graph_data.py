from __future__ import annotations

from functools import cached_property
from typing import Any

import torch
from torch_geometric.data import Batch, Data

#: Vocabulary attributes that native batching emits once per batch but PyG
#: duplicates per graph under ``Batch.from_data_list``.
_SHARED_LIST_ATTRS: tuple[str, ...] = (
    "vocab_roles",
    "vocab_categories",
    "vocab_predicates",
    "vocab_actions",
    "vocab_relations",
    "vocab_edge_kinds",
    "channel_names",
    "edge_channel_names",
)

#: Batch-invariant scalar graph attributes. Native batching emits one value;
#: ``Batch.from_data_list`` turns them into a per-graph tensor (ints) or list
#: (strings), so they are collapsed back when every graph agrees.
_SHARED_SCALAR_ATTRS: tuple[str, ...] = (
    "node_universe",
    "atom_expansion",
    "num_predicates",
    "has_anchor",
    "hyperedge_note",
)


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
    """Collapse a per-graph repetition of one batch-invariant scalar."""
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        if value.dim() == 0:
            return value
        entries = value.reshape(-1).tolist()
    elif isinstance(value, (list, tuple)):
        entries = list(value)
    else:
        return value
    if not entries:
        return value
    first = entries[0]
    return first if all(entry == first for entry in entries) else value


def normalize_derived_graph_batch_metadata(
    data: DerivedGraphData | Batch,
) -> DerivedGraphData | Batch:
    """Normalize shared vocabulary metadata after native or PyG batching.

    PyG keeps per-graph copies of list- and scalar-valued attributes when
    tensors are re-batched manually (e.g. ``Batch.from_data_list([...])``);
    this collapses identical vocabulary lists (``vocab_roles`` ...
    ``vocab_actions`` / ``vocab_relations``) and identical batch-invariant
    scalars (``node_universe``, ``atom_expansion``, ``num_predicates``,
    ``has_anchor``, ``hyperedge_note``) back into the single shared value the
    native batch path emits. Attributes the carrier does not hold are left
    absent, and no derived graph tensor channel is touched.

    Example:
        batch = normalize_derived_graph_batch_metadata(Batch.from_data_list(graphs))
    """
    for attr in _SHARED_LIST_ATTRS:
        value = getattr(data, attr, None)
        if value is None:
            continue
        setattr(data, attr, _normalize_shared_str_list(value))
    for attr in _SHARED_SCALAR_ATTRS:
        value = getattr(data, attr, None)
        if value is None:
            continue
        setattr(data, attr, _normalize_shared_scalar(value))
    return data


class DerivedGraphData(Data):
    """Integer-id homogeneous derived graph carrier for vanilla GNN layers.

    Core channels are ``x_ids`` (``[N, 6]``: role, ``relation_id + 1``, sign,
    goal level, history dt, category), ``edge_index`` (``[2, E]``) and
    ``edge_attr`` (``[E, 9]``: kind, pos_a, pos_b, ``rel_id + 1``, role, sign,
    goal level, history dt, category). Column 4 of ``x_ids`` and column 7 of
    ``edge_attr`` are the one *signed* channel (history dt); shift them by
    ``history_dt_offset`` before embedding.

    Per-graph scalars carried as tensors so they survive batching:
    ``history_dt_offset`` (``[1]`` / ``[B]``, always present), ``anchor_index``
    (``[1]`` / ``[B]``, present iff an anchor node was emitted; read it through
    :attr:`anchor_node_index` for the ``-1`` sentinel), ``num_hyperedges`` and
    ``num_tuples``.

    Optional strategy extras:

    - hyperedge view: ``hyperedge_index`` (``[2, sum(sizes)]``: node row,
      hyperedge row) and ``hyperedge_attr_ids`` (``[M, 6]``, same column order
      as ``x_ids``). Every hyperedge has at least one member, so
      ``hyperedge_index[1].max() + 1 == hyperedge_attr_ids.size(0)``.
    - tuple view: ``tuple_args`` / ``tuple_sizes`` / ``tuple_ptr`` (CSR over
      the flattened argument node ids) plus the per-tuple id channels
      ``tuple_rel_ids`` (raw relation id, ``-1`` = none), ``tuple_role_ids``,
      ``tuple_sign_ids``, ``tuple_level_ids``, ``tuple_dt_ids``,
      ``tuple_category_ids`` and the stacked convenience ``tuple_attr_ids``
      (``[T, 6]``, ``x_ids`` column order, so column 1 is ``tuple_rel_ids + 1``).
    - instance table: ``instance_node_indices`` (``[M]``, the reified node of
      each instance row, ``-1`` where the universe reifies nothing).
    - shortest-path bias: ``spd_src`` / ``spd_dst`` / ``spd_dist``.

    Shared string vocabularies (``vocab_roles``, ``vocab_predicates``,
    ``vocab_actions``, ``vocab_relations``, ``vocab_edge_kinds``,
    ``vocab_categories``, ``channel_names``, ``edge_channel_names``) and the
    batch-invariant scalars ``node_universe`` / ``atom_expansion`` /
    ``num_predicates`` / ``has_anchor`` ride along as plain python metadata
    attributes.

    The two *per-graph* name tables, ``node_names`` and ``object_names``, are
    nested identically: a single graph carries a flat ``list[str]``, and a
    batch of ``B`` graphs carries a list of ``B`` lists - on native batching,
    stream flushes and ``Batch.from_data_list`` alike. ``object_names`` names
    exactly the role-0 nodes of its graph and is a prefix of that graph's
    ``node_names``.

    There is deliberately no ``x``: every node channel is an integer id in
    ``x_ids``, so a shipped ``x`` could only be an all-zero placeholder that
    silently turns a stock layer into a constant function. ``data.x`` is
    ``None`` and such a call raises instead.

    ``tuple_sizes`` is the batched source of truth for the tuple CSR:
    ``tuple_ptr`` is *derived* from it on access, so a global CSR is produced
    identically by native batching and by ``Batch.from_data_list``. Carriers
    built by hand that supply only ``tuple_ptr`` keep the stored value.

    Reserved names: ``line_edge_index`` / ``line_edge_attr_ids`` are reserved
    but nothing emits them today - line-graph edges ride in ``edge_index``
    with edge kind ``line_share``.
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
        if isinstance(x_ids, torch.Tensor) and x_ids.dim() > 0:
            return int(x_ids.size(0))
        inferred = super().num_nodes
        return int(inferred) if inferred is not None else 0

    @num_nodes.setter
    def num_nodes(self, value: int | None) -> None:
        setattr(self._store, "num_nodes", value)

    @property
    def tuple_ptr(self) -> torch.Tensor | None:
        """Return the global CSR offsets over ``tuple_args``.

        Derived from ``tuple_sizes`` whenever the carrier holds it, so the
        invariant ``tuple_ptr.numel() - 1 == tuple_rel_ids.numel()`` holds on
        single graphs, native batches and ``Batch.from_data_list`` re-batches
        alike: PyG concatenates the per-graph CSRs (duplicating the offsets
        where graph fragments meet) while ``tuple_sizes`` simply concatenates.
        The recomputed value is written back into the store so ``to_dict()``,
        ``__getitem__`` and cross-stack adapters observe the same tensor.
        """
        derived = self._derived_tuple_ptr()
        if derived is None:
            return self._store.get("tuple_ptr", None)
        stored = self._store.get("tuple_ptr", None)
        if (
            not isinstance(stored, torch.Tensor)
            or stored.shape != derived.shape
            or not torch.equal(stored, derived)
        ):
            self._store["tuple_ptr"] = derived
        return derived

    @tuple_ptr.setter
    def tuple_ptr(self, value: Any) -> None:
        self._store["tuple_ptr"] = value

    @property
    def anchor_node_index(self) -> int:
        """Return this graph's anchor node index, or ``-1`` when absent.

        Recovers the spec's ``-1`` sentinel: the native encoder emits the
        ``anchor_index`` field *only* when an anchor node exists (a per-graph
        value cannot be a batch-invariant graph attr), and mirrors its presence
        in the ``has_anchor`` metadata flag.

        On a batch this returns the first graph's anchor; read the ``[B]``
        ``anchor_index`` tensor directly instead - its entries are already
        node-offset global indices, valid for every graph iff ``has_anchor``
        (the flag is batch-invariant, so it is all graphs or none).
        """
        if not self._has_anchor():
            return -1
        value = getattr(self, "anchor_index", None)
        if not isinstance(value, torch.Tensor) or value.numel() == 0:
            return -1
        return int(value.reshape(-1)[0])

    def __inc__(self, key: str, value: Any, *args, **kwargs) -> Any:
        # Node-offset channels. ``instance_node_indices`` must be named
        # explicitly: PyG's default only offsets keys containing "index", and
        # "indices" does not contain it, so the manual re-batching path would
        # otherwise disagree with the native builder's node offset.
        if key in {
            "tuple_args",
            "spd_src",
            "spd_dst",
            "instance_node_indices",
            "anchor_index",
        }:
            return self.num_nodes
        if key == "hyperedge_index":
            return torch.tensor(
                ((self.num_nodes,), (self._num_hyperedges(),)),
                dtype=torch.long,
                device=value.device if isinstance(value, torch.Tensor) else None,
            )
        if key == "tuple_ptr":
            return self._num_tuple_args()
        # Plain label / count channels: concatenate without any offset. The
        # default is already 0; naming them keeps the contract auditable.
        if key in {
            "hyperedge_attr_ids",
            "tuple_rel_ids",
            "tuple_role_ids",
            "tuple_sign_ids",
            "tuple_level_ids",
            "tuple_dt_ids",
            "tuple_category_ids",
            "tuple_attr_ids",
            "tuple_sizes",
            "history_dt_offset",
            "has_anchor",
            "num_hyperedges",
            "num_tuples",
        }:
            return 0
        return super().__inc__(key, value, *args, **kwargs)

    def __getitem__(self, key: str) -> Any:
        if key == "tuple_ptr":
            self._sync_tuple_ptr()
        return super().__getitem__(key)

    def to_dict(self) -> dict[str, Any]:
        self._sync_tuple_ptr()
        return super().to_dict()

    @cached_property
    def schema_summary(self) -> dict[str, int]:
        """Return basic size statistics for the stored derived graph."""
        edge_index = getattr(self, "edge_index", None)
        spd_src = getattr(self, "spd_src", None)
        edges = (
            int(edge_index.size(-1))
            if isinstance(edge_index, torch.Tensor) and edge_index.dim() >= 1
            else 0
        )
        spd_pairs = (
            int(spd_src.view(-1).numel()) if isinstance(spd_src, torch.Tensor) else 0
        )
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

        Encoded carriers derive ``tuple_ptr`` from ``tuple_sizes``, so the row
        count matches the per-tuple id channels on single graphs, native
        batches and manual re-batches alike. Hand-built carriers that supply
        only ``tuple_ptr`` keep the stored pointers; there, re-batching can
        duplicate the offsets where graph fragments meet, which surfaces as
        zero-width (all-masked) rows. Decreasing or out-of-range pointers are
        rejected instead of silently reading another graph's flattened
        arguments.
        """
        empty_args = torch.empty((0, 0), dtype=torch.long)
        empty_mask = torch.empty((0, 0), dtype=torch.bool)
        tuple_args = getattr(self, "tuple_args", None)
        tuple_ptr = getattr(self, "tuple_ptr", None)
        if not isinstance(tuple_args, torch.Tensor) or not isinstance(
            tuple_ptr, torch.Tensor
        ):
            return empty_args, empty_mask
        args_flat = tuple_args.long().view(-1)
        ptr = tuple_ptr.long().view(-1)
        num_tuples = max(ptr.numel() - 1, 0)
        if num_tuples == 0:
            return empty_args, empty_mask
        if (
            int(ptr[0]) < 0
            or int(ptr[-1]) > args_flat.numel()
            or bool((ptr.diff() < 0).any())
        ):
            raise ValueError(
                "tuple_ptr must be monotonically non-decreasing and within tuple_args"
            )
        sizes = ptr[1:] - ptr[:-1]
        width = int(sizes.max().item())
        out = args_flat.new_full((num_tuples, width), fill_value)
        mask = torch.zeros(
            (num_tuples, width), dtype=torch.bool, device=args_flat.device
        )
        for row in range(num_tuples):
            start = int(ptr[row])
            step = int(ptr[row + 1]) - start
            out[row, :step] = args_flat[start : start + step]
            mask[row, :step] = True
        return out, mask

    def _derived_tuple_ptr(self) -> torch.Tensor | None:
        """Return the CSR implied by ``tuple_sizes``, or ``None`` without it."""
        sizes = self._store.get("tuple_sizes", None)
        if not isinstance(sizes, torch.Tensor):
            return None
        flat = sizes.long().reshape(-1)
        return torch.cat((flat.new_zeros(1), torch.cumsum(flat, 0)))

    def _sync_tuple_ptr(self) -> None:
        """Materialize the derived ``tuple_ptr`` into the backing store."""
        if self._derived_tuple_ptr() is not None:
            _ = self.tuple_ptr

    def _has_anchor(self) -> bool:
        flag = getattr(self, "has_anchor", None)
        if flag is None:
            return getattr(self, "anchor_index", None) is not None
        if isinstance(flag, torch.Tensor):
            return bool(flag.reshape(-1).numel()) and bool(flag.reshape(-1)[0])
        if isinstance(flag, (list, tuple)):
            return bool(flag) and bool(flag[0])
        return bool(flag)

    def _num_hyperedges(self) -> int:
        stored = getattr(self, "num_hyperedges", None)
        if stored is not None:
            if isinstance(stored, torch.Tensor):
                flat = stored.view(-1).tolist()
                if len(flat) > 1:
                    return int(sum(int(entry) for entry in flat))
                return int(flat[0]) if flat else 0
            if isinstance(stored, (list, tuple)):
                return int(sum(int(entry) for entry in stored))
            return int(stored)
        hyperedge_attr_ids = getattr(self, "hyperedge_attr_ids", None)
        if (
            isinstance(hyperedge_attr_ids, torch.Tensor)
            and hyperedge_attr_ids.dim() > 0
        ):
            return int(hyperedge_attr_ids.size(0))
        hyperedge_index = getattr(self, "hyperedge_index", None)
        if (
            isinstance(hyperedge_index, torch.Tensor)
            and hyperedge_index.dim() == 2
            and hyperedge_index.size(1) > 0
        ):
            return int(hyperedge_index[1].max().item()) + 1
        return 0

    def _num_tuples(self) -> int:
        tuple_rel_ids = getattr(self, "tuple_rel_ids", None)
        if isinstance(tuple_rel_ids, torch.Tensor) and tuple_rel_ids.dim() > 0:
            return int(tuple_rel_ids.size(0))
        tuple_ptr = getattr(self, "tuple_ptr", None)
        if isinstance(tuple_ptr, torch.Tensor):
            return max(tuple_ptr.numel() - 1, 0)
        return 0

    def _num_tuple_args(self) -> int:
        tuple_args = getattr(self, "tuple_args", None)
        if isinstance(tuple_args, torch.Tensor):
            return int(tuple_args.numel())
        return 0


__all__ = [
    "DerivedGraphData",
    "normalize_derived_graph_batch_metadata",
]
