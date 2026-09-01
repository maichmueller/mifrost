"""Cross-stack export adapters for derived-graph encoder output.

The derived-graph encoders emit stock PyTorch Geometric carriers
(:class:`~mifrost.encoders.derived_graph_data.DerivedGraphData`) whose
integer-id channel contract is documented in
``docs/how-to/consume-with-vanilla-gnns.md``. This module converts one
encoded graph into the equivalent DGL or Jraph payload so downstream
projects built on those stacks can consume mifrost encodings without
reimplementing the channel layout.

Both adapters are thin structural converters and return a
``(graph, metadata)`` pair:

- ``graph`` is a :class:`dgl.DGLGraph` (from :func:`to_dgl`) or a
  :class:`jraph.GraphsTuple` (from :func:`to_jraph`) carrying exactly the
  node/edge channels of the PyG carrier.
- ``metadata`` is a plain dict with every carrier attribute that is not a
  core node/edge channel: the string vocabularies, the batch-invariant
  scalars, and all optional strategy tensors (hyperedge / tuple / instance /
  anchor / spd views) verbatim. The collection is derived from the carrier
  rather than from a fixed allowlist, so new encoder fields cannot be
  dropped silently.

DGL and Jraph are optional dependencies: both functions import their
target library lazily and raise :class:`ImportError` with an install hint
when it is missing. The adapters guarantee structural equality only — the
deep behavioural guarantee remains the PyG conformance suite
(``tests/encoding/test_gnn_conformance.py``).
"""

from __future__ import annotations

from typing import Any

__all__ = ["to_dgl", "to_jraph"]

#: Channels the adapters map onto the target library's own node/edge storage.
#: Everything else a carrier holds is metadata by definition, which is what
#: keeps :func:`_collect_metadata` from silently dropping a newly added field.
_CORE_ATTRS: frozenset[str] = frozenset(
    {
        "x",
        "x_ids",
        "edge_index",
        "edge_attr",
        "edge_attr_ids",
        "num_nodes",
        "batch",
        "ptr",
    }
)

#: Non-tensor carrier attributes, listed for documentation and stable
#: ordering. The sweep below picks up anything missing from this list.
_METADATA_ATTRS: tuple[str, ...] = (
    "vocab_roles",
    "vocab_categories",
    "vocab_predicates",
    "vocab_actions",
    "vocab_relations",
    "vocab_edge_kinds",
    "channel_names",
    "edge_channel_names",
    "node_names",
    "object_names",
    "node_universe",
    "atom_expansion",
    "num_predicates",
    "has_anchor",
    "hyperedge_note",
)

#: Strategy tensors, same purpose as :data:`_METADATA_ATTRS`.
_TENSOR_EXTRA_ATTRS: tuple[str, ...] = (
    "hyperedge_index",
    "hyperedge_attr_ids",
    "num_hyperedges",
    "tuple_args",
    "tuple_sizes",
    "tuple_ptr",
    "tuple_rel_ids",
    "tuple_role_ids",
    "tuple_sign_ids",
    "tuple_level_ids",
    "tuple_dt_ids",
    "tuple_category_ids",
    "tuple_attr_ids",
    "num_tuples",
    "instance_node_indices",
    "anchor_index",
    "history_dt_offset",
    "spd_src",
    "spd_dst",
    "spd_dist",
)


def _carrier_attribute_names(data: Any) -> tuple[str, ...]:
    """Return the carrier's own attribute names, or ``()`` when unavailable."""
    keys = getattr(data, "keys", None)
    if not callable(keys):
        return ()
    try:
        return tuple(str(key) for key in keys())
    except Exception:  # pragma: no cover - defensive: non-PyG duck types
        return ()


def _collect_metadata(data: Any) -> dict[str, Any]:
    """Return every carrier attribute that is not a core node/edge channel.

    The documented names in :data:`_METADATA_ATTRS` and
    :data:`_TENSOR_EXTRA_ATTRS` are collected first so the dict has a stable,
    readable order. A second pass then sweeps whatever else the carrier
    actually holds (minus :data:`_CORE_ATTRS`), so a field added to the
    encoder contract reaches DGL and Jraph consumers without anyone having to
    remember to extend a list here. Carriers that expose no ``keys()`` (plain
    duck-typed objects) only get the explicit pass.
    """
    metadata: dict[str, Any] = {}
    for attr in (*_METADATA_ATTRS, *_TENSOR_EXTRA_ATTRS):
        value = getattr(data, attr, None)
        if value is not None:
            metadata[attr] = value
    for attr in _carrier_attribute_names(data):
        if attr.startswith("_") or attr in _CORE_ATTRS or attr in metadata:
            continue
        value = getattr(data, attr, None)
        if value is not None:
            metadata[attr] = value
    return metadata


def _declared_hyperedge_count(data: Any) -> int | None:
    """Return the carrier's authoritative hyperedge count when it has one."""
    import torch

    declared = getattr(data, "num_hyperedges", None)
    if isinstance(declared, torch.Tensor):
        return int(declared.reshape(-1).sum().item())
    if isinstance(declared, (int, float)):
        return int(declared)
    if isinstance(declared, (list, tuple)) and declared:
        return int(sum(int(entry) for entry in declared))
    attr_ids = getattr(data, "hyperedge_attr_ids", None)
    if isinstance(attr_ids, torch.Tensor) and attr_ids.dim() > 0:
        return int(attr_ids.size(0))
    return None


def _require(module: str, install_hint: str) -> Any:
    try:
        return __import__(module)
    except ImportError as exc:
        raise ImportError(
            f"{module!r} is required for this cross-stack adapter but is not "
            f"installed; install it with `{install_hint}`"
        ) from exc


def _reject_batch(data: Any) -> None:
    """Refuse PyG ``Batch`` carriers: the adapters convert one graph."""

    try:
        from torch_geometric.data import Batch
    except ImportError:
        return
    if isinstance(data, Batch):
        raise TypeError(
            "cross-stack adapters convert one encoded graph; call "
            ".to_data_list() or index the batch first"
        )


def to_dgl(data: Any) -> tuple[Any, dict[str, Any]]:
    """Convert one derived-graph PyG carrier into ``(dgl graph, metadata)``.

    Raises :class:`TypeError` when ``data`` is a
    :class:`torch_geometric.data.Batch`: the adapters convert one encoded
    graph at a time.

    The returned graph is an int64 CPU ``dgl.graph((src, dst))`` over
    ``num_nodes`` nodes (isolated objects survive) with:

    - ``ndata["x_ids"]``: the ``[N, 6]`` float id channels verbatim,
    - ``ndata["role_ids"]``: convenience ``x_ids[:, 0].long()`` role column,
    - ``edata["edge_attr"]``: the ``[E, 9]`` float edge channels verbatim.

    The metadata dict carries the shared vocabularies
    (``vocab_roles`` / ``vocab_predicates`` / ``vocab_actions`` /
    ``vocab_relations`` / ``vocab_edge_kinds`` / ``channel_names`` /
    ``edge_channel_names``), ``node_names`` when exported, and every other
    non-core attribute the carrier holds. When the input carries hyperedge
    membership (``HypergraphIncidenceEncoder``), two extra entries appear
    because core DGL has no native hypergraph:

    - ``hyperedge_index``: the raw ``[2, sum(sizes)]`` membership tensor
      (node row, hyperedge row),
    - ``hyperedge_bipartite``: a second int64 DGL graph of ``N + H`` nodes
      in which each member node points at its hyperedge anchor, whose id is
      the hyperedge index offset by ``N``.

    ``H`` is the carrier's authoritative hyperedge count
    (``num_hyperedges`` / ``hyperedge_attr_ids.size(0)``) when it declares
    one, and ``max(hyperedge_index[1]) + 1`` otherwise. The encoder core
    guarantees no empty hyperedge (an arity-0 instance takes the anchor node
    as its single member), so the two agree by construction and a
    disagreement means a corrupted carrier — it is reported rather than
    silently under- or over-counting.
    """
    _reject_batch(data)
    dgl = _require("dgl", "pip install dgl")
    import torch

    num_nodes = int(data.num_nodes)
    src = data.edge_index[0].long().cpu()
    dst = data.edge_index[1].long().cpu()
    graph = dgl.graph(
        (src, dst),
        num_nodes=num_nodes,
        idtype=torch.int64,
        device=torch.device("cpu"),
    )
    graph.ndata["x_ids"] = data.x_ids.cpu()
    graph.ndata["role_ids"] = data.x_ids[:, 0].long().cpu()
    graph.edata["edge_attr"] = data.edge_attr.cpu()

    metadata = _collect_metadata(data)
    hyperedge_index = getattr(data, "hyperedge_index", None)
    if isinstance(hyperedge_index, torch.Tensor) and hyperedge_index.numel() > 0:
        members = hyperedge_index[0].long().cpu()
        anchor_rows = hyperedge_index[1].long().cpu()
        num_hyperedges = int(anchor_rows.max().item()) + 1
        declared = _declared_hyperedge_count(data)
        if declared is not None:
            if declared != num_hyperedges:
                raise ValueError(
                    "hyperedge membership disagrees with the declared hyperedge "
                    f"count ({num_hyperedges} implied by hyperedge_index[1], "
                    f"{declared} declared); the encoder core guarantees every "
                    "hyperedge has at least one member, so this carrier is "
                    "inconsistent"
                )
            num_hyperedges = declared
        metadata["hyperedge_bipartite"] = dgl.graph(
            (members, anchor_rows + num_nodes),
            num_nodes=num_nodes + num_hyperedges,
            idtype=torch.int64,
            device=torch.device("cpu"),
        )
    elif hyperedge_index is not None:
        # Empty membership: no hyperedges to anchor, but keep the entry so
        # consumers can rely on the key whenever the input carries the attr.
        empty = torch.empty(0, dtype=torch.int64)
        metadata["hyperedge_bipartite"] = dgl.graph(
            (empty, empty),
            num_nodes=num_nodes,
            idtype=torch.int64,
            device=torch.device("cpu"),
        )
    return graph, metadata


def to_jraph(data: Any) -> tuple[Any, dict[str, Any]]:
    """Convert one derived-graph PyG carrier into ``(GraphsTuple, metadata)``.

    Raises :class:`TypeError` when ``data`` is a
    :class:`torch_geometric.data.Batch`: the adapters convert one encoded
    graph at a time.

    The returned single-graph ``jraph.GraphsTuple`` stores:

    - ``nodes``: the ``[N, 6]`` float id channels,
    - ``edges``: the ``[E, 9]`` float edge channels,
    - ``senders`` / ``receivers``: ``edge_index[0]`` / ``edge_index[1]``,
    - ``n_node=[N]``, ``n_edge=[E]``,
    - ``globals``: the metadata dict itself (vocabularies plus every
      hyperedge/tuple/spd tensor).

    The metadata dict mirrors ``globals`` so callers that want to run
    ``jax.jit`` can swap ``globals`` out first: string-valued vocabularies
    are valid pytree leaves outside tracing but not valid JAX input types.
    """
    _reject_batch(data)
    jraph = _require("jraph", "pip install jraph jax")
    import jax.numpy as jnp

    metadata = _collect_metadata(data)
    graphs_tuple = jraph.GraphsTuple(
        nodes=jnp.asarray(data.x_ids.float()),
        edges=jnp.asarray(data.edge_attr.float()),
        receivers=jnp.asarray(data.edge_index[1].long()),
        senders=jnp.asarray(data.edge_index[0].long()),
        n_node=jnp.asarray([int(data.num_nodes)]),
        n_edge=jnp.asarray([int(data.edge_index.size(1))]),
        globals=metadata,
    )
    return graphs_tuple, dict(metadata)
