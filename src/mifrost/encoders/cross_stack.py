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
- ``metadata`` is a plain dict with every non-tensor vocabulary attribute
  plus all optional strategy tensors (hyperedge / tuple / spd views)
  verbatim.

DGL and Jraph are optional dependencies: both functions import their
target library lazily and raise :class:`ImportError` with an install hint
when it is missing. The adapters guarantee structural equality only — the
deep behavioural guarantee remains the PyG conformance suite
(``tests/encoding/test_gnn_conformance.py``).
"""

from __future__ import annotations

from typing import Any

__all__ = ["to_dgl", "to_jraph"]

_METADATA_ATTRS: tuple[str, ...] = (
    "vocab_roles",
    "vocab_predicates",
    "vocab_edge_kinds",
    "channel_names",
    "edge_channel_names",
    "node_names",
)

_TENSOR_EXTRA_ATTRS: tuple[str, ...] = (
    "hyperedge_index",
    "hyperedge_attr_ids",
    "tuple_args",
    "tuple_ptr",
    "tuple_rel_ids",
    "tuple_role_ids",
    "spd_src",
    "spd_dst",
    "spd_dist",
)


def _collect_metadata(data: Any) -> dict[str, Any]:
    """Return every present vocabulary attribute and strategy tensor."""
    metadata: dict[str, Any] = {}
    for attr in (*_METADATA_ATTRS, *_TENSOR_EXTRA_ATTRS):
        value = getattr(data, attr, None)
        if value is not None:
            metadata[attr] = value
    return metadata


def _require(module: str, install_hint: str) -> Any:
    try:
        return __import__(module)
    except ImportError as exc:
        raise ImportError(
            f"{module!r} is required for this cross-stack adapter but is not "
            f"installed; install it with `{install_hint}`"
        ) from exc


def to_dgl(data: Any) -> tuple[Any, dict[str, Any]]:
    """Convert one derived-graph PyG carrier into ``(dgl graph, metadata)``.

    The returned graph is an int64 CPU ``dgl.graph((src, dst))`` over
    ``num_nodes`` nodes (isolated objects survive) with:

    - ``ndata["x_ids"]``: the ``[N, 6]`` float id channels verbatim,
    - ``ndata["role_ids"]``: convenience ``x_ids[:, 0].long()`` role column,
    - ``edata["edge_attr"]``: the ``[E, 3]`` float edge channels verbatim.

    The metadata dict carries the shared vocabularies
    (``vocab_roles`` / ``vocab_predicates`` / ``vocab_edge_kinds`` /
    ``channel_names`` / ``edge_channel_names``), ``node_names`` when
    exported, and every strategy tensor unchanged. When the input carries
    hyperedge membership (``HypergraphIncidenceEncoder``), two extra
    entries appear because core DGL has no native hypergraph:

    - ``hyperedge_index``: the raw ``[2, M]`` membership tensor
      (node row, hyperedge row),
    - ``hyperedge_bipartite``: a second int64 DGL graph of ``N + M`` nodes
      in which each member node points at its hyperedge anchor, whose id
      is the hyperedge index offset by ``N``.
    """
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
    if hyperedge_index is not None:
        members = hyperedge_index[0].long()
        anchors = hyperedge_index[1].long() + num_nodes
        metadata["hyperedge_bipartite"] = dgl.graph(
            (members, anchors),
            num_nodes=num_nodes + int(hyperedge_index[1].max().item()) + 1,
            idtype=torch.int64,
            device=torch.device("cpu"),
        )
    return graph, metadata


def to_jraph(data: Any) -> tuple[Any, dict[str, Any]]:
    """Convert one derived-graph PyG carrier into ``(GraphsTuple, metadata)``.

    The returned single-graph ``jraph.GraphsTuple`` stores:

    - ``nodes``: the ``[N, 6]`` float id channels,
    - ``edges``: the ``[E, 3]`` float edge channels,
    - ``senders`` / ``receivers``: ``edge_index[0]`` / ``edge_index[1]``,
    - ``n_node=[N]``, ``n_edge=[E]``,
    - ``globals``: the metadata dict itself (vocabularies plus every
      hyperedge/tuple/spd tensor).

    The metadata dict mirrors ``globals`` so callers that want to run
    ``jax.jit`` can swap ``globals`` out first: string-valued vocabularies
    are valid pytree leaves outside tracing but not valid JAX input types.
    """
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
