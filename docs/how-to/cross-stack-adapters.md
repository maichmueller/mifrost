# How to Export Derived Graphs to DGL and Jraph

The derived-graph encoders speak PyTorch Geometric natively. When your
downstream model stack is built on **DGL** or **Jraph**, you do not need to
reimplement the channel contract: every derived facade can export its output
through thin conversion adapters in
`mifrost.encoders.cross_stack` (also exposed as `.to_dgl()` / `.to_jraph()`
methods on each encoder).

## When to Reach for These Adapters

- Your training loop or model zoo already uses DGL message-passing layers.
- You are running JAX/Flax research code that consumes `jraph.GraphsTuple`.
- You need structural parity checks across stacks in CI.

If you train in PyTorch, prefer the native PyG path (see
[consume-with-vanilla-gnns](consume-with-vanilla-gnns.md)); it stays the deep,
behaviour-tested guarantee.

Install targets lazily: `pip install dgl` or `pip install jraph jax`. The
adapters raise an `ImportError` with these hints when a stack is missing.

## The `(graph, metadata)` Contract

Both adapters return a tuple:

```python
from mifrost.encoders.cross_stack import to_dgl, to_jraph

data = encoder.encode_pyg(state)
dgl_graph, meta = to_dgl(data)
graphs_tuple, meta = to_jraph(data)
```

The graph object carries exactly the integer-id channels:

| Stack | Node channels | Edge channels | Topology |
| --- | --- | --- | --- |
| DGL | `ndata["x_ids"]` (`[N, 6]` float) plus convenience `ndata["role_ids"] = x_ids[:, 0].long()` | `edata["edge_attr"]` (`[E, 3]` float) | int64 CPU `dgl.graph((src, dst))` over `num_nodes` nodes |
| Jraph | `nodes` (`[N, 6]` float) | `edges` (`[E, 3]` float) | single-graph `GraphsTuple`, `senders=edge_index[0]`, `receivers=edge_index[1]`, `n_node=[N]`, `n_edge=[E]` |

`metadata` is a plain dict holding everything else:

- Vocabularies when present: `vocab_roles`, `vocab_predicates`,
  `vocab_edge_kinds`, `channel_names`, `edge_channel_names`.
- `node_names` / `object_names` style extras such as `node_names`.
- Strategy tensors verbatim: `hyperedge_index` / `hyperedge_attr_ids`,
  `tuple_args` / `tuple_ptr` / `tuple_rel_ids` / `tuple_role_ids`, and
  `spd_src` / `spd_dst` / `spd_dist`.

Attributes absent from the input (e.g. `vocab_roles` on
`ObjectFeatureEncoder` output) are simply omitted from the dict.

## Hyperedge Handling per Stack

`HypergraphIncidenceEncoder` output converts everywhere, but the target
representations differ:

- **PyG**: native `HypergraphConv` input via `hyperedge_index`.
- **DGL**: core DGL has no hypergraph primitive, so the adapter additionally
  returns `metadata["hyperedge_bipartite"]`: an int64 graph over `N + M`
  nodes in which each member node points at its hyperedge anchor, whose id is
  the membership column offset by `N`. The raw `[2, M]` tensor also rides in
  `metadata["hyperedge_index"]`.
- **Jraph**: the membership tensors ride in `globals` untouched; factor them
  into bipartite senders/receivers yourself if you want incidence-style
  message passing.

Tuple and spd tensors always pass through `metadata` unchanged on both
stacks.

## Worked Snippet: DGL

```python
import torch
import dgl
import dgl.nn as dglnn
import mifrost
from mifrost.encoders.cross_stack import to_dgl

encoder = mifrost.StarGraphEncoder(problem)
data = encoder.encode_pyg(state)
graph, meta = to_dgl(data)

embeds = torch.nn.ModuleList(
    [torch.nn.Embedding(int(graph.ndata["x_ids"][:, c].max()) + 2, 8)
     for c in range(6)]
)
feats = torch.cat(
    [embeds[c](graph.ndata["x_ids"][:, c].long()) for c in range(6)], dim=-1
)

conv = dglnn.GraphConv(feats.size(-1), 64)   # reverse edges are explicit
out = conv(graph, feats)
```

!!! note
    This snippet follows the stable public DGL API but could not be executed
    on the authoring machine (no macOS/py3.12 DGL wheel compatible with the
    pinned torch build — `dgl` installs yet fails mid-import on the graphbolt
    dylib). The adapter itself is exercised by structural tests wherever a
    usable wheel exists.

## Worked Snippet: Jraph

```python
import jax
import jraph

graphs_tuple, meta = to_jraph(data)

conv = jraph.GraphConvolution(update_node_fn=jax.nn.relu)
out = conv(graphs_tuple._replace(globals=None))  # GCN ignores edge features
```

For `jax.jit`, swap string-valued vocabularies out of `globals` first (they
are valid pytree leaves outside tracing but not valid JAX input types); use
the returned `metadata` dict instead.

## Facade Delegation

Every `_DerivedEncoderBase` facade exposes delegating methods so you never
need to import the module directly:

```python
graph, meta = encoder.to_dgl(data)
graphs_tuple, meta = encoder.to_jraph(data)
```

## Compatibility Guarantee

These adapters are intentionally thin: they copy channels, indices, and
metadata verbatim and are tested for **structural equality only** (node/edge
counts, sender/receiver equality, tensor equality, vocabulary identity). The
deep behavioural guarantee remains the PyG conformance suite
(`tests/encoding/test_gnn_conformance.py`) feeding ten stock PyG layers
forward+backward — a converted graph is the same payload, so passing there
implies the exported graph is faithful too.
