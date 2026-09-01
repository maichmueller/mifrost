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
| DGL | `ndata["x_ids"]` (`[N, 6]` float) plus convenience `ndata["role_ids"] = x_ids[:, 0].long()` | `edata["edge_attr"]` (`[E, 9]` float) | int64 CPU `dgl.graph((src, dst))` over `num_nodes` nodes |
| Jraph | `nodes` (`[N, 6]` float) | `edges` (`[E, 9]` float) | single-graph `GraphsTuple`, `senders=edge_index[0]`, `receivers=edge_index[1]`, `n_node=[N]`, `n_edge=[E]` |

The edge width is 9, not 3: since the losslessness work every edge carries the
six instance channels (`rel_id_plus_one`, `role`, `sign`, `goal_level`,
`history_dt`, `category`) on top of `kind` / `pos_a` / `pos_b`. Read it from
`edge_channel_names` in the metadata rather than hardcoding it. Note that
`edge_attr[:, 7]` — like `x_ids[:, 4]` — is **signed**; the shift you need is
in `metadata["history_dt_offset"]`.

`metadata` is a plain dict holding everything else:

- Vocabularies when present: `vocab_roles`, `vocab_relations`,
  `vocab_predicates`, `vocab_actions`, `vocab_categories`,
  `vocab_edge_kinds`, `channel_names`, `edge_channel_names`, plus the scalar
  descriptors `num_predicates`, `node_universe`, `atom_expansion` and
  `has_anchor`.
- `node_names` / `object_names` style extras such as `node_names`.
- Per-graph fields: `anchor_index`, `history_dt_offset`,
  `instance_node_indices`.
- Strategy tensors verbatim: `hyperedge_index` / `hyperedge_attr_ids` /
  `num_hyperedges`, `tuple_args` / `tuple_sizes` / `tuple_ptr` /
  `tuple_rel_ids` / `tuple_role_ids` / `tuple_sign_ids` / `tuple_level_ids` /
  `tuple_dt_ids` / `tuple_category_ids` / `tuple_attr_ids` / `num_tuples`,
  and `spd_src` / `spd_dst` / `spd_dist`.

The list above is the *documented* order, not an allowlist: `_collect_metadata`
sweeps every non-core attribute the carrier actually holds, so a field added
to the encoder contract reaches DGL and Jraph consumers without anyone
extending a tuple in `cross_stack.py`. Core node/edge channels (`x`, `x_ids`,
`edge_index`, `edge_attr`, `num_nodes`, `batch`, `ptr`) are the only
exclusions, because they are mapped onto the target library's own storage.

Attributes absent from the input (e.g. `vocab_roles` on
`ObjectFeatureEncoder` output) are simply omitted from the dict.

## Hyperedge Handling per Stack

`HypergraphIncidenceEncoder` output converts everywhere, but the target
representations differ:

- **PyG**: native `HypergraphConv` input via `hyperedge_index`.
- **DGL**: core DGL has no hypergraph primitive, so the adapter additionally
  returns `metadata["hyperedge_bipartite"]`: an int64 graph over `N + M`
  nodes (`M` hyperedges) in which each member node points at its hyperedge
  anchor, whose id is the hyperedge index offset by `N`. The raw
  `[2, sum(sizes)]` membership tensor also rides in
  `metadata["hyperedge_index"]`, and the per-hyperedge
  labels in `metadata["hyperedge_attr_ids"]`, an `[M, 6]` table in `x_ids`
  column order (this replaces the old `[M]` `hyperedge_attr_ids` role vector).
- **Jraph**: the membership tensors ride in `globals` untouched; factor them
  into bipartite senders/receivers yourself if you want incidence-style
  message passing.

`M` used to be inferred as `max(hyperedge_index[1]) + 1`, which undercounted
whenever the last instance had arity 0 and therefore produced a zero-member
hyperedge. The encoder core now guarantees every hyperedge has at least one
member — an arity-0 instance takes the graph's anchor node — so the inferred
count and the declared `num_hyperedges` / `hyperedge_attr_ids.size(0)` agree
by construction. `to_dgl` prefers the declared count and raises `ValueError`
on a disagreement rather than silently mis-sizing the bipartite graph.

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

x_ids = graph.ndata["x_ids"]
columns = []
for c in range(x_ids.size(1)):
    ids = x_ids[:, c].long()
    if c == 4:                               # history_dt is the signed channel
        ids = ids + meta["history_dt_offset"]
    columns.append(torch.nn.Embedding(int(ids.max()) + 2, 8)(ids))
feats = torch.cat(columns, dim=-1)

conv = dglnn.GraphConv(feats.size(-1), 64)   # reverse edges are explicit
out = conv(graph, feats)
```

Column 4 carries the history age with its sign, so it is negative for history
rows and the naive `Embedding(max + 2)` recipe raises `IndexError` the moment
history is supplied. `metadata["history_dt_offset"]` is the shift that makes
it a valid index; see
[the PyG how-to](consume-with-vanilla-gnns.md#column-4-is-the-one-signed-channel).

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
