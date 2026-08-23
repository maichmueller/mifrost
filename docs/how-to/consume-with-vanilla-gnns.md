# How to Consume Derived Graphs with Vanilla GNNs

The derived-graph encoders turn a planning problem's objects and atoms into
plain homogeneous PyTorch Geometric graphs. No heterogeneous machinery, custom
collation, or bespoke layer is required: the output is a stock `Data` object
whose node and edge features are integer-id channels that you embed with
`nn.Embedding` before message passing.

All facades take a *problem* (`pymimir.Problem` or a `pytyr.PlanningTask`),
not a bare `Domain`, because object tables are problem-scoped.

## Which Facade Should I Use?

| If you want ... | Use | Notes |
| --- | --- | --- |
| Reified atom nodes (star view) | `StarGraphEncoder` | Object nodes plus one node per grounded literal/action anchor; position-aware star edges |
| A compact objects-only graph | `ObjectGraphEncoder` | Atoms become directed argument-pair edges via `atom_expansion="clique" \| "chain" \| "star_first"` |
| Atom co-occurrence structure | `AtomLineGraphEncoder` | Star universe plus fact-fact `line_share` edges bounded by `line_graph_max_degree` |
| Native hypergraph layers | `HypergraphIncidenceEncoder` | Literal instances as hyperedges; emits `hyperedge_index` / `hyperedge_attr_ids` for PyG `HypergraphConv` |
| Transformer attention biases | `TransformerBiasEncoder` | Objects-only clique projection plus sparse shortest-path fields `spd_src` / `spd_dst` / `spd_dist` (tune with `spd_max_hops`) |

`TupleTensorEncoder` additionally exposes a padded-free CSR tuple view
(`tuple_args` / `tuple_ptr` / `tuple_rel_ids` / `tuple_role_ids`) — see its
class docstring.

`TransformerBiasEncoder` spd distances are even bipartite hop counts: two
objects sharing one fact sit at distance 2, and `spd_max_hops` must be >= 2.

## Channel Contract

Every facade encodes to the same carrier layout:

- `x_ids`: `[N, 6]` float tensor of integer node channels
- `edge_index`: `[2, E]`
- `edge_attr`: `[E, 3]` float tensor of integer edge channels

Node channels, in column order:

| Column | Name | Values |
| --- | --- | --- |
| 0 | `role` | index into `vocab_roles` = `["object", "fact", "goal", "subgoal", "history", "action"]` |
| 1 | `predicate_id_plus_one` | predicate id + 1; `0` means "none" |
| 2 | `sign` | `0` = positive literal, `1` = negated literal |
| 3 | `goal_level` | small int goal/subgoal layer |
| 4 | `history_dt` | small int history age |
| 5 | `category` | index into `vocab_categories` = `["static", "fluent", "derived"]` |

Edge channels, in column order:

| Column | Name | Values |
| --- | --- | --- |
| 0 | `kind` | index into `vocab_edge_kinds` = `["arg_fwd", "arg_bwd", "clique_fwd", "clique_bwd", "chain_fwd", "chain_bwd", "star_first_fwd", "star_first_bwd", "nullary_self", "action_fwd", "action_bwd", "line_share"]` |
| 1 | `pos_a` | argument position of the source endpoint |
| 2 | `pos_b` | argument position of the target endpoint |

With the default `include_metadata=True`, the graph also carries plain Python
metadata attributes: `vocab_roles`, `vocab_predicates`, `vocab_edge_kinds`,
`vocab_categories`, `channel_names`, `edge_channel_names`, and (when
`export_node_names=True`) `node_names` / `object_names`. Pass
`include_metadata=False` to emit tensors only. The vocabularies stay aligned
across states of one problem, so per-channel embedding sizes can be sized once
from `max + 2` headroom.

One caveat on column 1: for nodes whose role is `action`, it carries the
**action-schema id + 1** against the action vocabulary — not a predicate id.
Sizing that channel's embedding from `vocab_predicates` alone can overflow on
domains with many action schemas; size it by
`max(len(vocab_predicates), len(action schemas)) + 1` or filter action-role
rows out before embedding.

**Directed by design.** Derived graphs carry explicit reverse edges labeled
`arg_bwd`, `clique_bwd`, and so on rather than relying on conversion-time
mirroring: the schema flag `include_reverse_edges` disables mirroring for this
family, so converting single graphs and re-batching them by hand agrees
exactly with `encode_batch_pyg`. (Other families, such as `ColorEncoder`,
intentionally keep undirected mirroring.) Because the graph is directed, feed
it to directed-capable layers or rely on the explicit reverse edges for
symmetric message passing.

## Embedding Recipe

Embed each channel separately, concatenate, then feed any message-passing
layer. Attention layers consume `edge_attr` directly via `edge_dim=3`:

```python
import torch
import torch.nn as tnn
from torch_geometric.nn import GATv2Conv

data = encoder.encode_pyg(state)          # DerivedGraphData
num_channels = data.x_ids.size(1)

embeddings = tnn.ModuleList(
    [
        tnn.Embedding(int(data.x_ids[:, c].max().item()) + 2, 8)
        for c in range(num_channels)
    ]
)
feats = torch.cat(
    [embeddings[c](data.x_ids[:, c].long()) for c in range(num_channels)],
    dim=-1,
)

conv = GATv2Conv(feats.size(-1), 64, edge_dim=3)
out = conv(feats, data.edge_index, data.edge_attr.float())

# Read out over object-role rows only (role channel == 0)
object_mask = data.x_ids[:, 0] == 0
graph_embedding = out[object_mask].mean(dim=0)
```

Layers without edge features (GCN/SAGE/GIN/PNA/...) simply ignore
`edge_attr`; layers with `edge_dim=3` (GAT/GATv2/TransformerConv/NNConv/CGConv)
consume it as-is. For `HypergraphIncidenceEncoder` output, pass
`hyperedge_index` to `torch_geometric.nn.HypergraphConv`. For
`TransformerBiasEncoder` output, use `spd_src` / `spd_dst` / `spd_dist` to
build attention-bias terms over the clique-projected object nodes.

## Batching

Use `encode_batch_pyg([...])`; collation is handled natively and returns a
standard PyG `Batch` with `batch` / `ptr` vectors, so `global_mean_pool` and
friends work unchanged:

```python
batch = encoder.encode_batch_pyg([state_a, state_b])
out = conv(embed(batch.x_ids.long()).flatten(1), batch.edge_index)
graph_out = global_mean_pool(out, batch.batch)
```

Shared vocabulary metadata is normalized back to single lists after batching.
If you re-batch singles manually (`Batch.from_data_list([...])`), PyG keeps
per-graph copies of those list attributes;
`normalize_derived_graph_batch_metadata(batch)` collapses them back to one
shared list. Strategy extras (`hyperedge_index`, `spd_*`, tuple views) carry
increment rules so offsets stay correct across concatenated graphs.

## Streaming

Every facade provides `.stream()`, mirroring the `ColorEncoder` stream API:

```python
stream = encoder.stream()
stream.append(state_a)
stream.append(state_b)
batch = stream.flush_pyg()
```

## Backend Selection

Pymimir problems and pytyr planning tasks are auto-detected. Pass
`backend="pymimir"` or `backend="pytyr"` to override detection explicitly.
Outputs from different backends share the same channel contract and
vocabulary semantics.

## Compatibility Guarantee

`tests/encoding/test_gnn_conformance.py` is the executable proof of this
page's contract: every facade is fed through forward+backward passes of ten
stock PyG layers (plus `HypergraphConv` on the hypergraph output), including
the embedding path and pooled batch training. A runnable walkthrough lives in
`examples/encoders/derived_graph_example.py`.

## Visualizing Derived Encodings

Every facade ships `to_networkx(data)`, `draw(data, ...)`, and
`summarize(data)` helpers that decode the integer channels back into names:

- Nodes are shaped and colored by role: blue circles are objects, orange
  circles facts, green/red diamonds goals/subgoals, squares history/action
  anchors, pink pentagons the auxiliary hyperedge anchors.
- Edges are styled per kind: solid gray argument edges (reverse directions
  are hidden by default since the forward edge already carries the position
  label), colored projections for clique/chain/star-first views, dashed cyan
  `line_share` shortcuts, dotted pink memberships.
- `include_hyperedges`, `include_reverse_edges`, `include_line_shares`, and
  `include_self_loops` on `to_networkx` control exactly which structural
  pieces are materialized into the NetworkX graph; `edge_labels=True` on
  `draw` annotates each arrow with its kind and positions.

```python
data = encoder.encode_pyg(state)
encoder.draw(data)                  # matplotlib axes with legend
print(encoder.summarize(data))      # "roles: ... ; kinds: ..."
```

`examples/encoders/derived_graph_example.py` renders all five graph-shaped
strategies side by side into `derived_graph_example.png`.
