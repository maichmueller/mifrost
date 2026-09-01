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
| A compact objects-only graph | `ObjectGraphEncoder` | Atoms become fully labeled argument-pair edges via `atom_expansion="clique" \| "chain" \| "star_first"`, plus an anchor node and the tuple instance table |
| Atom co-occurrence structure | `AtomLineGraphEncoder` | Star universe plus fact-fact `line_share` edges bounded by `line_graph_max_degree` |
| Native hypergraph layers | `HypergraphIncidenceEncoder` | Literal instances as hyperedges; emits `hyperedge_index` / `hyperedge_attr_ids` (`[M, 6]`) for PyG `HypergraphConv` |
| Transformer attention biases | `TransformerBiasEncoder` | Objects-only clique projection plus sparse shortest-path fields `spd_src` / `spd_dst` / `spd_dist` (tune with `spd_max_hops`) |

`TupleTensorEncoder` additionally exposes a padding-free CSR tuple view
(`tuple_args` / `tuple_ptr` plus one id vector per node channel, also stacked
as `tuple_attr_ids`) — see its class docstring. `ObjectGraphEncoder` and
`TransformerBiasEncoder` carry the same tuple channels unconditionally,
because a pairwise projection cannot group the arguments of an arity >= 3
instance on its own.

`TransformerBiasEncoder` spd distances are even bipartite hop counts: two
objects sharing one fact sit at distance 2, and `spd_max_hops` must be >= 2.

!!! warning "Objects-only facades need an edge-conditioned layer"
    `ObjectGraphEncoder` and `TransformerBiasEncoder` reify no fact, goal,
    subgoal or history node, so their `x_ids` is *information-free*: on
    `blocks/smedium` it has 2 distinct rows for the whole graph (3 once
    actions are supplied), and the only non-zero column is `role`. All of the
    state content is on `edge_attr` and in the tuple instance table. Embedding
    `x_ids` alone — the recipe below, applied naively — hands your model a
    near-constant feature matrix. See
    [Objects-only facades: the content is on the edges](#objects-only-facades-the-content-is-on-the-edges).

## Channel Contract

Every facade encodes to the same carrier layout:

- `x_ids`: `[N, 6]` float tensor of integer node channels
- `edge_index`: `[2, E]`
- `edge_attr`: `[E, 9]` float tensor of integer edge channels

There is deliberately no `x` — see [There is no `data.x`](#there-is-no-datax).

Node channels, in column order (`channel_names`):

| Column | Name | Values |
| --- | --- | --- |
| 0 | `role` | index into `vocab_roles` = `["object", "fact", "goal", "subgoal", "history", "action", "anchor"]` |
| 1 | `relation_id_plus_one` | relation id + 1 over the unified space (see below); `0` means "none" |
| 2 | `sign` | `0` = positive literal, `1` = negated literal |
| 3 | `goal_level` | small int goal/subgoal layer (`0` = top goal) |
| 4 | `history_dt` | history age — **signed**, negative for history rows |
| 5 | `category` | index into `vocab_categories` = `["static", "fluent", "derived", "action"]` |

Edge channels, in column order (`edge_channel_names`):

| Column | Name | Values |
| --- | --- | --- |
| 0 | `kind` | index into `vocab_edge_kinds` (13 entries, listed below) |
| 1 | `pos_a` | argument position of the source endpoint |
| 2 | `pos_b` | argument position of the target endpoint |
| 3 | `rel_id_plus_one` | relation id + 1 of the instance this edge was derived from; `0` = none |
| 4 | `role` | that instance's role id |
| 5 | `sign` | that instance's sign |
| 6 | `goal_level` | that instance's goal level |
| 7 | `history_dt` | that instance's history age — **signed** |
| 8 | `category` | that instance's category |

`vocab_edge_kinds` is

```python
["arg_fwd", "arg_bwd", "clique_fwd", "clique_bwd", "chain_fwd", "chain_bwd",
 "star_first_fwd", "star_first_bwd", "nullary_self", "action_fwd",
 "action_bwd", "line_share", "unary_self"]
```

Columns 3..8 apply to every edge kind in every node universe, so an
objects-only projection edge is as self-describing as a star edge. The single
exception is `line_share`, whose columns 3..8 are all zero: it is not derived
from one instance but connects two reified fact nodes that already carry their
own six channels, so nothing is lost.

A node row is only *populated* when the encoder reifies a node for the
instance. Objects carry role `0` and zeros everywhere else — they have no
labels of their own — and in the objects-only universe there are no other
rows except the anchor, so columns 1..5 stay zero. Columns 1..5 of `x_ids`
therefore say something only in the star-shaped universes; see
[Objects-only facades](#objects-only-facades-the-content-is-on-the-edges).

With the default `include_metadata=True`, the graph also carries plain Python
metadata attributes: `vocab_roles`, `vocab_relations`, `vocab_predicates`,
`vocab_actions`, `vocab_edge_kinds`, `vocab_categories`, `num_predicates`,
`node_universe`, `atom_expansion`, `has_anchor`, `channel_names`,
`edge_channel_names`, and (when `export_node_names=True`) `node_names` /
`object_names`. Pass `include_metadata=False` to emit tensors only — the
tensors still describe the state completely, but you lose the ability to
decode any id back to a name and to split the relation id space. The
vocabularies stay aligned across states of one problem, so per-channel
embedding sizes can be sized once from `max + 2` headroom.

### Column 1 is a unified relation space

Predicates and action schemas share one id space:

```text
relation_id(predicate p)      = p                 for p in 0 .. P - 1
relation_id(action schema a)  = P + a             P = data.num_predicates
```

`vocab_relations` names the whole space and equals
`vocab_predicates + vocab_actions`, so one embedding table of size
`len(vocab_relations) + 1` covers nodes of every role. On `blocks/smedium`,
`num_predicates` is 7 with `vocab_predicates == ["clear", "handempty",
"holding", "on", "ontable", "number", "object"]` and `vocab_actions ==
["pick-up", "put-down", "stack", "unstack"]`; the fact `(on a b)` gets
`relation_id_plus_one == 4` and the grounded action `@(unstack a b)` gets
`11` — schema id 3 shifted by `P`, decoding through
`vocab_relations[10] == "unstack"`.

Sizing that channel by `max(len(vocab_predicates), len(action schemas)) + 1`
was **wrong** and no amount of headroom fixed it: on `blocks/smedium` that
formula gives 8, and predicate id 3 (`on`) and action-schema id 3 (`unstack`)
still shared embedding row 3. Under the unified space the two are ids 3 and
10, and the correct size is `len(vocab_relations) + 1 == 12`. To recover the
old split without string compares:

```python
rel = data.x_ids[:, 1].long() - 1              # -1 == "none"
is_action = rel >= data.num_predicates
schema_id = rel[is_action] - data.num_predicates
```

`tuple_rel_ids` and `hyperedge_attr_ids[:, 1]` index the same unified space
(`tuple_rel_ids` stores the **raw** relation id, with `-1` for "none", while
`x_ids`, `edge_attr[:, 3]` and `hyperedge_attr_ids[:, 1]` all store id + 1).

### Column 4 is the one signed channel

`x_ids[:, 4]` and `edge_attr[:, 7]` carry the history age exactly as supplied,
so history rows are **negative**. Every other channel is a non-negative id.
The naive `nn.Embedding` recipe therefore raises
`IndexError: index out of range in self` the moment history is present.

Each graph exports the shift you need:

| Field | Shape | Use |
| --- | --- | --- |
| `history_dt_offset` | `[1]`, `[B]` when batched | `ids = x_ids[:, 4].long() + history_dt_offset` (batched: `+ history_dt_offset[batch]`) |

It is `-min(0, min emitted dt)`, so it is `0` when no history is supplied, and
the inverse is `dt = ids - history_dt_offset`. With two history steps on
`blocks/smedium` the raw column ranges over `[-2, 0]`, `history_dt_offset` is
`2`, and the shifted ids range over `[0, 2]` — an embedding of size 4.
`history_dt_offset` is a tensor field, not metadata, so it survives
`include_metadata=False`.

The same offset shifts `edge_attr[:, 7]` and `tuple_dt_ids` (equivalently
`tuple_attr_ids[:, 4]`). On `ObjectGraphEncoder` and `TransformerBiasEncoder`
those are the *only* places the history age appears — `x_ids[:, 4]` stays at
`[0, 0]` there, so shifting it is a no-op. See
[Objects-only facades](#objects-only-facades-the-content-is-on-the-edges).

### Per-graph auxiliary fields

| Field | Shape | Present when | Meaning |
| --- | --- | --- | --- |
| `history_dt_offset` | `[1]` / `[B]` | always | shift for column 4, above |
| `anchor_index` | `[1]` / `[B]` | an anchor node was emitted | node index of the auxiliary `<nullary>` anchor (globally offset in a batch) |
| `has_anchor` | int metadata | `include_metadata=True` | `1` iff an anchor node was emitted |
| `instance_node_indices` | `[M]` | instance tables are on | reified node index per instance row |

The anchor is one extra node, role `anchor` (id 6), appended as the **last**
node of the graph so no pre-existing index shifts. It is emitted exactly when
`node_universe == "objects_only"` or hyperedge incidence is on — i.e. for
`ObjectGraphEncoder`, `TransformerBiasEncoder`, and
`HypergraphIncidenceEncoder`. It stands in for the empty argument list: an
arity-0 instance hangs its `nullary_self` loop and its hyperedge membership
there. It is **not** an object — absent from `object_names`, never in
`spd_*` — so object-role pooling (`x_ids[:, 0] == 0`) already excludes it.
`anchor_index` is present exactly when an anchor is; the spec's `-1` sentinel
is a single-graph convenience:

```python
anchor = int(data.anchor_index[0]) if bool(getattr(data, "has_anchor", 0)) else -1
```

In a batch, `anchor_index` is `[B]` of global node indices and equals
`ptr[1:] - 1`, since the anchor is always its graph's last node.

`instance_node_indices` reports where each instance row was reified. In the
star universe that is a real node index for every row. In `objects_only` it is
`-1` for everything except arity-0 rows (which point at the anchor) and action
rows (actions are reified in every universe). **That `-1` is not batch-safe**:
the field increments by the node offset, so a `-1` in graph *k* collates to
`offset_k - 1`. Mask by the instance itself rather than by sign — a row is
meaningful in `objects_only` exactly when its tuple slot is empty or its
`tuple_role_ids` entry is `5` (action).

### Strategy extras

| Field | Shape | Emitted by |
| --- | --- | --- |
| `hyperedge_index` | `[2, sum(sizes)]` — row 0 member node ids, row 1 hyperedge ids | `HypergraphIncidenceEncoder` |
| `hyperedge_attr_ids` | `[M, 6]`, same column order as `x_ids` | `HypergraphIncidenceEncoder` |
| `num_hyperedges` | `[1]` / `[B]` | `HypergraphIncidenceEncoder` |
| `tuple_args` / `tuple_sizes` / `tuple_ptr` | flattened argument node ids, per-row arity, CSR offsets derived from the sizes | `TupleTensorEncoder`, `ObjectGraphEncoder`, `TransformerBiasEncoder` |
| `tuple_rel_ids`, `tuple_role_ids`, `tuple_sign_ids`, `tuple_level_ids`, `tuple_dt_ids`, `tuple_category_ids` | `[T]` each | same |
| `tuple_attr_ids` | `[T, 6]` — the six above stacked in `x_ids` column order (so column 1 is `tuple_rel_ids + 1`) | same |
| `num_tuples` | `[1]` / `[B]` | same |
| `spd_src` / `spd_dst` / `spd_dist` | `[K]` each | `TransformerBiasEncoder` |

`tuple_ptr` is derived from `tuple_sizes` on every path, so
`tuple_ptr.numel() - 1 == tuple_rel_ids.numel()` holds after native batching
and after a manual `Batch.from_data_list` alike.

**Directed by design.** Derived graphs carry explicit reverse edges labeled
`arg_bwd`, `clique_bwd`, and so on rather than relying on conversion-time
mirroring: the schema flag `include_reverse_edges` disables mirroring for this
family, so converting single graphs and re-batching them by hand agrees
exactly with `encode_batch_pyg`. (Other families, such as `ColorEncoder`,
intentionally keep undirected mirroring.) `line_share` shortcuts honour the
same flag. Because the graph is directed, feed it to directed-capable layers
or rely on the explicit reverse edges for symmetric message passing.

## Embedding Recipe

### There is no `data.x`

`x_ids` is the only node-feature source these encoders emit. The carrier does
not define `x` at all, so `data.x` is `None` and
`conv(data.x, data.edge_index)` raises immediately on any stock layer
(`AttributeError: 'NoneType' object has no attribute 'size'` from `GCNConv`,
`TypeError: 'NoneType' object is not subscriptable` from `GATv2Conv`).

That is the intended behaviour, and it is a change. The carrier used to ship
an all-zero `[N, 6]` float `x` as a `num_nodes` placeholder, so the PyG reflex
`conv(data.x, data.edge_index)` ran, trained, and converged — on nothing. A
zero-filled feature matrix carries no information: on `blocks/smedium`, a
`GCNConv` over 13 nodes returns **1 distinct output row**, identical for every
node, where the same layer over `x_ids` returns 10. The entire encoding was
discarded, silently, with no error and no warning. Since this family's contract
is that a loss must be explicit rather than silent, the placeholder is gone and
the reflex now fails loudly instead. Embed `x_ids`, as below.

`num_nodes` is set explicitly on the carrier, so nothing downstream depended on
`x` to size the graph; batching, `ptr`, and pooling are unaffected.

### Embedding the channels

This is the recipe for the star-shaped facades — `StarGraphEncoder`,
`AtomLineGraphEncoder`, `HypergraphIncidenceEncoder`, `TupleTensorEncoder` —
whose `x_ids` rows carry real labels because every literal instance has a node.
For `ObjectGraphEncoder` and `TransformerBiasEncoder`, read
[the next subsection first](#objects-only-facades-the-content-is-on-the-edges).

Embed each channel separately, concatenate, then feed any message-passing
layer. Take the edge-feature width from the tensor rather than hardcoding it —
it is 9 today and `edge_channel_names` is the authority:

```python
import torch
import torch.nn as tnn
from torch_geometric.nn import GATv2Conv

data = encoder.encode_pyg(state)          # DerivedGraphData
num_channels = data.x_ids.size(1)
HISTORY_DT = 4                            # the one signed channel

columns = []
for c in range(num_channels):
    ids = data.x_ids[:, c].long()
    if c == HISTORY_DT:
        ids = ids + data.history_dt_offset  # shift the signed channel
    embedding = tnn.Embedding(int(ids.max().item()) + 2, 8)
    columns.append(embedding(ids))
feats = torch.cat(columns, dim=-1)

conv = GATv2Conv(feats.size(-1), 64, edge_dim=data.edge_attr.size(1))
out = conv(feats, data.edge_index, data.edge_attr.float())

# Read out over object-role rows only (role channel == 0).
# The anchor node has role 6, so this mask already excludes it.
object_mask = data.x_ids[:, 0] == 0
graph_embedding = out[object_mask].mean(dim=0)
```

In batch mode the shift is per graph:

```python
ids = batch.x_ids[:, HISTORY_DT].long() + batch.history_dt_offset[batch.batch]
```

Sizing an embedding from `int(ids.max()) + 2` is a convenience for one graph.
For a model that must accept every state of a problem, size from the
vocabularies instead: `len(vocab_relations) + 1` for column 1,
`len(vocab_roles)` for column 0, `len(vocab_categories)` for column 5, and
your own bound on goal depth and history length for columns 3 and 4.

Layers without edge features (GCN/SAGE/GIN/PNA/...) simply ignore
`edge_attr`; layers with an `edge_dim` (GAT/GATv2/TransformerConv/NNConv/
CGConv) consume it as-is. For `HypergraphIncidenceEncoder` output, pass
`hyperedge_index` to `torch_geometric.nn.HypergraphConv` — with
`use_attention=True`, project `hyperedge_attr_ids` to `in_channels` first. For
`TransformerBiasEncoder` output, use `spd_src` / `spd_dst` / `spd_dist` to
build attention-bias terms over the clique-projected object nodes.

### Objects-only facades: the content is on the edges

`ObjectGraphEncoder` and `TransformerBiasEncoder` reify no fact, goal, subgoal
or history node — that is the whole point of the objects-only universe — so
there is no node row for any of those labels to live on. Their `x_ids` is
consequently **information-free**: every object row is all zeros, the anchor
row is `[6, 0, 0, 0, 0, 0]`, and no other row exists. Measured on
`blocks/smedium`:

| Facade | `N` | distinct `x_ids` rows | non-zero `x_ids` columns | non-zero `edge_attr` columns |
| --- | ---: | ---: | --- | --- |
| `StarGraphEncoder` | 13 | 7 | `0, 1, 5` | `0, 1, 2, 3, 4, 8` |
| `ObjectGraphEncoder` | 4 | 2 | `0` | `0, 1, 2, 3, 4, 8` |
| `TransformerBiasEncoder` | 4 | 2 | `0` | `0, 1, 2, 3, 4, 8` |

Two distinct feature rows for the entire graph. Run the recipe above on that
and you get a near-constant node-feature matrix: three distinct rows out of
five even in the full-fidelity encoding with goals, subgoals, history and
actions supplied. Nothing is lost by the *encoder* — see
[Information Content](#information-content), both facades are lossless as
emitted — but a consumer that reads only `x_ids` throws all of it away.

Two consequences:

- **You need an edge-conditioned layer.** `GCNConv` / `SAGEConv` / `GIN` /
  `PNA` ignore `edge_attr` entirely and will see nothing but topology. Use
  `GATv2Conv`, `TransformerConv`, `NNConv` or `CGConv` with
  `edge_dim=data.edge_attr.size(1)`, or consume the tuple instance table
  directly.
- **`history_dt_offset` applies to the edges, not to `x_ids`.** The signed
  history age never reaches `x_ids[:, 4]` in this universe: with two history
  steps on `blocks/smedium` that column stays at `[0, 0]` while
  `edge_attr[:, 7]` and `tuple_dt_ids` both range over `[-2, 0]` and
  `history_dt_offset` is `2`. Shift **column 7 of `edge_attr`** and
  **`tuple_dt_ids`** (equivalently `tuple_attr_ids[:, 4]`); shifting
  `x_ids[:, 4]` here is a no-op.

An edge-conditioned pass, verified end to end on the full-fidelity
`blocks/smedium` encoding (5 nodes, 23 edges):

```python
import torch
import torch.nn as tnn
from torch_geometric.nn import GATv2Conv

data = encoder.encode_pyg(state)      # ObjectGraphEncoder / TransformerBiasEncoder

# x_ids carries only the role column here, so give each node a role embedding
# and let the edges carry the state content.
node_feats = tnn.Embedding(len(data.vocab_roles), 32)(data.x_ids[:, 0].long())

# Embed every edge channel. Column 7 is the signed one in edge_attr.
EDGE_HISTORY_DT = 7
edge_columns = []
for c in range(data.edge_attr.size(1)):
    ids = data.edge_attr[:, c].long()
    if c == EDGE_HISTORY_DT:
        ids = ids + data.history_dt_offset
    edge_columns.append(tnn.Embedding(int(ids.max().item()) + 2, 8)(ids))
edge_feats = torch.cat(edge_columns, dim=-1)          # [E, 72]

conv = GATv2Conv(node_feats.size(-1), 64, edge_dim=edge_feats.size(-1))
out = conv(node_feats, data.edge_index, edge_feats)   # [N, 64]

object_mask = data.x_ids[:, 0] == 0        # role 0; excludes the anchor (role 6)
graph_embedding = out[object_mask].mean(dim=0)
```

`edge_attr` has 19 distinct rows out of 23 on that encoding, so `edge_feats`
does too, and the layer returns 5 distinct output rows out of 5 — the state is
back. Contrast the `x_ids`-only recipe on the same graph: 3 distinct feature
rows out of 5, and none of the history, sign or goal-level information at all.

The alternative is to skip the topology and read the instance table, which
`objects_only` always emits. `tuple_attr_ids` is `[T, 6]` in `x_ids` column
order, so the same per-channel embedding applies, with the same shift on
column 4:

```python
rows = data.tuple_attr_ids.long().clone()
rows[:, 4] += data.history_dt_offset       # column 4 is the signed one here too
labels = torch.cat(
    [tnn.Embedding(int(rows[:, c].max().item()) + 2, 8)(rows[:, c])
     for c in range(rows.size(1))],
    dim=-1,
)                                          # [T, 48]

# Argument object node ids of instance i, in argument order:
#   data.tuple_args[data.tuple_ptr[i]:data.tuple_ptr[i + 1]]
sizes = data.tuple_ptr[1:] - data.tuple_ptr[:-1]
row_of_arg = torch.repeat_interleave(torch.arange(sizes.numel()), sizes)
```

`row_of_arg` maps each entry of `tuple_args` to its instance, ready for a
`scatter` pool of whatever per-object representation you have. Note that
object nodes in this universe carry no features of their own — object identity
is positional, not featural — so pool `out` from the graph pass above, or your
own object embeddings, rather than the role embedding.

## Invariants You Can Rely On

These hold for every facade, every backend, and every state of a problem.

1. **No empty hyperedges.**
   `hyperedge_index[1].max() + 1 == hyperedge_attr_ids.size(0)`. An arity-0
   instance takes the anchor as its single member, so `HypergraphConv`'s
   inference of `num_edges` — which is exactly `hyperedge_index[1].max() + 1`
   — always agrees with the attribute table. This previously failed: with
   `(handempty)` as the goal on `blocks/smedium` the encoder records 9
   instances, and dropping the anchor memberships (the old zero-member
   behaviour) makes PyG infer 8 against 9 attribute rows.
2. **Every instance leaves a trace in every universe.** Arity >= 2 gets its
   projection or star edges, arity 1 a `unary_self` loop on its argument,
   arity 0 a `nullary_self` loop on the anchor.
3. **Instances that differ in any channel produce different edge rows.**
   Relation, role, sign, goal level, history age, and category all ride on
   columns 3..8, so a goal literal is never byte-identical to a state fact.
4. **The graph view is a set; the instance tables are multisets.** Passing the
   same literal twice yields one node and one set of edges, but two tuple rows
   and two hyperedges. On `blocks/smedium`, `goals=[lit, lit]` leaves the star
   view at 12 nodes / 23 edges while the tuple table goes from 9 rows to 10.
5. **Node names are injective.** Goal suffixes render `[g]`, `[sg]`, `[ssg]`,
   `[sssg]`, then `[sg4]`, `[sg5]`, ... so the same atom at different subgoal
   levels never collides.
6. **The anchor is not an object.** Absent from `object_names`, never in
   `spd_*`, always the last node.
7. **Native batching equals manual re-batching.** `encode_batch_pyg([...])`
   agrees with `Batch.from_data_list([enc(s) for s in ...])` on every field,
   including `anchor_index`, `history_dt_offset`, `instance_node_indices`,
   `hyperedge_attr_ids`, and the `tuple_*` vectors.

## Information Content

Every derived encoder carries the full literal-instance content of the state.
The unit is the instance tuple
`(role, relation_id, arguments, sign, goal_level, history_dt, category)`, and
the per-encoder verdicts, the exact recovery procedures, and the residual
losses are stated in
[Derived Encoding Strategies § Information Content](../explanation/derived-encoding-strategies.md#information-content).
The short version:

- **Star / line-graph views** recover the instance *set* from `x_ids`,
  `edge_index`, and `edge_attr` alone. For each non-object, non-anchor node,
  channels 1..5 give the labels and its outgoing `arg_fwd` (or `action_fwd`)
  edges give the arguments, sorted by `pos_a`.
- **Hypergraph and tuple views** recover the instance *multiset*, arguments in
  order and repeats included.
- **Objects-only views** recover the multiset from the tuple table, which is
  why `objects_only` forces those channels on. The pairwise projection *by
  itself* cannot say which pairs of an arity >= 3 instance belong together.
  Note the reading order: nothing here is on `x_ids`, so a consumer must go to
  `edge_attr` or the tuple table —
  [Objects-only facades](#objects-only-facades-the-content-is-on-the-edges)
  covers what that means for a model. Lossless *as emitted* is not the same as
  reachable through any recipe.
- **`include_metadata=False`** keeps every tensor but drops the
  vocabularies, so ids can no longer be decoded to names.

Here is the star-view recovery, in full:

```python
import collections

def instances(data):
    """Recover the literal-instance set from a star-shaped derived graph."""
    x, ei, ea = data.x_ids.long(), data.edge_index, data.edge_attr.long()
    args = collections.defaultdict(dict)
    for e in range(ei.size(1)):
        if int(ea[e, 0]) in (0, 9):            # arg_fwd, action_fwd
            args[int(ei[0, e])][int(ea[e, 1])] = int(ei[1, e])
    out = []
    for v in range(x.size(0)):
        role = int(x[v, 0])
        if role in (0, 6):                     # object, anchor
            continue
        slots = args.get(v, {})
        out.append((
            role,
            int(x[v, 1]) - 1,                  # relation id, -1 == none
            tuple(slots[k] for k in sorted(slots)),
            int(x[v, 2]), int(x[v, 3]), int(x[v, 4]), int(x[v, 5]),
        ))
    return collections.Counter(out)
```

On `blocks/smedium` with negated goals, two subgoal layers, two history steps,
one grounded action, and an `(on a a)` goal, this reproduces the encoder's own
tuple table exactly — 16 instances, argument order and repeats included.

## Batching

Use `encode_batch_pyg([...])`; collation is handled natively and returns a
standard PyG `Batch` with `batch` / `ptr` vectors, so `global_mean_pool` and
friends work unchanged. Batch is the fast lane for encoding N states in one
call: one native batch assembly replaces N Python-level encodes, and
`scripts/benchmark_derived_encoders.py` prints a batch-vs-singles audit
(`--problems large --batch-sizes 32,256`). Typical advantage is 1.1-1.8x;
on the largest graphs with edge-heavy views (atom line graph at n=256) the
per-entry cost is on par with singles (~2 us/entry) while still saving the
N-1 call round-trips — repeated states are prepared once and reused.

```python
batch = encoder.encode_batch_pyg([state_a, state_b])
out = conv(embed(batch.x_ids.long()).flatten(1), batch.edge_index)
graph_out = global_mean_pool(out, batch.batch)
```

Per-graph scalars stack rather than collapse: `history_dt_offset` and
`anchor_index` become `[B]` vectors, indexed with `batch.batch` (for the
former) or read positionally (for the latter). Node-index fields
(`anchor_index`, `instance_node_indices`, `tuple_args`, and row 0 of
`hyperedge_index`) carry node-offset increment rules; label fields
(`hyperedge_attr_ids`, every `tuple_*_ids`, `history_dt_offset`) do not
increment.

Shared vocabulary metadata is normalized back to single lists after batching.
If you re-batch singles manually (`Batch.from_data_list([...])`), PyG keeps
per-graph copies of those list and scalar attributes;
`normalize_derived_graph_batch_metadata(batch)` collapses them back to one
shared value — the vocabularies including `vocab_relations` / `vocab_actions`,
and the batch-invariant scalars `node_universe`, `atom_expansion`,
`num_predicates`, `has_anchor` and `hyperedge_note`.

The *per-graph* name lists are not shared and stay nested one level, so
`node_names[k]` and `object_names[k]` are graph `k`'s own lists — on both the
native and the `Batch.from_data_list` path. A two-state batch of
`blocks/smedium` gives `object_names == [['a', 'b', 'c'], ['a', 'b', 'c']]`
either way, not one flattened list.

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

## Upgrading: What Changed Incompatibly

The losslessness work changed the carrier in ways that break code written
against the previous contract:

| Was | Is now |
| --- | --- |
| `x` is an all-zero `[N, 6]` float placeholder, so `conv(data.x, ...)` runs and silently returns a constant graph | `x` is **removed**; `data.x` is `None` and stock layers raise. Embed `x_ids` |
| `edge_attr` is `[E, 3]` | `edge_attr` is `[E, 9]`; read the width from `edge_attr.size(1)` / `edge_channel_names`, and pass it as `edge_dim` |
| `channel_names[1] == "predicate_id_plus_one"` | `channel_names[1] == "relation_id_plus_one"` |
| `x_ids[:, 1]` holds a predicate id for facts and an *action-schema* id for actions, in colliding spaces | one unified relation space; action schemas are shifted by `num_predicates` |
| `hyperedge_role_ids`, a `[M]` role vector | `hyperedge_attr_ids`, an `[M, 6]` table with all six channels; `hyperedge_role_ids` is **removed** |
| `vocab_roles` has 6 entries | 7 entries; `anchor` = 6 is appended (0..5 unchanged) |
| `vocab_edge_kinds` has 12 entries | 13 entries; `unary_self` = 12 is appended (0..11 unchanged) |
| `vocab_categories` has 3 entries | 4 entries; `action` = 3 is appended, and action nodes/edges/rows now report category 3 instead of 0 (`static`) |
| `objects_only` materializes no auxiliary node | one extra `anchor` node, always last; `x_ids[:, 0]` now contains role 6 |
| `objects_only` may omit tuple channels | `objects_only` always emits them |
| arity-0 hyperedges have zero members | they have the anchor as their single member, so `M` and PyG's inference agree |
| `line_share` always emits both directions | it honours `include_reverse_edges` |
| a repeated instance duplicates its graph edges | it does not; only the instance tables grow |
| goal levels >= 3 all render `[sssg]` | level `L >= 4` renders `[sg<L>]`, keeping `node_names` injective |

New graph attributes: `vocab_relations`, `vocab_actions`, `num_predicates`,
`has_anchor`. New fields: `anchor_index`, `history_dt_offset`,
`instance_node_indices`, `tuple_sign_ids` / `tuple_level_ids` /
`tuple_dt_ids` / `tuple_category_ids`, the stacked `tuple_attr_ids`, and the
counts `num_tuples` / `num_hyperedges`.

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
  anchors, a hexagon the `<nullary>` anchor, pink pentagons the auxiliary
  hyperedge anchors.
- Edges are styled per kind: solid gray argument edges (reverse directions
  are hidden by default since the forward edge already carries the position
  label), colored projections for clique/chain/star-first views, dashed cyan
  `line_share` shortcuts, dotted pink memberships.
- `include_hyperedges`, `include_reverse_edges`, `include_line_shares`, and
  `include_self_loops` on `to_networkx` control exactly which structural
  pieces are materialized into the NetworkX graph; `edge_labels=True` on
  `draw` annotates each arrow with its kind and positions. Note that the
  `nullary_self` and `unary_self` loops that carry arity-0 and arity-1
  instances are self-loops, so `include_self_loops=False` hides them — and
  with them the only trace some instances leave in an objects-only view.

```python
data = encoder.encode_pyg(state)
encoder.draw(data)                  # matplotlib axes with legend
print(encoder.summarize(data))      # "roles: ... ; kinds: ..."
```

`examples/encoders/derived_graph_example.py` renders all five graph-shaped
strategies side by side into `derived_graph_example.png`.
