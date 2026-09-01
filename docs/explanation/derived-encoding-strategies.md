# Derived Encoding Strategies

A planning state is relational. The fact `(on a b)` does not belong to a
single object; it ties `a` and `b` together at specific positions, and other
predicates tie three or more objects at once. A vanilla GNN layer, however,
only knows how to pass messages along pairwise edges between nodes. Before
message passing can start, the relational structure has to be written down as
one graph somehow.

The derived-graph encoders are six different answers to that problem. Each one
materializes the full structure of a state up front — facts, goals, history,
actions — into a plain homogeneous PyTorch Geometric graph, so that stock
layers work unchanged. They differ only in which *shape* the relations take:
extra nodes, direct object-to-object edges, shortcut edges, hyperedges,
distance tables, or tuple rows.

They deliberately do **not** differ in how much they remember. Every strategy
carries the full literal-instance content of the state it encodes; where a
shape cannot express something on its own, the encoder emits the auxiliary
node, self-loop, or side table that makes it expressible. What each view costs
you is compute and topology, not information. The
[Information Content](#information-content) section below states the exact
recovery procedure per encoder, and names every residual loss that survives.

This page explains the shapes and their trade-offs. For runnable code, tensor
layouts, and batching, see
[How to Consume Derived Graphs with Vanilla GNNs](../how-to/consume-with-vanilla-gnns.md).

## Choosing at a Glance

| Strategy | Nodes | Where relations live | Best for |
| --- | --- | --- | --- |
| `StarGraphEncoder` | objects + one node per fact/goal/history/action | hub edges from each fact node to its argument objects | faithful baseline for standard message passing |
| `ObjectGraphEncoder` | objects + anchor | labeled edges between argument objects (`clique` / `chain` / `star_first`), plus the tuple instance table | small, cheap object-only graphs |
| `AtomLineGraphEncoder` | objects + fact nodes (star universe) | star edges plus fact-to-fact `line_share` shortcuts | co-occurring facts that should interact in one hop |
| `HypergraphIncidenceEncoder` | objects + fact nodes + anchor | true n-ary hyperedges over object nodes | native hypergraph layers (`HypergraphConv`) |
| `TransformerBiasEncoder` | objects + anchor (clique) | clique edges, shortest-path-distance fields, tuple instance table | graph transformers with attention biases |
| `TupleTensorEncoder` | objects + fact nodes | star edges plus variable-length tuple rows | set transformers / DeepSets-style models |

All six take a problem (`pymimir.Problem` or `pytyr.PlanningTask`), optionally
with goals, actions, and history, write into the same integer-channel carrier,
and support `.stream()` and batch encoding.

## `StarGraphEncoder`: Every Fact Gets a Node

The most direct fix is *reification* — giving each grounded fact its own node.
In the star view, the fact `(on a b)` becomes one fact node wired to the
object nodes `a` and `b` by `arg_fwd` edges, each labeled with its argument
position (`a` at position 0, `b` at position 1). Reversed `arg_bwd` edges
point back from objects to facts by default.

Nothing is merged away. Goal literals, subgoal literals, history literals, and
grounded actions also become their own typed nodes; actions connect to their
arguments with `action_fwd` / `action_bwd` edges. A fact with no arguments
gets a `nullary_self` self-loop so message passing still reaches it.

On `data/pddl/blocks/smedium.pddl` — three blocks, five initial facts, two
goal literals — the initial state encodes to 13 nodes and 27 edges:

```text
a  b  c                              objects            (role 0)
(object a) (object b) (object c)     static type facts  (role 1, category 0)
(clear a) (handempty) (on a b) (on b c) (ontable c)     (role 1, category 1)
(on b a)[g] (on c b)[g]              goal literals      (role 2)
```

The suffixes are part of the node name, and they are injective: goal level 0
renders `[g]`, then `[sg]`, `[ssg]`, `[sssg]`, and level `L >= 4` renders
`[sg<L>]`, so the same atom at subgoal levels 3, 4 and 5 yields three
distinctly named nodes rather than three copies of `[sssg]`.

The trade-off is fidelity versus size. This view is the most faithful — every
literal instance is explicitly present — but a state with many grounded facts
produces many hub nodes. For graph-level predictions you typically pool over
the object-role rows only.

Pick this when you want maximum fidelity from a completely standard
message-passing setup and can afford the larger graphs.

## `ObjectGraphEncoder`: Facts Become Edges Between Objects

The opposite extreme: no fact nodes. Each fact turns into direct directed
edges between its argument objects, and a configurable combiner
(`atom_expansion`) decides which pairs get edges:

- `clique` — connect every pair of arguments. `(on a b)` yields the edge
  `a -> b`; an arity-3 fact yields a triangle. Richest, but an arity-k fact
  costs O(k^2) edges.
- `chain` — connect consecutive arguments: `a -> b`. Cheapest, but
  non-adjacent arguments sit far apart in the graph.
- `star_first` — connect the first argument to all others, so the head object
  broadcasts to the rest.

**Every projected edge carries its originating instance's full label.** Beyond
the edge kind and the two argument positions, each row records the relation
id, role, sign, goal level, history age, and category of the fact it came
from. A goal literal is therefore never confusable with a state fact, and two
predicates of the same arity never collapse onto the same edge label. On
`blocks/smedium` the objects-only clique view is 4 nodes and 14 edges, all
distinguishable:

```text
        a -> a          unary_self    pos=(0,0)  rel=object     role=fact  cat=static
        b -> b          unary_self    pos=(0,0)  rel=object     role=fact  cat=static
        c -> c          unary_self    pos=(0,0)  rel=object     role=fact  cat=static
        a -> a          unary_self    pos=(0,0)  rel=clear      role=fact  cat=fluent
        a -> b          clique_fwd    pos=(0,1)  rel=on         role=fact  cat=fluent
        b -> a          clique_bwd    pos=(1,0)  rel=on         role=fact  cat=fluent
        b -> c          clique_fwd    pos=(0,1)  rel=on         role=fact  cat=fluent
        c -> b          clique_bwd    pos=(1,0)  rel=on         role=fact  cat=fluent
        c -> c          unary_self    pos=(0,0)  rel=ontable    role=fact  cat=fluent
        b -> a          clique_fwd    pos=(0,1)  rel=on         role=goal  cat=fluent
        a -> b          clique_bwd    pos=(1,0)  rel=on         role=goal  cat=fluent
        c -> b          clique_fwd    pos=(0,1)  rel=on         role=goal  cat=fluent
        b -> c          clique_bwd    pos=(1,0)  rel=on         role=goal  cat=fluent
<nullary> -> <nullary>  nullary_self  pos=(0,0)  rel=handempty  role=fact  cat=fluent
```

Low arities used to vanish here; they no longer do. An arity-1 fact such as
`(clear a)` emits a `unary_self` loop on its single argument, and an arity-0
fact such as `(handempty)` emits a `nullary_self` loop on the **anchor** — one
auxiliary node, role `anchor`, appended as the last node of the graph and
named `<nullary>`. The anchor is not an object: it is absent from
`object_names` and never appears in the shortest-path tables, so object-role
pooling and `spd_*` consumers are unaffected. Its index is reported as
`anchor_index`, and the objects-only universe always emits it, which keeps the
node count stable across all states of one problem.

What a pairwise projection still cannot express on its own is **which pairs
belong to the same instance**. `data/pddl/tri/p1.pddl` — the bundled ternary
domain, with `(between ?x ?y ?z)`, `(link ?x ?y)` and nullary `(flag)` —
makes this concrete. Under `clique` its initial state is 5 nodes and 25
edges, and the two facts `(between a b c)` and `(between a b d)` each emit

```text
a -> b  clique_fwd  pos=(0,1)  rel=between  role=fact  sign=0  lvl=0  dt=0  cat=fluent
```

— two byte-identical parallel edges. Nothing in the topology says whether
`a -> c` or `a -> d` belongs to the first fact or the second. That is a
genuine limit of pairwise projection, not an oversight, so the encoder does
not rely on the topology alone: `node_universe="objects_only"` **forces the
tuple instance table on**. `ObjectGraphEncoder` and `TransformerBiasEncoder`
therefore always carry `tuple_args` / `tuple_ptr` and the six `tuple_*_ids`
channels (also stacked as `tuple_attr_ids`), and that table is the
authoritative, order-preserving instance list. For `tri/p1` it holds nine
rows, three of them arity 3, and lists `(between a b c)` as `[a, b, c]` and
`(between a b d)` as `[a, b, d]` — exactly the grouping the edges cannot
carry.

One thing this view does still reify: grounded **actions**. Passing
`actions=[...]` adds one node per grounded action in every node universe,
wired to its argument objects by `action_fwd` / `action_bwd` edges, because an
action is not a relation over objects that a projection could stand in for.
On `blocks/smedium` with the single applicable action supplied, the
objects-only graph is 5 nodes — `a`, `b`, `c`, `@(unstack a b)`, `<nullary>` —
with the anchor still last.

One consequence for the consumer: because nothing is reified, `x_ids` in this
view is information-free — every object row is zeros and the anchor row is
`[6, 0, 0, 0, 0, 0]`. All the state content sits on `edge_attr` and in the
tuple table, so this view needs an edge-conditioned layer or a model that
reads the instance table. The how-to has the
[recipe](../how-to/consume-with-vanilla-gnns.md#objects-only-facades-the-content-is-on-the-edges).

Pick this when you want compact object-only graphs, are happy to read instance
grouping off the tuple table instead of the topology, and your model consumes
edge features.

## `AtomLineGraphEncoder`: Shortcuts Between Facts That Share an Object

This is the star view plus one addition. In the plain star view, the facts
`(clear a)` and `(on a b)` interact only indirectly: a message from one to the
other travels through the shared object `a`, taking two hops. This encoder
adds a direct `line_share` edge between such facts, so object-mediated
interaction becomes a single hop.

Shortcuts stay bounded. An object incident to many facts would otherwise gain
a quadratic web of shortcut edges, so objects whose degree exceeds
`line_graph_max_degree` (default 32) get no shortcuts at all. A fact pair
sharing `m` objects yields `m` shortcut edges per direction, each with its own
`pos_a` / `pos_b` — those are distinct, information-bearing edges, not
duplicates. On `blocks/smedium`, `(on a b)` and the goal `(on b a)[g]` share
both objects and so get two forward shortcuts, one per shared object;
18 of the 20 connected pairs share exactly one object and get one. Shortcuts
also honour `include_reverse_edges`: the default emits 44 `line_share` edges
(71 edges in total), and `include_reverse_edges=False` emits the 22 forward
ones only (36 in total).

`line_share` is the one edge kind whose instance-label columns are all zero.
It is not derived from a single instance — it connects two reified fact nodes
that already carry their own six channels — so there is nothing for it to copy
and nothing lost by leaving those columns at "none".

Pick this when interactions between co-occurring facts matter and two-hop
paths through shared objects lose too much signal.

## `HypergraphIncidenceEncoder`: Relations as True Hyperedges

A clique projection pretends an n-ary relation is a bag of pairwise ones.
This encoder does not pretend: the fact `(on a b)` becomes one hyperedge whose
member list is exactly `(a, b)`, and a ternary fact becomes one membership
group over three objects instead of three pairwise edges. Member order follows
argument order and repeats survive, so `(on a a)` is a hyperedge with two
members, both `a`, and is distinguishable from a unary fact about `a`.

Every encoded literal instance becomes a hyperedge over its argument object
nodes — state facts, static facts, goal and subgoal literals, history
literals, and grounded actions alike. Each hyperedge carries the *same six
channels as a node row*: role, relation id + 1, sign, goal level, history age,
and category, stacked as `hyperedge_attr_ids`, an `[M, 6]` table. The
underlying star graph is still present; on top of it, `encode_pyg` reports
membership as `hyperedge_index` (member node ids paired with hyperedge ids),
ready for `torch_geometric.nn.HypergraphConv`.

**No hyperedge is ever empty.** An arity-0 instance takes the anchor node as
its single member, so every hyperedge has at least one member and

```text
hyperedge_index[1].max() + 1 == hyperedge_attr_ids.size(0)
```

holds by construction. That equality is exactly what `HypergraphConv` infers
when you do not pass `num_edges`, and it used to be false. On `blocks/smedium`
with `(handempty)` supplied as the goal, the encoder records 9 instances; the
nullary ones are the last, and with the anchor memberships removed — the old
behaviour, where an arity-0 instance produced a zero-member hyperedge — PyG
infers only 8, silently mis-aligning every attention weight past the gap. With
the anchor memberships present, the inference is 9 and matches
`hyperedge_attr_ids`.

Pick this when the arity of a relation carries meaning and you want a native
hypergraph layer rather than pairwise approximations.

## `TransformerBiasEncoder`: Distances for Graph Transformers

Graph transformers do not run local message passing; every token attends to
every other token. What they lack is structural knowledge about *how far
apart* two objects are. This encoder supplies it.

The base graph is the objects-only clique projection — anchor, `unary_self`
loops, labeled edges and forced tuple table included, exactly as
`ObjectGraphEncoder` builds it. On top, precomputed shortest-path distances
between objects are emitted as sparse triplets (`spd_src`, `spd_dst`,
`spd_dist`). Distance counts hops in the bipartite object-fact sense: crossing
one shared fact costs 2. In blocks-world terms, if `(on a b)` and `(on b c)`
hold, then `a` and `c` sit at distance 4 — two facts apart; on
`blocks/smedium` the encoder emits exactly `(a, b, 2)`, `(a, c, 4)`,
`(b, c, 2)`. Pairs farther than `spd_max_hops` (default 4) are simply omitted,
so the tables stay sparse. Because distances advance in steps of two,
`spd_max_hops` must be at least 2; smaller values are rejected. The anchor is
not an object and never appears in these triplets.

Feed these triplets into your attention score as bias terms and the transformer
can reason globally while still knowing which objects nearly touch.

Pick this when your model is a graph transformer and you want planning
structure injected as attention biases rather than message passing — and note
that it inherits the objects-only reading rule above: `x_ids` says nothing
here, so the state must come off `edge_attr`, the tuple table, or `spd_*`.

## `TupleTensorEncoder`: Facts as Rows for Set Models

Some models prefer facts as a flat set of rows, not as graph topology. This
encoder builds the star view and additionally exposes every literal instance
as one variable-length tuple row: `{(handempty), (on a b), (clear a)}` becomes
three rows, where the middle row lists argument nodes `a` and `b`.

Concretely, the output adds `tuple_args` (flattened argument node ids) and
`tuple_ptr` (CSR offsets delimiting each row, so arities vary without
padding), plus one id vector per node channel: `tuple_rel_ids`,
`tuple_role_ids`, `tuple_sign_ids`, `tuple_level_ids`, `tuple_dt_ids`, and
`tuple_category_ids`. Together those mirror `x_ids`'s six channels exactly, so
a tuple row and a fact-node row say the same thing in two layouts; the
convenience view `tuple_attr_ids` stacks them into a `[T, 6]` table in `x_ids`
column order. A set transformer or DeepSets-style network attends over these
rows directly instead of walking edges.

On `blocks/smedium` the initial state yields 10 rows with
`tuple_ptr = [0, 1, 2, 3, 4, 4, 6, 8, 9, 11, 13]` — the repeated `4` is the
nullary `(handempty)` row, an empty argument slice rather than a missing row.

Pick this when your architecture consumes sets of tuples natively and graph
convolution is the wrong shape of computation.

## Information Content

The contract for this family is that an encoding carries the **full
information load of the STRIPS state**, unconditionally. The unit of that load
is the *literal-instance multiset*: each instance is

```text
(role, relation_id, arguments[0..k), sign, goal_level, history_dt, category)
```

with `k >= 0` and repeated arguments allowed. An encoder is **lossless** iff
that multiset and the object table are exactly recoverable from the tensors it
emits together with the graph-attribute vocabularies. Anything short of that
is named below, with the mechanism.

| Encoder | Verdict |
| --- | --- |
| `StarGraphEncoder` | Lossless up to multiplicity — recovers the instance **set** exactly from the graph alone |
| `AtomLineGraphEncoder` | Lossless up to multiplicity — identical to the star view; `line_share` adds redundant structure only |
| `HypergraphIncidenceEncoder` | **Lossless** — the incidence table is a multiset record; the star graph is also present |
| `TupleTensorEncoder` | **Lossless** — the tuple table is the canonical multiset record |
| `ObjectGraphEncoder` | **Lossless as emitted** via the forced tuple table; the projection *alone* is lossy for arity >= 3 grouping |
| `TransformerBiasEncoder` | **Lossless as emitted** — same as `ObjectGraphEncoder`; `spd_*` is derived and adds no independent content |

Lossless *as emitted* is a statement about the tensors, not about any
particular way of reading them. For the two objects-only facades the content
is entirely on `edge_attr` and the tuple table, so a model that embeds `x_ids`
alone discards all of it while the encoding itself remains complete; the
how-to's
[objects-only recipe](../how-to/consume-with-vanilla-gnns.md#objects-only-facades-the-content-is-on-the-edges)
is the reading that recovers it.

### Recovering from a star-shaped view

`StarGraphEncoder`, `AtomLineGraphEncoder`, and the star graph carried inside
`HypergraphIncidenceEncoder` and `TupleTensorEncoder` all admit the same
procedure:

1. **Objects.** Node rows with `x_ids[:, 0] == 0`, in index order; their names
   are `object_names`.
2. **Instances.** Every other node row except the anchor
   (`x_ids[:, 0] == 6`). Its channels 1..5 give `relation_id + 1`, sign, goal
   level, history age, and category directly; channel 0 gives the role.
3. **Arguments.** For a fact node `v`, its outgoing `arg_fwd` edges
   (kind 0; `action_fwd`, kind 9, for action nodes) each carry
   `pos_a == pos_b == position`. Sort by position to get the argument tuple.
   Repeated arguments give two distinct edges at two distinct positions, so
   `(on a a)` reconstructs correctly. No `arg_fwd` edge means arity 0 — such a
   node carries a `nullary_self` loop instead.

Verified on `blocks/smedium` with negated goals, two subgoal layers, two
history steps, one grounded action, and an `(on a a)` goal: the
reconstruction equals the encoder's own instance table exactly — 16
instances, none repeated — for both the star and the line-graph views. The
runnable version of this procedure is in
[the how-to](../how-to/consume-with-vanilla-gnns.md#information-content).

**The residual loss: multiplicity.** The graph view is a *set*; the
instance tables are *multisets*. Node interning is idempotent, and since the
set/multiset split was made explicit, edge emission is too — so passing the
same literal twice yields one node and one set of edges, not two. On
`blocks/smedium`, `goals=[lit, lit]` leaves the star view at 12 nodes and 23
edges either way, while the tuple table grows from 9 rows to 10 and the
hyperedge table from 9 to 10. If the multiplicity of an exactly-repeated
instance is meaningful to you, read it from the instance table; the topology
will not tell you. For a genuine STRIPS state — a *set* of atoms — the two
agree.

### Recovering from the hyperedge incidence

Group `hyperedge_index` by its second row: the member node ids of hyperedge
`h`, in emission order, are that hyperedge's arguments in argument order,
after dropping `anchor_index` (which stands in for the empty argument list of
an arity-0 instance). Row `h` of `hyperedge_attr_ids` supplies the same six
channels as a node row. This is a multiset record: two identical goal literals
produce two hyperedges.

### Recovering from an objects-only view

Read the tuple table, which `objects_only` now always emits:
`tuple_ptr[i]:tuple_ptr[i + 1]` slices `tuple_args` into row `i`'s argument
node ids, and the six `tuple_*_ids` vectors supply the channels. This is a
multiset record and is exact, including arity 0 (empty slice), arity 1, and
the arity >= 3 grouping the edges cannot express.

The **projection alone** is lossy, and precisely so:

- Arity >= 3 instances of the same relation, role, sign, level, age and
  category that share an argument pair emit byte-identical parallel edges, and
  the remaining pairs cannot be attributed to one instance or the other. The
  `tri/p1` example above is the minimal case.
- Consequently, the number of instances of a relation is not readable off the
  edge multiset for arity >= 3.

Everything else survives: a goal literal and a state fact over the same
objects differ in the `role` column, a negated literal differs in `sign`, a
subgoal layer in `goal_level`, a history literal in `history_dt`, and a
different predicate in `rel_id_plus_one`. Arity 0 leaves a `nullary_self` loop
on the anchor, arity 1 a `unary_self` loop on its argument. Under the previous
contract none of that was true: the objects-only clique view of
`blocks/smedium` emitted four forward edges all labeled `(clique_fwd, 0, 1)`
over an all-zero `x_ids` — two from state facts, two from goals, mutually
indistinguishable — and unary and nullary facts left no trace at all.

### The loss you opt into: `include_metadata=False`

`include_metadata=False` drops every graph attribute, the vocabularies
included. The tensors still describe the state completely as *integers* —
`x_ids`, `edge_attr`, `history_dt_offset`, `anchor_index`, and the instance
tables all survive — but without `vocab_relations`, `vocab_roles`,
`vocab_categories`, `vocab_edge_kinds`, `num_predicates`, `node_names` and
`object_names` there is no way to decode an id back to a predicate, action, or
object *name*, nor to split the relation id space into predicates and action
schemas. Keep the default `include_metadata=True` whenever you need to name
what you decoded.

## Roadmap: Beyond State Graphs, Task-Structure Encodings

Everything above encodes *states* of one problem. The research literature also
benchmarks GNNs on task-level graphs: the IPC graph dataset (Ferber et al.
2019, [github.com/IBM/IPC-graph-data](https://github.com/IBM/IPC-graph-data))
publishes Problem Description Graphs (PDGs, grounded) and Abstract Structure
Graphs (ASGs, lifted and acyclic) for every competition instance, originally
to learn planner selection from graph embeddings.

The first member of this family now ships: **`LiftedTaskEncoder`**
(`mifrost.encoders.lifted`, built on the custom toolkit) encodes the
ASG-style *lifted task structure* — one node per object, predicate schema,
action schema, and action parameter, wired by lifted precondition, effect,
and goal edges with conditional-effect group indices — as a state-independent
homogeneous graph suited to planner-selection-style readouts (global mean
pooling over the graph). See the module docstring for the full channel and
edge-kind tables; tests live in
`tests/encoding/test_lifted_task_encoder.py`. Still future work for this
family: PDG-style *grounded* transition graphs (operator/variable bipartite
graphs over ground atoms), causal links, and domain-transition graphs; the
custom toolkit described in
[How to Write Your Own Encoder](../how-to/write-your-own-encoder.md) lowers
the barrier for prototyping them in pure Python today.

## Reading the Outputs

All six strategies write the same integer-id channels: node features live in
`x_ids`, an `[N, 6]` table holding role, relation (id + 1), sign, goal level,
history age, and category, while edge features live in `edge_attr`, an
`[E, 9]` table holding the edge kind, the two endpoint argument positions, and
the six channels of the instance the edge was derived from.

`x_ids` is the only node-feature source: these carriers deliberately do not
define `x`. A zero-filled placeholder used to sit there so that the PyG reflex
`conv(data.x, data.edge_index)` would run — and it ran on nothing, returning
one identical row for every node with no error. That is precisely the kind of
silent loss this family refuses to have, so `data.x` is now `None` and stock
layers raise instead. See
[the how-to](../how-to/consume-with-vanilla-gnns.md#there-is-no-datax).

Column 1 indexes a **unified relation space**: predicates take ids
`0 .. P - 1` and action schemas take `P + a`, where `P` is the graph attribute
`num_predicates`. `vocab_relations` names the whole space and is exactly
`vocab_predicates + vocab_actions`, so a single embedding table covers both
without predicate id `a` and action-schema id `a` ever landing on the same
row. Embed each column separately and consult the vocabulary metadata carried
on the graph (`vocab_roles`, `vocab_relations`, `vocab_predicates`,
`vocab_actions`, `vocab_edge_kinds`, `vocab_categories`; also exposed as
`ROLE_NAMES` and `EDGE_KIND_NAMES` in `mifrost.encoders.derived`) — the how-to
page linked above owns the full channel tables and the embedding recipe,
including the one channel that is signed.
