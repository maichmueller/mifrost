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

This page explains the shapes and their trade-offs. For runnable code, tensor
layouts, and batching, see
[How to Consume Derived Graphs with Vanilla GNNs](../how-to/consume-with-vanilla-gnns.md).

## Choosing at a Glance

| Strategy | Nodes | Where relations live | Best for |
| --- | --- | --- | --- |
| `StarGraphEncoder` | objects + one node per fact/goal/history/action | hub edges from each fact node to its argument objects | faithful baseline for standard message passing |
| `ObjectGraphEncoder` | objects only | directed edges between argument objects (`clique` / `chain` / `star_first`) | small, cheap object-only graphs |
| `AtomLineGraphEncoder` | objects + fact nodes (star universe) | star edges plus fact-to-fact `line_share` shortcuts | co-occurring facts that should interact in one hop |
| `HypergraphIncidenceEncoder` | objects + fact nodes | true n-ary hyperedges over object nodes | native hypergraph layers (`HypergraphConv`) |
| `TransformerBiasEncoder` | objects only (clique) | clique edges plus shortest-path-distance fields | graph transformers with attention biases |
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
gets a small self-loop so message passing still reaches it.

The trade-off is fidelity versus size. This view is the most faithful — every
literal instance is explicitly present — but a state with many grounded facts
produces many hub nodes. For graph-level predictions you typically pool over
the object-role rows only.

Pick this when you want maximum fidelity from a completely standard
message-passing setup and can afford the larger graphs.

## `ObjectGraphEncoder`: Facts Become Edges Between Objects

The opposite extreme: no fact nodes at all. Each fact turns into direct
directed edges between its argument objects, and a configurable combiner
(`atom_expansion`) decides which pairs get edges:

- `clique` — connect every pair of arguments. `(on a b)` yields the edge
  `a -> b`; an arity-3 fact yields a triangle. Richest, but an arity-k fact
  costs O(k^2) edges.
- `chain` — connect consecutive arguments: `a -> b`. Cheapest, but
  non-adjacent arguments sit far apart in the graph.
- `star_first` — connect the first argument to all others, so the head object
  broadcasts to the rest.

The edge label remembers the direction and the two argument positions. Be
aware of what it does not remember: the predicate itself, and anything about
facts with fewer than two arguments. Two predicates of the same arity produce
the same edge pattern, and unary facts like `(handempty)` leave no trace in
the graph. This is the least faithful view — the price for the smallest node
sets.

Pick this when you want compact object-only graphs for quick baselines and can
accept the information loss.

## `AtomLineGraphEncoder`: Shortcuts Between Facts That Share an Object

This is the star view plus one addition. In the plain star view, the facts
`(clear a)` and `(on a b)` interact only indirectly: a message from one to the
other travels through the shared object `a`, taking two hops. This encoder
adds a direct `line_share` edge between such facts, so object-mediated
interaction becomes a single hop.

Shortcuts stay bounded. An object incident to many facts would otherwise gain
a quadratic web of shortcut edges, so objects whose degree exceeds
`line_graph_max_degree` (default 32) get no shortcuts at all.

Pick this when interactions between co-occurring facts matter and two-hop
paths through shared objects lose too much signal.

## `HypergraphIncidenceEncoder`: Relations as True Hyperedges

A clique projection pretends an n-ary relation is a bag of pairwise ones.
This encoder does not pretend: the fact `(on a b)` becomes one hyperedge whose
member set is exactly `{a, b}`, and a hypothetical ternary fact would be one
membership group over three objects instead of three pairwise edges.

Every encoded literal instance becomes a hyperedge over its argument object
nodes — state facts, static facts, goal and subgoal literals, history
literals, and grounded actions alike. Each hyperedge keeps a role id, so the
model can tell a state fact from a goal literal. The underlying star graph is
still present; on top of it, `encode_pyg` reports membership as
`hyperedge_index` (member node ids paired with hyperedge ids) plus
`hyperedge_attr_ids`, ready for `torch_geometric.nn.HypergraphConv`.

Pick this when the arity of a relation carries meaning and you want a native
hypergraph layer rather than pairwise approximations.

## `TransformerBiasEncoder`: Distances for Graph Transformers

Graph transformers do not run local message passing; every token attends to
every other token. What they lack is structural knowledge about *how far
apart* two objects are. This encoder supplies it.

The base graph is the objects-only clique projection. On top, precomputed
shortest-path distances between objects are emitted as sparse triplets
(`spd_src`, `spd_dst`, `spd_dist`). Distance counts hops in the bipartite
object-fact sense: crossing one shared fact costs 2. In blocks-world terms,
if `(on a b)` and `(on b c)` hold, then `a` and `c` sit at distance 4 — two
facts apart. Pairs farther than `spd_max_hops` (default 4) are simply omitted,
so the tables stay sparse. Because distances advance in steps of two,
`spd_max_hops` must be at least 2; smaller values are rejected.

Feed these triplets into your attention score as bias terms and the transformer
can reason globally while still knowing which objects nearly touch.

Pick this when your model is a graph transformer and you want planning
structure injected as attention biases rather than message passing.

## `TupleTensorEncoder`: Facts as Rows for Set Models

Some models prefer facts as a flat set of rows, not as graph topology. This
encoder builds the star view and additionally exposes every literal instance
as one variable-length tuple row: `{(handempty), (on a b), (clear a)}` becomes
three rows, where the middle row lists argument nodes `a` and `b`.

Concretely, the output adds `tuple_args` (flattened argument node ids),
`tuple_ptr` (CSR offsets delimiting each row, so arities vary without
padding), plus `tuple_rel_ids` (predicate id) and `tuple_role_ids` (which lane
the instance came from: fact, goal, subgoal, history, action). A set
transformer or DeepSets-style network attends over these rows directly instead
of walking edges.

Pick this when your architecture consumes sets of tuples natively and graph
convolution is the wrong shape of computation.

## Roadmap: Beyond State Graphs, Task-Structure Encodings

Everything on this page encodes *states* of one problem. The research
literature also benchmarks GNNs on task-level graphs: the IPC graph dataset
(Ferber et al. 2019, [github.com/IBM/IPC-graph-data](https://github.com/IBM/IPC-graph-data))
publishes Problem Description Graphs (PDGs, grounded) and Abstract Structure
Graphs (ASGs, lifted and acyclic) for every competition instance, originally
to learn planner selection from graph embeddings. PlanBench
(Valmeekam et al.) covers many IPC domains too, but targets LLM evaluation
and ships no GNN encodings. Operator/variable bipartite graphs, causal
links, and domain-transition graphs are natural members of this family, and
the custom toolkit described in
[How to Write Your Own Encoder](../how-to/write-your-own-encoder.md) lowers
the barrier for prototyping them in pure Python today. One gap remains:
a PDG-style encoder needs domain-level action schemas with preconditions and
effects, which `StateView` does not yet expose — flagged as future work for
both the toolkit and a first-class family here.

## Reading the Outputs

All six strategies write the same integer-id channels: node features live in
`x_ids`, an `[N, 6]` table holding role, predicate (id + 1), sign, goal level,
history age, and category, while edge features live in `edge_attr`, an `[E,
3]` table holding the edge kind and the two endpoint argument positions. Embed
each column separately and consult the vocabulary metadata carried on the
graph (`vocab_roles`, `vocab_predicates`, `vocab_edge_kinds`,
`vocab_categories`; also exposed as `ROLE_NAMES` and `EDGE_KIND_NAMES` in
`mifrost.encoders.derived`) — the how-to page linked above owns the full
channel tables and the embedding recipe.
