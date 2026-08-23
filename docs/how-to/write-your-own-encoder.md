# How to Write Your Own Encoder

The derived-graph families ship a fixed menu of graph shapes. When you need a
different shape — new node roles, exotic edges, your own channel layout — you
do not have to drop into C++. The `mifrost.encoders.custom` toolkit is pure
Python: it hands you planner-neutral views of the problem and state
(`StateView`, plus neutral `Atom` / `Literal` records), small building blocks
for nodes, edges, and vocabularies, and a writer that emits the same native
`BatchEncoding` objects every built-in family produces. You get
native-quality outputs, batching, streaming, and backend parity without a
build step; the C++ composition layer
([Compositional flat encoding](../flat-composition.md)) remains the tool for
production-grade *new encoder families* implemented natively.

## The Four Pieces

**1. `StateView` — one planner-neutral facade.** Built once over a pymimir
`Problem` or a pytyr `PlanningTask`; everything downstream sees plain Python
records instead of planner types:

```python
from mifrost.encoders.custom import StateView

view = StateView(problem)          # auto-detects the backend; override with backend=
view.objects                       # ["a", "b", "c", ...]
view.predicates                    # [PredicateInfo("on", 2, "fluent"), ...]
view.state_facts(state)            # tuple[Atom] — Atom(predicate="on", args=("a", "b"))
```

Per-state lanes arrive neutralized too: `state_facts(state)`,
`goal_literals(state)`, `static_facts`, and
`neutral_actions(state, actions)` all speak `Atom` / `Literal`.

**2. Tables — interned rows, edges, vocabularies.**

```python
from mifrost.encoders.custom import EdgeSink, NodeTable, Vocabulary

nodes = NodeTable()                # interned by any hashable key
fact = nodes.id_for(("fact", "(on a b)"), role="fact", channels=(4,))
edges = EdgeSink()                 # kinds live in an edge-kind Vocabulary
edges.add_both(fact, obj_id, "arg_fwd", "arg_bwd", pos_a=0, pos_b=0)
edge_index, edge_attr = edges.to_arrays()   # int64 [2, E], float32 [E, 3]
```

**3. `GraphWriter` — one graph under construction.** Wraps a fresh
`BatchBuilder` plus one `NodeTable` and one `EdgeSink`, adds named
graph-level vocabularies (`vocab_<name>` attributes on the output), and
`finish()` returns the native encoding:

```python
writer = GraphWriter(view)         # graph_kind="homo", export_node_names=True by default
fact = writer.add_node(("fact", atom.display), role="fact", channels=(pid,))
writer.vocabulary("predicates").id_for(atom.predicate)
writer.set_vocab_attr("predicates")          # writes graph attr vocab_predicates
encoding = writer.finish()
```

**4. `CustomGraphEncoder` + `CustomStream` — the encoder shell.** Subclass,
implement exactly one method, and inherit encode/batch/stream machinery:

```python
class MyEncoder(CustomGraphEncoder):
    accepted_kwargs = frozenset({"my_weight"})     # extra kwargs you accept

    def encode_state(self, out, state, *, goals=None, actions=None,
                     subgoal_layers=None, history_subgoals=None,
                     history_max_steps=None, **kwargs):
        ...  # lanes arrive as neutral Atom/Literal records already

# MyEncoder(problem).stream() -> CustomStream with append/update/remove/flush
```

## A Minimal Worked Example

A tiny star-shaped encoder: one node per object, one fact node per state
fact wired to its arguments with forward/backward position-labeled edges,
goal literals as separate `goal`-role nodes, and the predicate vocabulary
exported as a graph attribute:

```python
from mifrost.encoders.custom import CustomGraphEncoder


class FactStarEncoder(CustomGraphEncoder):
    """Object nodes + fact/goal literal nodes, star-wired by argument."""

    def encode_state(self, out, state, *, goals=None, actions=None,
                     subgoal_layers=None, history_subgoals=None,
                     history_max_steps=None, **kwargs):
        predicates = out.vocabulary("predicates")

        object_ids = {
            name: out.add_node(("object", name), role="object", name=name)
            for name in self.view.objects
        }

        for atom in self.view.state_facts(state):
            fact_id = out.add_node(
                ("fact", atom.display),
                role="fact",
                channels=(predicates.id_for(atom.predicate) + 1,),
            )
            for pos, arg in enumerate(atom.args):
                out.add_both(fact_id, object_ids[arg],
                             "arg_fwd", "arg_bwd", pos_a=pos, pos_b=pos)

        for lit in goals or ():
            goal_id = out.add_node(
                ("goal", lit.atom.display),
                role="goal",
                channels=(predicates.id_for(lit.atom.predicate) + 1,),
            )
            for pos, arg in enumerate(lit.atom.args):
                out.add_both(goal_id, object_ids[arg],
                             "arg_fwd", "arg_bwd", pos_a=pos, pos_b=pos)

        out.set_vocab_attr("predicates")
```

Note what `encode_state` does *not* do: call `finish()`. The base class
finishes the writer for you after your method returns, which is why the same
subclass works unchanged for single, batched, and streamed encoding.

Using it end to end — single graphs, batches, and streams:

```python
import pymimir

domain = pymimir.Domain("data/pddl/blocks/domain.pddl")
problem = pymimir.Problem(domain, "data/pddl/blocks/small.pddl", mode="lifted")
state = problem.get_initial_state()

encoder = FactStarEncoder(problem)

data = encoder.encode_pyg(state)             # torch_geometric.data.Data
print(data.x_ids.shape)                      # [N, C] float32 node channels
print(data.edge_attr.shape)                  # [E, 3]: kind, pos_a, pos_b
print(data.vocab_predicates)                 # attr written by set_vocab_attr

successor = problem.get_initial_state() \
    .generate_applicable_actions()[0].apply(state)
batch = encoder.encode_batch_pyg([state, successor])
print(batch.num_graphs)                      # collation handled for you

stream = encoder.stream()                    # stores cheap (state, kwargs) recipes
first = stream.append(state)
second = stream.append(successor)
stream.update(first, successor)              # replace the recipe at id `first`
stream.remove(second)
encoding = stream.flush()
print(encoding.schema_flags["custom_encoder"])   # True
```

Node channels here are a single column (predicate id + 1, zero for objects);
edge columns follow the same `(kind, pos_a, pos_b)` layout used by the
derived families — see
[How to Consume Derived Graphs with Vanilla GNNs](consume-with-vanilla-gnns.md)
for how such integer-id channels feed stock GNN layers.

## House Rules You Inherit for Free

Subclassing `CustomGraphEncoder` buys the whole non-PyG surface of the
built-in encoders:

- **Kwargs validation.** Unknown keyword arguments raise `TypeError` naming
  the offender. The base accepts the standard lanes (`goals`, `actions`,
  `subgoal_layers`, `history_subgoals`, `history_max_steps`) and anything you
  declare in `accepted_kwargs`.
- **Native outputs + PyG fast path.** `encode` / `encode_batch` return native
  `BatchEncoding` objects (schema flag `custom_encoder=True` marks them);
  `encode_pyg` / `encode_batch_pyg` convert through the standard machinery.
- **Batch collation.** `encode_batch` merges per-state encodings via
  `mifrost.batch_encodings`; no custom collation code needed.
- **Streams.** `stream()` returns a `CustomStream` storing `(state, kwargs)`
  recipes; `append` / `update` / `remove` / `set_reuse_removed` / `flush`
  work like the built-in stream facades.
- **Backend parity.** One subclass serves both planners. Test both backends
  on identical PDDL inputs the way
  `tests/encoding/test_custom_toolkit.py` does — pymimir and pytyr views must
  agree on objects, predicates, schemas, facts, and goals.
- **Making a scheme public.** To export from `mifrost.encoders` (and the
  package root), register it in the manifest in
  `src/mifrost/_encoder_public.py`:

  ```python
  LazyEncoderExport("FactStarEncoder", ".custom", "FactStarEncoder"),
  ```

  `top_level=True` (the default) also requires listing the name in the
  manifest's `top_level_order` tuple so it appears as
  `mifrost.FactStarEncoder`; pass `top_level=False` to keep it inside the
  `mifrost.encoders` namespace only.
- **Compatibility bar.** New schemes should clear the same bar as the derived
  families: forward/backward passes through stock PyG layers, as exercised in
  `tests/encoding/test_gnn_conformance.py`.

## Pitfalls

- **First interning wins.** `NodeTable.id_for` ignores channels and names on
  repeat keys. If a node needs different features per lane (a fact that is
  also a goal), use distinct keys or roles rather than re-interning.
- **Stream state lifetime.** `CustomStream` stores references to the states
  you append; they must outlive the stream. Planner state objects can be
  invalidated by further search — keep your own copies alive.
- **Graph attrs only via the writer.** Mutate outputs through
  `set_attr` / `set_vocab_attr` (and `set_flag` / `register_field` /
  `set_field`); do not reach into the `BatchBuilder`.
- **`edge_attr` is always 3 channels** `(kind, pos_a, pos_b)`, even for empty
  edge sets — attention layers with `edge_dim=3` keep working.
- **Do not retain the `GraphWriter` after `finish()`**; each call to
  `encode_state` receives a fresh writer, and finished ones are spent.
