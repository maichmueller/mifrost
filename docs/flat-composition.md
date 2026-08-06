# Compositional flat encoding

`mifrost/core/encoders/flat/flat_composition.hpp` provides the native
composition boundary for future flat encoders.  A downstream encoder declares
components once, compiles the declarations, and reuses the resulting
`CompiledFlatPlan` (or `FlatBatchRuntime`) for every batch.

The lifecycle is deliberately fixed:

1. Components declare structured `RelationKey` schemas, graph fields, and node
   types.
2. Compilation assigns stable relation and node-type ids, validates field
   ownership, validates projection arity and node references, and freezes the
   declarations.
3. For each graph, components plan symbolic nodes, prepare, emit native
   relation tuples, and write their owned fields.
4. One `BatchBuilder` receives the graph and the runtime writes relation counts,
   relation arguments, and relation-instance sizes exactly once.
5. Components write owner-scoped non-field metadata; the object component
   publishes object names without a second materialized encoding.
6. Components write native node feature columns through
   `FlatNodeFeatureWriter`; feature dimensions and row counts are checked
   against the shared node plan.
7. `finalize_batch_encoding()` optionally performs the existing relation-major
   collation pass.

Virtual dispatch is limited to the component/graph lifecycle.  Fact, action,
goal, and transition loops remain inside native component implementations;
there is no Python callback or per-relation component dispatch in this layer.
`FlatRelationProjection` performs slot remapping with integer relation ids and
supports source slots, constants, and graph-local node references.
`FlatProjectionHandle` carries the component and declaration identity rather
than relying only on a component-local ordinal, so recompilation can insert
other projections without silently rebinding an existing emitter.
Components that emit many tuples should resolve `FlatGraphContext::relation_id`
once before their hot loop and call the integer-id overload of `emit`.
`FlatSchemaPlanBuilder::register_relation_alias()` adds a non-exported symbolic
alias; compilation resolves aliases to the canonical relation id, and both
emission and projections use that compiled mapping.
`FlatGraphScratch` is the graph-local preparation channel: a component can
materialize expensive native lookup state in `prepare_graph()` and share it
with later emit/field phases without mutating the compiled component or using
Python callbacks.

## Native built-in components

The header also includes small backend-neutral components for the common
carrier operations:

- `FlatObjectNodeComponent` declares one node type and adds the input object's
  names in deterministic order.
- `FlatNodeRecordComponent` declares one typed node table and selects matching
  `FlatCompositionInput::nodes` records.
- `FlatRelationEmitterComponent` declares a set of relation keys/layouts and
  emits `FlatCompositionInput::relations` using already-resolved integer ids.
  Set a relation record's optional `component` owner when several relation
  emitters declare the same relation. With the default `unique_owner` policy,
  an empty owner is valid only when exactly one component declares that
  relation; `broadcast` must be selected explicitly to duplicate it.
- `FlatFieldEmitterComponent` declares owned graph fields and writes typed
  `FlatCompositionInput::fields` values with the same shape checks as
  `BatchBuilder`.

Components can also implement `write_metadata()` through
`FlatMetadataWriter`; this is the native path for object names and future
graph-level metadata that is not a numeric graph field.
Metadata declarations are compiled with single-owner checks, just like graph
fields, so two components cannot silently overwrite object-name metadata.
`write_node_features()` and `FlatNodeFeatureWriter` are the corresponding
native path for node tensors; components may emit zeros or semantic features,
but must use the preplanned row count. Feature columns have compile-time
single-owner declarations and cannot be appended twice for one graph.
Graph attributes can be declared with
`FlatMetadataPlanBuilder::claim_graph_attr()` and written through the variant
overload on `FlatMetadataWriter`; duplicate attribute ownership is rejected at
compile time.

`FlatCompositionInput` is intentionally a generic carrier, not a semantic model.
Backend adapters own object/action/goal interpretation. New adapters should
present that interpretation through the statically dispatched Views in
`mifrost/core/views/concepts.hpp` and use the canonical traversal primitives in
`mifrost/core/views/canonical.hpp`; they must not construct a second owning
semantic object for every atom or action on the normal path. Legacy semantic
snapshot adapters remain available for compatibility and explicit conversion
APIs.

Adapters must resolve each relation key with the compiled schema once per plan
(for example, with `compiled.schema().id_for(...)`) before filling a relation
record. A record with an unknown id, wrong arity, missing field, or mismatched
field dtype is rejected at the native boundary rather than silently producing
an incompatible batch.

`FlatCompositionInputBuilder` is the optional adapter-side convenience for this
boundary.  Its relation-key overloads are setup operations; use
`relation_id()` plus the integer overload when constructing a large relation
stream, and call `finish()` only after all graph-local values have been added.
Construct it from `FlatSchemaPlan` when the adapter needs the plan's compiled
relation aliases as well as canonical keys.

`compare_flat_batch_encodings()` compares the native carrier, including node
tables, columns, graph attributes/fields, pointers, schema metadata, and lazy
target-name materializations.  Migration tests should run this comparator on a
legacy encoding and a composed encoding and treat any non-empty `mismatch` path
as a parity failure.  It is deliberately an exact comparison; normalization,
sorting, or tolerance belongs in a test fixture only when the legacy contract
explicitly permits it.

### Semantic flat components

The semantic relation encoder compiles entity, fact, goal/derivation, action,
history, field, target-metadata, and LGAN components. Its graph plan contains
only graph-local semantic lookup state and target/node identities. The
canonical View traversal primitives are available to backend-specific
instantiations. Templated `encode(...)` overloads accept granular state, goal,
subgoal, action, and history Views and synchronously fill the encoder's private
graph-key preparation state. The explicit `SemanticFlatRelationInput` remains
the owned compatibility input. Both paths emit directly into the runtime's
`BatchBuilder`; neither constructs `FlatCompositionInput`, callback ranges, or
an intermediate encoding.

The overall boundary is:

```text
backend values
  -> granular borrowed Views
  -> canonical statically dispatched family algorithm
  -> graph-derived intern/index/working structures
  -> BatchBuilder
```

"Graph-derived structures" means compact identity pools and indices into them
-- the atom and action intern pools, goal-level references, entity indices,
unique target identities, occurrence ordering, fact membership, relation-major
metadata, history ordering, and composition-plan state. It does not mean a
per-lane owning mirror of the input: the direct path never builds
`std::vector<SemanticAtom> state_facts` / `std::vector<SemanticLiteral> goals` /
`std::vector<SemanticGroundAction> actions` copies just to cross from the View
API into the algorithm, and the compatibility path is read through borrowed
references to the lanes `SemanticFlatRelationInput` already owns.

One conversion is still outstanding on this path: `PreparedRelationGraph`
selects its storage mode with a nullable-pointer check that the range carriers
repeat per element. Both modes execute the same source algorithm, so output is
unaffected, but those carriers are intended to become concrete ranges selected
once at the public entry point.

The semantic horizon encoder uses the same runtime. Its graph plan prepares
candidate identities, exact transition deltas, and goal-membership sets. A
transition-semantic component emits state, goal, action, and effect tuples;
the topology component independently emits parent, sibling, and cousin tuples.
Fields and LGAN edges are derived after those components have populated the
same final sink. One-shot, caller-owned builder, and batch APIs all execute
these compiled plans.

The neutral C++ suite covers exact output parity between public execution
forms, the full semantic policy matrix, mixed target-name metadata, relation
argument layouts, and concurrent reuse of one immutable compiled engine.
`mifrost_bench_semantic_flat_composition` reports one-shot, caller-owned builder,
and 32-graph batch throughput for a representative semantic workload. It is a
stable regression benchmark; compare results to a recorded build from the
pre-refactor commit rather than treating two public wrappers as independent
implementations.

`FlatRelationMajorWriter` is the shared finalization seam.  The semantic flat
relation and horizon engines let their compiled plan invoke this writer exactly
once for one-shot and batch results. Caller-owned `BatchBuilder` paths invoke
the same writer through the public `finalize_batch_encoding()` operation after
the caller commits and builds the batch.

## Backend boundary and Views

The reusable implementation lives in the planner-neutral core:

- `FlatEncoderPlan`, `CompiledFlatPlan`, and `FlatBatchRuntime` are the public
  C++ extension seam for new native flat encoders.
- `SemanticFlatRelationEncoderEngine` and `SemanticFlatHorizonEncoderEngine`
  are the canonical built-in assemblies. Both execute direct semantic
  components through a compiled plan. The former accepts both direct View lanes
  and owned semantic records; the latter retains semantic transition DAG input
  because a horizon DAG is itself an owned semantic snapshot.
- Planner adapters provide task-scoped contexts and granular non-owning Views.
  The same canonical algorithms are instantiated once for PyTyr and once for
  Pymimir, keeping native types and nanobind ABI domains isolated. Semantic
  snapshot adapters remain an explicit compatibility boundary for callers that
  request owned records; they are not required by the View contract.

The Pymimir-only `FlatRelationEncoderEngine` and `FlatHorizonEncoderEngine`
retain their historical constructors and streaming contracts while adapting
native values through the same task-scoped View concepts and canonical
semantic engines. They are not the extension seam for new encoders. New
backend-neutral and downstream encoders must use the core composition API and
a backend View context.

The normal Pymimir Flat, Color, HGraph, successor, batch, and stream paths now
enter through that direct View boundary. The legacy semantic-record APIs remain
available for capsules, explicit snapshots, and compatibility callers. A direct
encode may still create private graph-local preparation state before relation
emission; this is distinct from materializing a public
`SemanticFlatRelationInput` and never stores a planner-native object in the
neutral engine.

### View lifetime and streaming

Direct backend streams encode each appended step immediately and retain only
the resulting native batch encoding. A caller must keep the planning task,
problem, and View context alive through the append call; a stream must not
retain a lazy range after its source state has been destroyed. Deferred flush
tests should exercise both mutable removal and replacement so a stale View
cannot be observed.

## Generic composition capabilities

Mifrost exposes only generic `FlatCompositionCapability` bits.  A downstream
adapter may define its own named assemblies and validate them with
`flat_composition_capabilities_satisfied()` or
`flat_composition_missing_capabilities()`.  Mifrost does not publish
downstream mode names or parity claims; exact mode parity belongs in the
downstream repository.

## Compatibility and performance rules

- Compile a plan once per encoder configuration and cache it.
- Share a `std::shared_ptr<const CompiledFlatPlan>` with one `FlatBatchRuntime`
  per worker; components must keep compiled state
  immutable (graph-local state belongs in the graph context), and do not append
  already-materialized `BatchEncoding` values to compose components.
- Keep relation ordering, tuple arity, node identity, and graph-field ownership
  deterministic. Backend adapters should translate to semantic inputs once;
  they must not rebuild relation carriers or append materialized encodings.
- Set `FlatCompositionConfig::relation_args_node_type` when the composed plan
  uses a node table other than `entity`; relation argument fields then receive
  the correct graph offset source.
