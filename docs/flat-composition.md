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

The storage mode is chosen exactly once. `ViewSource` and
`CompatibilitySource` are two concrete borrowed accessors over their own
storage; `validate_source`, `make_context` and `prepare_source` are templated
over them and instantiated for both. Preparation returns a fully resolved
`PreparedRelationGraph` holding borrowed spans settled once plus compact
graph-derived state, so no emitter re-tests which kind of input it came from.

On the goal lane specifically, the category-grouped list holds
`{entry_index, effective_level}` rather than copied literals, so the *emission*
passes share one occurrence list instead of each rebuilding one. The effective
level of a repeated goal — the highest level it appears at — is resolved once
during preparation rather than on each emission pass. Repeated goals keep every
occurrence and every occurrence keeps the same effective level, matching the
historical map-assignment semantics.

Preparation itself is not copy-free, and the claim above is about emission, not
about the whole encode. Resolving effective levels sorts a copy of the
occurrence list, and each family then builds its own grouped view of it; the
flat history lane likewise materializes its references into
`PreparedHistoryEntry`. These are bounded graph-working structures, not a
per-lane mirror of the input, but "each identity materialized exactly once" is
not true end to end. Replacing the sorted copy with an effective-level index
over the compact references is a known, unclaimed optimization.

The semantic horizon encoder uses the same runtime. Its graph plan prepares
candidate identities, exact transition deltas, and goal-membership sets. A
transition-semantic component emits state, goal, action, and effect tuples;
the topology component independently emits parent, sibling, and cousin tuples.
Fields and LGAN edges are derived after those components have populated the
same final sink. One-shot, caller-owned builder, and batch APIs all execute
these compiled plans.

The assembly ownership model is shared by semantic encoders rather than owned
by Horizon. `SemanticAnnotations`, `SemanticEncoderInput<Source>`, and
`SemanticAssemblyComponents<Component>` live in the backend-neutral common
layer. They define sidecar lifetime, borrowed-or-owned inputs, exclusive
pre-compilation component ownership, and the one-way freeze operation. Concrete
encoder families still define their own component contracts and compiled
runtimes: flat encoders use `FlatEmitterComponent`; heterogeneous and
homogeneous encoders are not forced through a lowest-common-denominator emitter
API.

The canonical flat relation and flat Horizon encoders both expose concrete
assembly builders. Added components use the ordinary `FlatEmitterComponent`
lifecycle and receive the corresponding public prepared view through
`FlatInputView`. Canonical and downstream components therefore plan nodes and
emit into the same graph-local `FlatRelationSink`:

```cpp
SemanticFlatRelationAssemblyBuilder builder(schema, config);
builder.add_component(std::make_unique<MyNativeRelationComponent>());
auto engine = std::move(builder).compile();

SemanticAnnotations annotations;
annotations.emplace<MyGraphData>("my_data", /* constructor arguments */);
auto batch = engine.encode(SemanticFlatRelationGraphInput(input, std::move(annotations)));
```

The Horizon specialization follows the same lifecycle:

```cpp
SemanticFlatHorizonAssemblyBuilder builder(schema, config);
builder.add_component(std::make_unique<MyNativeHorizonComponent>());
auto engine = std::move(builder).compile();

SemanticAnnotations annotations;
annotations.emplace<MyGraphData>("my_data", /* constructor arguments */);
auto batch = engine.encode(SemanticFlatHorizonInput(dag, std::move(annotations)));
```

`SemanticFlatRelationPreparedGraph` and `SemanticFlatHorizonPreparedGraph` are
read-only. The relation view exposes objects, facts, goals, canonical entity and
target mappings, action/history identities, fact membership, and annotations.
The Horizon view additionally exposes the source DAG, candidate identities,
topology, and exact transition deltas. Both are façades over the canonical
graph preparation consumed by built-ins; no extension-only prepared graph is
copied or maintained. Components must not retain a prepared view or returned
span after a lifecycle callback.

Reference-backed semantic inputs borrow their source through synchronous
`encode` or `encode_batch`; shared-pointer constructors own the source instead.
Annotation values are held as `shared_ptr<const T>`, so copied batch inputs
safely share immutable sidecar data without engine-local or global mutable
state. `SemanticFlatHorizonAnnotations` remains a source-compatible wrapper for
`SemanticAnnotations` and preserves the Horizon-specific exported type.

Compilation freezes canonical and downstream components together. The
resulting engine is immutable and reusable across batches and threads; one
combined node-planning phase, relation sink, relation-field write, and final
relation-major collation are performed for each encoding. Relation aliases and
projections declared by downstream components are resolved against that
combined schema, and canonical relation ids are rebound after compilation so
additional relation names cannot shift built-in emitters onto the wrong ids.
Component ownership is transferred exclusively into the builder. Since a
compiled engine may be used concurrently, component lifecycle callbacks must
remain logically const and thread-safe; graph-local mutable work belongs in
`FlatGraphScratch`.

This generalization intentionally separates lifecycle from packing kernels.
The shared common layer is ready for heterogeneous and homogeneous concrete
builders, but those families need family-specific compiled contexts before they
can offer the same one-pass extension guarantee. Merely accepting common
annotations in their current imperative encoders would not make their schema,
node planning, and relation writes compositional. Flat relation and flat
Horizon are therefore the first complete family implementations of the shared
assembly contract.

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
- The PyTyr Flat, Color, HGraph, and successor encoders run their engine inside
  the PyTyr extension module, so a Tyr state reaches the algorithm as Views and
  only the finished neutral encoding crosses the ABI boundary as a capsule. A
  capsule says where a value crossed, not how it was produced.
- Each family also exposes `prepare(...)` plus a prepared-batch `encode_batch`.
  A `ViewPreparation` owns its compact pools and borrows nothing from the
  backend state, so it is the unit a batch or a stream holds. This is not a loop
  over the single-graph append: the flat batch has genuine cross-graph passes
  (target-name suppression, shared batch constants) that must see every graph
  before any of them is encoded.

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

Direct backend streams resolve each appended step immediately and retain only
planner-neutral state -- a native batch encoding, or a `ViewPreparation` when
the flush needs every graph at once. A caller must keep the planning task,
problem, and View context alive through the append call; a stream must not
retain a lazy range after its source state has been destroyed. Deferred flush
tests should exercise both mutable removal and replacement so a stale View
cannot be observed.

A prepared handle is borrowed by the flush, never consumed, so flushing the same
stream twice is well defined and returns the same encoding.

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
