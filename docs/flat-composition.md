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
  emitters share one input; an empty owner is a deliberate broadcast.
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

`FlatCompositionInput` is intentionally a carrier, not a semantic model.
Backend adapters own object/action/goal interpretation and populate it.  They
must resolve each relation key with the compiled schema once per plan (for
example, with `compiled.schema().id_for(...)`) before filling a relation
record.  A record with an unknown id, wrong arity, missing field, or mismatched
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
The neutral C++ suite includes a minimal semantic relation encoder fixture that
compares a composed state-fact carrier against the existing semantic engine
field-for-field; larger semantic migrations should extend that fixture before
switching their public path.

`FlatRelationMajorWriter` is the shared finalization seam.  The semantic flat
relation and horizon engines use the same writer as a composed plan, so
relation-argument collation is not a second, subtly different implementation
during migration.

## External downstream modes

The six `concurrent_internal_*` modes and `FlatCompositeEncoderEngine` that
motivated this layer are downstream consumers, not Mifrost classes.  The
`FlatExternalModeContract` table records their required capabilities so a
downstream adapter can compile a plan without reimplementing batching or
schema logic.  It intentionally does not assert output parity: exact parity
must be tested in the downstream repository against its legacy fixtures.  The
Mifrost baseline remains the existing relation/horizon native and semantic
encoder test suites.
`flat_external_mode_satisfied()` and
`flat_external_mode_missing_components()` let that adapter reject an incomplete
assembly before encoding; they do not instantiate or claim to implement the
downstream modes.

## Compatibility and performance rules

- Compile a plan once per encoder configuration and cache it.
- Use `FlatBatchRuntime` per worker; components must keep compiled state
  immutable (graph-local state belongs in the graph context), and do not append
  already-materialized `BatchEncoding` values to compose components.
- Keep relation ordering, tuple arity, node identity, and graph-field ownership
  deterministic.  Existing backend-specific encoders remain unchanged until a
  component migration has an exact carrier comparator.
- Set `FlatCompositionConfig::relation_args_node_type` when the composed plan
  uses a node table other than `entity`; relation argument fields then receive
  the correct graph offset source.
