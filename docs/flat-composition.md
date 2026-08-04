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

### Semantic flat migration seam

`SemanticFlatCompositionInput` is the source-side seam for semantic flat
composition.  It is deliberately backend-neutral: a semantic adapter can
populate objects, resolved relation records, fields, and metadata directly
from a `SemanticFlatRelationInput` or `SemanticTransitionDAG`, while the
compiled plan owns native emission and finalization.

The semantic relation and horizon engines populate this seam directly from
their semantic inputs. Production encoding executes only the compiled path;
legacy encoders are independent parity oracles used by tests and benchmarks.
The direct path is explicit through
`FlatInputView::from(SemanticFlatCompositionInput)`, and composition or
execution errors propagate to the caller.

The neutral C++ suite includes a minimal semantic relation encoder fixture that
compares a composed state-fact carrier against the existing semantic engine
field-for-field; larger semantic migrations should extend that fixture before
switching their public path.

`mifrost_bench_semantic_flat_composition` compares the composed semantic path
with its legacy-only carrier path on a representative input. The composed
measurement currently includes the legacy parity oracle; it is therefore a
regression baseline for the migration, not a claim of the eventual oracle-free
speedup.

`FlatRelationMajorWriter` is the shared finalization seam.  The semantic flat
relation and horizon engines use the same writer as a composed plan, so
relation-argument collation is not a second, subtly different implementation
during migration.

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
  deterministic.  Existing backend-specific encoders remain unchanged until a
  component migration has an exact carrier comparator.
- Set `FlatCompositionConfig::relation_args_node_type` when the composed plan
  uses a node table other than `entity`; relation argument fields then receive
  the correct graph offset source.
