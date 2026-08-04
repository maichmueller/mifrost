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
5. `finalize_batch_encoding()` optionally performs the existing relation-major
   collation pass.

Virtual dispatch is limited to the component/graph lifecycle.  Fact, action,
goal, and transition loops remain inside native component implementations;
there is no Python callback or per-relation component dispatch in this layer.
`FlatRelationProjection` performs slot remapping with integer relation ids and
supports source slots, constants, and graph-local node references.
Components that emit many tuples should resolve `FlatGraphContext::relation_id`
once before their hot loop and call the integer-id overload of `emit`.

## External downstream modes

The six `concurrent_internal_*` modes and `FlatCompositeEncoderEngine` that
motivated this layer are downstream consumers, not Mifrost classes.  The
`FlatExternalModeContract` table records their required capabilities so a
downstream adapter can compile a plan without reimplementing batching or
schema logic.  It intentionally does not assert output parity: exact parity
must be tested in the downstream repository against its legacy fixtures.  The
Mifrost baseline remains the existing relation/horizon native and semantic
encoder test suites.

## Compatibility and performance rules

- Compile a plan once per encoder configuration and cache it.
- Use `FlatBatchRuntime` per worker; do not append already-materialized
  `BatchEncoding` values to compose components.
- Keep relation ordering, tuple arity, node identity, and graph-field ownership
  deterministic.  Existing backend-specific encoders remain unchanged until a
  component migration has an exact carrier comparator.
- Set `FlatCompositionConfig::relation_args_node_type` when the composed plan
  uses a node table other than `entity`; relation argument fields then receive
  the correct graph offset source.
