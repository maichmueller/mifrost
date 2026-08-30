# Changelog

This project uses Git tags and GitHub releases for release notes.

## Unreleased

## 0.6.0 - 2026-08-30

- Centralized helper-driven local CMake build modes for release, debug, stubs,
  CI-like tests, and benchmarks.

### Encoders

### Breaking / behavioral

- `BatchBuilder.set_graph_attr` now rejects writing a *different* value under an
  existing key within one batch (previously silent last-write-wins); identical
  values remain no-ops. Heterogeneous batch `ptr`/`batch` metadata for node
  types first seen in later graphs is now padded correctly instead of
  attributing their nodes to earlier graphs. Derived-graph encoders validate
  object indices against the problem's object table instead of failing
  silently (or opaquely) on out-of-range arguments.

- Added the derived-graph family of homogeneous encoders for vanilla GNN
  pipelines: `StarGraphEncoder`, `ObjectGraphEncoder`, `AtomLineGraphEncoder`,
  `HypergraphIncidenceEncoder`, `TransformerBiasEncoder`, and
  `TupleTensorEncoder` (each with a `*Stream` variant). All facades expose
  visualization helpers (`to_networkx`, `draw`, `summarize`) and convert to
  cross-stack payloads via the DGL/Jraph adapters (`to_dgl`, `to_jraph`).
- Added the pure-Python custom encoder toolkit `mifrost.encoders.custom`:
  planner-neutral records (`StateView`, `Atom`, `Literal`, ...), interning
  tables (`Vocabulary`, `NodeTable`, `EdgeSink`), a `GraphWriter`, composable
  `CustomGraphEncoder`/`CustomStream` with batching and streaming, and a
  verification harness (`assert_backend_parity`, `channel_summary`,
  `conformance_smoke`).
- Added `LiftedTaskEncoder`: encodes the planning problem's own lifted
  structure (predicate/action schemas, parameters, preconditions, effects,
  goal) as one GNN-consumable graph; state-independent unless
  `include_state_facts=True`.
- Added `ObjectFeatureEncoder`: compact objects-only graph that keeps unary
  predicates as per-object feature channels instead of dropping them.
- Performance: memoized state-fact preparation, batch fast lanes in the
  custom toolkit, and transparent interning plus buffer reserves in the
  derived-graph engine.
- Fixes: single/batch conversion consistency across the derived family,
  PyTyr action-structure parity with the Pymimir lanes, and history lanes now
  honor `history_max_steps`.

### Backend migration

- Added interchangeable, per-instance Pymimir and PyTyr runtimes for every
  public encoder family. Backend selection can be inferred from the constructor
  input or stated explicitly with `backend="pymimir"` / `backend="pytyr"`;
  both planners can be used in one process without global backend state.
- Split the native package into a planner-free `_neutral_core` plus optional
  `_pymimir_adapter` and `_pytyr_adapter` modules. Install planners with the
  `pymimir`, `pytyr`, or `backends` extras, and select source-build adapters
  with `MIFROST_BUILD_BACKENDS=core|pymimir|pytyr|both`.
- Added native-only C++ SDK builds: `MIFROST_BUILD_PYTHON=OFF` now works with
  either adapter independently. Historical Pymimir CMake targets and public
  include paths remain available through compatibility aliases/forwarders.
- Made `BatchEncoding`, PyG conversion, native batching, serialization,
  checkpointing, and schema fingerprints planner-neutral. Compatible outputs
  from Pymimir and PyTyr can be mixed in one batch and learning pipeline.
- Added backend-neutral Horizon DAG conversion when a raw
  `rustworkx.PyDiGraph` is passed to a selected encoder. The standalone
  `transition_dag_from_rustworkx(...)` helper remains Pymimir-returning for
  backward compatibility.
- Made `update_relations(...)` planner-neutral for HGraph, Horizon, and
  transition encoders, including topology-relation preservation for Horizon.

### Migration notes and compatibility

- Existing Pymimir encoder construction and observable fast paths remain
  supported. Legacy `register_*_adapter` hooks still convert custom wrappers to
  `pymimir.advanced` values; they are not a global backend mechanism and do not
  participate in PyTyr runtimes.
- Planner identities and repository indices are no longer serialization or
  cross-backend identities. Parity and persisted schemas use normalized
  semantic predicate, object, atom, literal, and action keys.
- PyTyr and Pymimir may expose equivalent predicates/facts in different orders.
  Pymimir compatibility paths preserve their historical order; cross-backend
  comparisons should map indices back to semantic names.
- Semantic Horizon goal satisfaction now retains predicate category in atom
  identity. This corrects a legacy Pymimir false-positive when static, fluent,
  and derived repository indices collide.
- `data/pddl/blocks_eq/domain.pddl` contains malformed syntax accepted by
  Pymimir's lenient parser but rejected by PyTyr. Use a valid equality fixture
  for cross-backend workflows; the existing file remains for compatibility
  regression coverage.
- Unknown encoder kwargs raise `TypeError`. Unsupported non-empty action or
  history lanes on Color, Horizon, and transition families raise descriptive
  `ValueError`s instead of being silently ignored.
