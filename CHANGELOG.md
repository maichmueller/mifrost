# Changelog

This project uses Git tags and GitHub releases for release notes.

## Unreleased

- Centralized helper-driven local CMake build modes for release, debug, stubs,
  CI-like tests, and benchmarks.

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
