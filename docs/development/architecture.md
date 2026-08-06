# Architecture Overview

`mifrost` uses a `src/` layout with a thin public Python package on top of an
internal native core.

## Repository Layout

- `src/mifrost/` is the public Python package imported by users.
- `src/_core/mifrost/` contains the native implementation, nanobind bindings,
  schema helpers, and encoder engines.
- `src/CMakeLists.txt` ties the native sources into the build.
- `src/mifrost/_core.pyi` provides the public stub for the extension module.
  Nanobind generates it, git ignores it, CI regenerates it before packaging,
  and `pyproject.toml` includes the generated file in release artifacts.

The Python layer stays intentionally small: it exposes the user-facing API,
package-level helpers, and convenience wrappers, while the native layer owns
the actual encoding and batching logic.

## Runtime Shape

1. Python callers construct encoder objects from `src/mifrost/`.
2. Those wrappers delegate to the native core in `src/_core/mifrost/`.
3. Native encoders produce `BatchEncoding` as the primary output.
4. Python helpers convert that native output to PyTorch Geometric objects only
   when needed.

This keeps the package native-first without changing the boundary between the
Python API and the C++ implementation.

## Dependency Boundaries

The encoder facade has three intentionally narrow boundaries. New encoders
should reuse these boundaries instead of reproducing their implementation.

### Input adaptation

`mifrost.encoders.types` owns the adapter registries and conversion of one
domain object. `mifrost.encoders._batch_contract.prepare_core_batch_inputs`
owns recursive adaptation of the standard batch lanes (`states`, `goals`,
`actions`, `subgoal_layers`, `history_subgoals`, and `successors`). It preserves
shared/per-state `BatchParam` structure while adapting only leaf values.

Python encoder facades validate lane-specific policy, call this boundary once,
and pass its result to a native parser or engine. They must not implement their
own recursive wrapper-to-native traversal.

### Encoding and conversion

`EncoderBase` owns the non-stream lifecycle and `StreamEncoderBase` owns the
stream lifecycle. Their shared internal conversion boundary accepts either a
native `BatchEncoding` or a normalized encoding dictionary and produces PyG
data. A custom encoder can override `_dict_to_pyg`; native encoders retain the
direct native conversion fast path.

Unknown runtime keyword arguments fail at this boundary. An encoder that adds a
runtime option must declare it in `_accepted_kwargs` and consume it in `_encode`
or `_encode_batch`. A lane that extends its parent's implementation should
compose `super()._accepted_kwargs()`; an adapter lane should declare only the
arguments that reach its own implementation. This makes spelling mistakes
visible without leaking parent-only implementation arguments.

### Native implementation

Bindings translate Python arguments and expose engine/config objects. Planning,
schema construction, graph emission, and batch collation stay in
`src/_core/mifrost/`. Native code must not depend on the Python facade. Shared
native behavior belongs in `core/` or `input_handling/`, not in multiple binding
initializers.

### Header layering

The encoder stack is layered strictly bottom-up, and each layer may only
depend on the ones above it in this list:

```text
core/semantic/records.hpp          semantic record and key definitions
    core/views/concepts.hpp        View concepts
    core/views/semantic_preparation.hpp   borrowed inputs, compact pools
        core/encoders/<family>/    family preparation and canonical algorithms
            backends/<backend>/    backend adapters and exported engines
```

`core/semantic/records.hpp` holds `SemanticAtom`, `SemanticLiteral`,
`SemanticGroundAction`, `SemanticHistoryEntry`, `SemanticTaskContext`, the
owning `SemanticFlatRelationInput` compatibility DTO, and their hash/ordering
helpers. It depends on no encoder, batch, or View header.
`semantic_flat_relation_encoder.hpp` includes it and re-exports every name from
namespace `mifrost`, so the exported ABI and existing includes are unchanged.

The View layer must not include an encoder header. It used to include the Flat
encoder header while the Flat encoder header includes the Flat View bridge,
which includes the View layer again; that cycle made the headers
order-sensitive. `tests/cpp/view_preparation_scaling_test.cpp` includes
`core/views/semantic_preparation.hpp` and nothing else from the encoder stack,
so the cycle cannot silently return.

### Planning Views

The canonical encoder algorithms use the operation-bearing concepts in
`mifrost/core/views/concepts.hpp`. `AtomView`, `LiteralView`,
`GroundActionView`, and `StateView` expose only the IDs, ranges, and predicates
that an algorithm needs; they do not own repository objects and do not use a
virtual base class. `mifrost/core/views/canonical.hpp` contains statically
dispatched traversal and satisfaction primitives shared by backend
instantiations.

Each backend has a task-scoped context and lazy Views:

- `backends/pytyr/views.hpp` borrows the PyTyr planning task and its compact
  repository-index tables.
- `backends/pymimir/views.hpp` borrows a Pymimir problem and builds compact
  lookup tables for that problem.
- `core/semantic/views.hpp` adapts retained semantic records for neutral
  algorithms and tests without introducing another owning model.

View values are cheap, copyable handles. The backend task/problem and the
context must outlive every View and every lazy range derived from it. Native
templates are instantiated separately in each adapter, preserving PyTyr and
Pymimir ABI isolation while sharing the algorithm source.

### PyTyr and the ABI boundary

PyTyr and Pymimir link different nanobind ABI generations, so a Tyr state
cannot be passed to an engine object owned by the core extension module: the
two modules do not share a type registry. That constraint is about *nanobind*,
not about C++. Both modules link the same neutral library, so the canonical
engines are ordinary C++ objects that either module can construct and call.

The PyTyr direct-View encoders (`_NativeDirectFlatEncoder`,
`_NativeDirectColorEncoder`, `_NativeDirectHGraphEncoder`,
`_NativeDirectSuccessorEncoder`) exploit exactly that: the engine is constructed
and run inside the PyTyr module, where Tyr types are visible, so a state and its
actions reach the canonical algorithm as granular Views. Only the finished,
planner-neutral `BatchEncoding` crosses back, as a capsule. No owning
`SemanticFlatRelationInput` is built for a normal encode.

Every PyTyr family that has a direct encoder now routes its public runtime
through it: single encode, batch, and stream, for Flat, Color, HGraph and the
successor family.

A runtime keeps two engines -- the compatibility one and the direct encoder's
own instance -- so `update_relations` has to reach both. Updating only the
compatibility engine leaves every encode on the arity table the direct encoder
was constructed with, and the result stays internally consistent, so only a
comparison against the compatibility engine reveals it.

Batches and streams use the same boundary. A batch prepares and encodes every
state in one crossing. A stream cannot do that -- it must hold graphs between
appends -- so each appended step is prepared immediately into a
`ViewPreparation`, which owns compact pools and borrows nothing from the Tyr
state; the handle travels as a capsule that the flush *borrows* rather than
consumes, so the same handles can be flushed repeatedly.

Three things still cross as owned records, for reasons that are not removable by
restructuring:

- Goal, subgoal, and history literals arrive from Python as compact tuples.
  There is no native planning value to borrow from, so they are expanded into
  `SemanticLiteral` vectors and then borrowed by semantic Views.
- `append_into_builder` takes a `BatchBuilder` registered in the core module's
  nanobind registry, which the PyTyr module cannot accept. This is the one path
  where the ABI split itself, not the encoding, forces an owned input.
- `make_input` / `make_inputs` remain the explicit compatibility route for
  callers that want the semantic records themselves, and the horizon and
  flat-horizon families still encode from owned inputs -- a horizon DAG is
  itself an owned semantic snapshot.

### Direct and compatibility encoder paths

Native backend entry points use a direct path whenever the input is still a
borrowed planning value:

```text
backend values
  -> granular borrowed Views
  -> canonical statically dispatched family algorithm
  -> graph-derived intern/index/working structures
  -> BatchBuilder
```

The Pymimir Flat, Color, HGraph, successor, batch, and stream entry points use
this path, and so do the PyTyr Flat, Color, HGraph, and successor entry points,
including their batch and stream forms -- see
"PyTyr and the ABI boundary" below for how a direct path is possible across two
nanobind ABI generations. The semantic engines also retain an explicit
compatibility path for owned `SemanticFlatRelationInput` records. That path is
required by capsules, semantic transition DAGs, and callers that intentionally
snapshot inputs:

```text
owned semantic records -> semantic compatibility encoder -> BatchEncoding
```

The Color, HGraph, and successor direct overloads traverse granular Views before
dispatching to the mature graph emitters. View traversal builds only compact
graph-working records (flattened goal levels, unique action records, filtered
history, state facts, and fact membership) needed for emission; it does not
create a complete semantic-record mirror. Compatibility
`SemanticFlatRelationInput` values bypass that View preparation and are consumed
through borrowed references to their existing lanes, so compatibility encoding
does not copy the complete input into another graph carrier. This keeps
planner-library types out of the neutral core without duplicating backend
algorithms.

Every family has exactly one canonical algorithm, templated on the input and
instantiated for both the borrowed preparation and the owning compatibility
DTO. There is no second algorithm per backend, and no runtime storage-mode
flag reaches per-element code.

- Color, HGraph and the successor encoder template `encode_impl` on the input
  type; the lane accessors (`semantic_state_facts`, `semantic_actions`,
  `semantic_history`, ...) resolve statically.
- Flat declares two concrete borrowed sources, `ViewSource` and
  `CompatibilitySource`, and templates `validate_source`, `make_context` and
  `prepare_source` over them. Preparation is the only code that knows which
  kind of input is being encoded; it produces a fully resolved
  `PreparedRelationGraph` — borrowed spans and pointers settled once, plus
  compact graph-derived working state — so the emitters have nothing left to
  test.

Direct does not mean that the final graph is emitted without planning state.
The neutral engine still creates a graph-local preparation object for schema
ordering, validation, and relation emission. Unlike a compatibility input, that
object is private to the encode call, is not a public semantic record, and does
not retain planner-native values.

The templated adapter boundary is deliberately granular: state, goals,
subgoal layers, actions, and history are accepted as constrained View ranges,
then traversed synchronously into the encoder's fixed graph-key preparation
state. The public `SemanticFlatRelationInput` remains the owned compatibility
representation; direct View calls do not expose another owning DTO or callback
range. New canonical algorithms should continue to use the operation-based View
concepts directly and should not introduce backend-specific type erasure into
those concepts.

### What is borrowed, what is owned

These five categories are deliberately distinct. "Preparation" never means a
full per-lane owning copy of the input.

| Category | Example | Lifetime |
| --- | --- | --- |
| Borrowed input range | `NativeGoalLiteralsView`, `StateView`, action `TransformRange` | Valid only while the backend problem/state, the task context, and the `views::Context` are alive. Consumed synchronously inside one encode call. |
| Compact graph-derived pool | `ViewPreparation::atom_pool` / `action_pool` plus their hash indices, goal-level refs, filtered history refs, fact-membership set | Private to one encode call. Holds each unique graph identity once; lanes keep only indices, so lane order and multiplicity survive deduplication. |
| Owning compatibility DTO | `SemanticFlatRelationInput` | Owned by the caller. Used by capsules, semantic transition DAGs, and callers that intentionally snapshot. Consumed through borrowed references to its existing lanes, not copied into another carrier. |
| Capsule / ABI snapshot | PyTyr `_make_input_capsule`, `_encode_capsule` | Required because Pymimir and PyTyr are built against incompatible nanobind ABI generations. A capsule is an explicit ABI transport, never evidence that a path is or is not View-based: `_make_input_capsule` transports an owning input, while `_encode_capsule` transports only the finished neutral encoding produced by a direct-View encode. |
| Stream lifetime snapshot | `HGraphStreamEncoder`, flat/horizon stream caches | Stores the completed native batch encoding, never a lazy View, so the source state may be released after `append` returns. |

Task contexts and backend planning repositories must remain alive through every
encode or stream append that consumes a View. Batch adapters materialize each
input while its source ranges are alive and retain only the resulting native
batch encoding.

Two lifetime rules are enforced in the types rather than by convention:

- `NativeGoalLayersView` owns its occupied-level list, so
  `adapter.make_goal_views(goals).subgoal_layers_view()` -- where the
  `NativeGoalViews` is a temporary -- is safe. The literal spans it carries are
  still borrowed from the `GoalInputs`, which must outlive the iteration.
- Interning is a lookup index, never an ordering. `atom_pool` and `action_pool`
  keep first-use insertion order and are the only things iterated;
  `atom_indices` / `action_indices` exist purely for O(1) lookup.

### Goal levels: sparse by default

Native goal Views represent goal levels sparsely: `NativeGoalLayersView` visits
only occupied levels, so a single goal at a very high level costs one entry
rather than a dense run of empty layers.

`SemanticFlatRelationInput::subgoal_layers` is positional, so the compatibility
conversion in `SemanticProblemAdapter::make_input` has to build a dense vector.
That conversion takes the *consuming encoder's* configured `max_goal_level` from
its caller, and additionally clamps to `kDenseGoalLayerTransportLimit`, which is
a transport-safety bound on the vector -- not an encoder capability limit. Each
encoder family independently rejects levels it cannot represent, using its own
configuration, before suffix or schema indexing. No family's limit is baked into
the backend-neutral adapter.

### Lane-aware preparation

Preparation populates only the lanes the selected path actually reads. The
successor side of the successor-HGraph algorithm reads the object table and the
successor state facts and nothing else, so it uses
`canonical::detail::make_state_only_view_preparation` rather than building
default-goal, action, and history records that are immediately discarded.

## Adding an Encoder Family

1. Add or reuse a native config and engine with a stable, batch-oriented entry
   point.
2. Define the supported optional lanes in `_lane_specs.py` and validate them at
   the facade boundary.
3. Reuse `prepare_core_batch_inputs` for standard batch lanes. Add a new lane to
   that boundary only when it has repository-wide meaning.
4. Declare non-standard runtime arguments through `_accepted_kwargs`.
5. Test observable behavior through `encode`, `encode_batch`, and the stream
   boundary, including wrapper/advanced-object parity and malformed input.
6. Add the encoder to the public export table, generated API reference, and
   encoder coverage documentation.

## Improvement Roadmap

The repository has strong native/Python separation and broad behavior tests.
The remaining architectural risk is concentrated in a few large modules rather
than spread across the package. Priorities are:

1. Keep the visualization and encoder boundaries aligned. Flat, Color, HGraph,
   Horizon HGraph, successor, and transition families now adapt Pymimir and
   PyTyr through task-scoped Views and canonical semantic engines while their
   public method signatures remain stable.
2. Decompose the largest native encoder translation units by stable concepts:
   schema construction, traversal/planning, and graph emission. Keep the engine
   class as the small public interface and add boundary tests before moving
   implementation details.
3. Expand static typing from the core public types to all Python facade modules
   incrementally. Each newly covered module should first eliminate `Any` at its
   public boundary rather than adding blanket ignores.
4. Split benchmark orchestration from result formatting and baseline comparison
   so each can be tested without running the native benchmark suite.

## Tests

- Python test suites live in `tests/encoding/`, `tests/python/`, and
  `tests/native/`.
- Native behavior and binding coverage live in `tests/cpp/`.
- Shared test helpers live at `tests/conftest.py` and `tests/parity_utils.py`.

## Documentation

- User-facing dynamic graph field behavior: `docs/how-to/dynamic-graph-fields.md`
- Historical design note for dynamic attr collation:
  `docs/development/dynamic_graph_fields_plan.md`
- Build and test workflow: `docs/development/build-and-test.md`
