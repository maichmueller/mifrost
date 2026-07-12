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

1. Complete the visualization-boundary migration for the flat and color
   families. HGraph and Horizon NetworkX conversion, drawing, and specialized
   target-tree layout now use a private data-only context boundary while their
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
