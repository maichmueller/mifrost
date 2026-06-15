# mifrost

`mifrost` is a high-performance graph encoding library for planning states and transitions.
It combines C++ encoder engines (via nanobind) with Python-facing APIs.
The API is native-first: encoders return `BatchEncoding` by default, and PyTorch Geometric
(`Data` / `HeteroData`) objects are built on demand.

[![Unit & C++ Tests](https://github.com/maichmueller/mifrost/actions/workflows/tests.yml/badge.svg)](https://github.com/maichmueller/mifrost/actions/workflows/tests.yml)
[![Install Smoke](https://github.com/maichmueller/mifrost/actions/workflows/pip.yml/badge.svg)](https://github.com/maichmueller/mifrost/actions/workflows/pip.yml)
[![Wheel Builds](https://github.com/maichmueller/mifrost/actions/workflows/wheels.yml/badge.svg)](https://github.com/maichmueller/mifrost/actions/workflows/wheels.yml)
[![Performance Monitor](https://github.com/maichmueller/mifrost/actions/workflows/perf-batch-encodings.yml/badge.svg)](https://github.com/maichmueller/mifrost/actions/workflows/perf-batch-encodings.yml)
[![Docs Check](https://github.com/maichmueller/mifrost/actions/workflows/docs-check.yml/badge.svg)](https://github.com/maichmueller/mifrost/actions/workflows/docs-check.yml)

| Workflow | Platforms | Python versions |
| --- | --- | --- |
| Unit & C++ Tests | macOS, Ubuntu | 3.12-3.14 |
| Install Smoke | macOS, Ubuntu | 3.12-3.14 |
| Wheel Builds | macOS, manylinux, musllinux | cibuildwheel `cp3.(12-14)` |
| Docs Check | Ubuntu | 3.12 |

## Documentation

- Website: https://maichmueller.github.io/mifrost/
- Tutorials: https://maichmueller.github.io/mifrost/tutorials/first-encoding/
- Encoder selection guide: https://maichmueller.github.io/mifrost/how-to/choose-an-encoder/
- API reference: https://maichmueller.github.io/mifrost/reference/api/

## What it does

- Encodes planning states into graph structures for GNN pipelines.
- Supports single, batch, and stream-oriented encoding workflows.
- Exposes multiple encoder families:
  - `HGraphEncoder`
  - `HorizonEncoder`
  - `TransitionHGraphEncoder` / `TransitionEffectsHGraphEncoder`
  - `ColorEncoder`
  - `ILGEncoder`
- Returns native `BatchEncoding` objects, with explicit helpers for:
  - PyG conversion (`encode_pyg`, `encode_batch_pyg`, `as_pyg`)

## Requirements

- Python `>= 3.12`
- A working C++ toolchain
- `pymimir>=0.13.60` available in the Python environment used for build/runtime
- For Python-side graph assembly: `torch` and `torch-geometric`
- For source builds: Conan (or `CONAN_COMMAND`/`CONAN_CMD` pointing to it)

### Python dependency files

- `requirements.txt`: runtime Python dependency set
- `requirements/build.txt`: Python tooling for source builds (`conan`, `cmake`, `ninja`, etc.)
- `requirements/test.txt`: test-only Python dependencies
- `requirements/perf.txt`: performance-gate dependencies
- `requirements/dev.txt`: convenience union of build + test + perf + quality tools
- `requirements/constraints-ci.txt`: CI-only version constraints used by workflows
- `pyproject.toml` extras: `.[test]`, `.[perf]`, `.[dev]`

## Installation

### From PyPI

```bash
pip install mifrost
```

### From source (wheel)

```bash
git clone https://github.com/maichmueller/mifrost.git
cd mifrost
pip install .
```

If Conan is not on your `PATH`, set:

```bash
export CONAN_COMMAND=/path/to/conan
```

If `pymimir` is installed but CMake cannot locate it, set:

```bash
export MIFROST_MIMIR_CMAKE_DIR="$(python -c 'import pymimir; print(pymimir.get_cmake_dir())')"
```

### Editable install (development)

```bash
python -m pip install --no-build-isolation \
  --config-settings=editable.rebuild=true \
  -Cbuild-dir=build_editable \
  -e .
```

This enables import-triggered rebuild behavior from scikit-build-core for local development.

### Reusable C++ SDK from the installed package

Installed wheels also ship the reusable native SDK alongside the Python module:

```bash
python -c 'import mifrost; print(mifrost.get_include_dir())'
python -c 'import mifrost; print(mifrost.get_cmake_dir())'
python -c 'import mifrost; print(mifrost.get_library_dir())'
```

Use `mifrost.get_cmake_dir()` in downstream CMake projects to extend `CMAKE_PREFIX_PATH`
or pass `-Dmifrost_DIR="$(python -c 'import mifrost; print(mifrost.get_cmake_dir())')"`.


## Quick start (native-first)

```python
import mifrost

# domain: pymimir wrapper or advanced domain
encoder = mifrost.HGraphEncoder(domain)

# state: pymimir wrapper or advanced state
encoding = encoder.encode(state)         # BatchEncoding
data = encoding.as_pyg()                 # HeteroData

batch_encoding = encoder.encode_batch([state1, state2, state3])  # BatchEncoding
batch = batch_encoding.as_pyg(as_batch=True)                     # HeteroDataBatch

# explicit PyG convenience helpers
data2 = encoder.encode_pyg(state)
batch2 = encoder.encode_batch_pyg([state1, state2, state3])
encoding_dict = batch_encoding.as_dict()      # dictionary form
# note: encoding_dict["tensors"] entries are DLPack-exporting values
# (consume with torch.utils.dlpack.from_dlpack(...) or mifrost.encoding_to_tensors(...))
```

Example output (trimmed):

```text
type(encoding): BatchEncoding
type(data): HeteroData
node_type_count: 28
node_types_head:
  ['[+]on[g]', '_symbol_', 'clear', 'object', 'ontable',
   '[+]clear[g]', '[+]clear[g][sat]', '[+]handempty[g]']
edge_type_count: 54

type(batch_encoding): BatchEncoding
type(batch): HeteroDataBatch
num_graphs: 3
node_type_count: 28

encoding keys:
  ['node_feature_dims', 'node_names', 'num_graphs', 'object_names', 'schema', 'tensors']
schema keys:
  ['edge_tensors', 'edge_types', 'extensions', 'flags', 'graph_kind',
   'node_tensors', 'node_types', 'version']
tensor_count: 164
tensor_keys_head:
  ['_symbol_|0|object/edge_index_0', '_symbol_|0|object/edge_index_1',
   'object|0|_symbol_/edge_index_0', 'object|0|_symbol_/edge_index_1', ...]
```

Stream workflow:

```python
stream = encoder.stream()
stream.append(state1)
stream.append(state2)

batch_encoding = stream.flush()  # BatchEncoding
batch = batch_encoding.as_pyg(as_batch=True)    # HeteroDataBatch

# convenience
batch2 = stream.flush_pyg(as_batch=True)

# mutable stream (supports update/remove)
mutable = encoder.mutable_stream()
sid = mutable.append(state1)
mutable.update(sid, state2)
mutable.remove(sid)
```

Example output:

```text
type(batch): HeteroDataBatch
num_graphs: 2
node_type_count: 28
node_types_head:
  ['[+]clear[g]', '[+]clear[g][sat]', '[+]handempty[g]', '[+]handempty[g][sat]',
   '[+]holding[g]', '[+]holding[g][sat]', '[+]on[g]', '[+]on[g][sat]']
```

## Encoding quick lookup

### C++ (`HGraphEncoderEngine`)
All methods **append into an existing `BatchBuilder`** (they do not clear it). Call `builder.next_graph()` to commit one graph.

- `encode(state, builder)` / `encode_state(state, builder)`
  - State-only graph (objects + current facts).

- `encode_step<GoalTag>(state, goals_span, actions_span, builder)`
  - Convenience overload for typed goal literals (wraps into `GoalInputs` internally).

- `encode(state, goals: GoalInputs, actions_span, builder)`
  - Full step graph (state + goals + optional actions; plus optional derived relations depending on config).

- `encode(state, goals, actions_span, history_subgoals, history_max_steps, builder)`
  - Full step graph plus history nodes/links (`dt`-tagged subgoals).

### C++ streaming

- `HGraphStreamEncoder` (append-only)
  - Direct append into one persistent builder.
  - `append(...) -> id`, `flush()`, `flush_pyg()`, `reset()`.

- `HGraphMutableStreamEncoder` (cached/mutable, via `StreamEncoderBase`)
  - Supports `update/remove` with id stability and cache merge on flush.
  - `append(...) -> id`, `update(id, ...)`, `remove(id)`,
    `flush()`, `flush_pyg()`,
    `reset()`, `set_reuse_removed(bool)`.

---

### Python (`HGraphEncoder`)
- `encode(state, *, ...) -> BatchEncoding`
  - One graph, native encoding object.

- `encode_pyg(state, *, ...) -> HeteroData`
  - One graph, explicit PyG conversion path.

- `encode_batch(states, *, ...) -> BatchEncoding`
  - Many graphs, native batch encoding object.

- `encode_batch_pyg(states, *, ...) -> HeteroDataBatch`
  - Many graphs, explicit PyG conversion path.

- `encode_batch` argument semantics
  - `states` is always the batch axis.
  - Each extra batch argument is either shared (applies to all states) or a per-state
    sequence of length `len(states)` with optional `None` entries.
  - Batch parsing/execution is C++-backed across encoders.
  - High-level `encode_batch(...)` accepts wrapper/native planning inputs and converts
    wrapper/adapter-backed values in Python before entering the C++ batch parser.
  - Low-level `_core._parse_*` helpers and C++ batch internals are strict advanced-only.
  - Unsupported batch kwargs are ignored silently by encoder batch APIs.
  - Per-state argument length mismatches raise `ValueError`.
  - Example:
    - Shared goals/actions:
      - `encoder.encode_batch(states, goals=goals, actions=actions)`
    - Per-state goals/actions:
      - `encoder.encode_batch(states, goals=[goals0, goals1], actions=[[a0], None])`

  - Migration notes (hard break):
    - Old encoder-specific inference/ignoring of batch kwargs was removed.
    - Low-level `_core._parse_*` batch parser helpers no longer accept adapter-backed
      custom Python types.
      - High-level encoder `encode_batch(...)` now performs Python-side adapter conversion.
    - `Transition*Encoder` requires aligned `successors`; unsupported kwargs are ignored.
    - `HorizonEncoder` accepts per-state `dags/goals/subgoal_layers`; unsupported kwargs are ignored.
    - `ColorEncoder` ignores `actions` in batch mode.

- `stream() -> HGraphEncoderStream`
  - Create an append-only stream encoder backed by the same C++ engine.

- `mutable_stream() -> HGraphMutableEncoderStream`
  - Create a mutable stream encoder supporting update/remove.

### Python streaming (`HGraphEncoderStream`, append-only)
- `append(state, *, goals=None, actions=None, subgoal_layers=None, history_subgoals=None, history_max_steps=None) -> id`
- `flush() -> BatchEncoding`
- `flush_pyg(...) -> PyG`

### Python mutable streaming (`HGraphMutableEncoderStream`)
Cached stream wrapper (ids + edits), merges on flush.

- `append(state, *, goals=None, actions=None, subgoal_layers=None, history_subgoals=None, history_max_steps=None) -> id`
- `update(id, state, *, ...)`
- `remove(id)`
- `flush() -> BatchEncoding`
- `flush_pyg(...) -> PyG`

## Extending input types (adapter API)

Encoders are strict for native `pymimir` (advanced) types, but you can register explicit adapters
for custom wrappers:

```python
import mifrost

mifrost.register_state_adapter(MyStateType, lambda s: s.to_advanced_state())
mifrost.register_domain_adapter(MyDomainType, lambda d: d.to_advanced_domain())
mifrost.register_literal_adapter(MyLiteralType, lambda l: l.to_advanced_literal())
mifrost.register_action_adapter(MyActionType, lambda a: a.to_advanced_action())
```

Adapters are matched by **exact concrete type**.

Batch hard-break note:
- High-level `encode_batch(...)` / `encode_batch_pyg(...)` use adapter registries
  during Python-side preprocessing before calling strict C++ batch parsing.
- Direct low-level `_core._parse_*` batch helper calls remain advanced-only and
  reject adapter-backed objects.

## Development

### Configure and build C++ targets

```bash
python configure.py --config Release --build_dir build
python cbuild.py build
```

Or use CMake presets:

```bash
cmake --preset local-release
cmake --build build/local-release
```

Build benchmarks:

```bash
python configure.py --config Release --build_dir build_bench --with_benchmarks
python cbuild.py build_bench --bench
```

Batch collation microbenchmark (`batch_encodings`, including `pyobj` collation):

```bash
python scripts/benchmark_batch_encodings.py --repeats 400 --warmup 50
```

### Tests

Python tests:

```bash
pytest -q
```

C++ tests (after configure/build):

```bash
./build/<...>/src/mifrost_tests
```

### Profiling

```bash
python scripts/profile_encoding.py --domain blocks --problem small
python scripts/profile_encoding.py --profile cprofile --include-goals
python scripts/profile_encoding.py --benchmark-pyg --no-export-node-names
```

Comprehensive scaling benchmark suite (diverse state batches, goals/actions combinations,
transition encoders, horizon DAG-size sweeps):

```bash
python scripts/benchmark_encoder_suite.py \
  --domain blocks \
  --problem smedium \
  --max-states 256 \
  --batch-sizes 8,32,128 \
  --horizon-dag-sizes 8,32,64 \
  --horizon-batch-sizes 4,16 \
  --include-lgan-values false,true \
  --repeats 5 \
  --warmup 1 \
  --output-json /tmp/encoder_bench_suite.json
```

Include PyG conversion paths in the same run:

```bash
python scripts/benchmark_encoder_suite.py --benchmark-pyg
```

## Output assembly notes

- Native-first:
  - `encode(...)` / `encode_batch(...)` return `BatchEncoding`.
  - `BatchEncoding.as_pyg(...)` converts to PyG.
- Convenience:
  - `encode_pyg(...)` / `encode_batch_pyg(...)` return PyG directly.
- `BatchEncoding` supports `as_dict()`, `schema_fingerprint()`, `save(...)`, `load(...)`.
- `mifrost.batch_encodings([...])` batches single-graph encodings natively with schema checks.
- Use `mifrost.encoding_to_tensors(encoding.as_dict())` when feeding a custom downstream pipeline.

## Encoder architecture note

- See `docs/development/architecture.md` for the current Python/native layout
  and encoder architecture notes.

## License

GPL-3.0-only
