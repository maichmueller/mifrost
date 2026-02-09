# mifrost

`mifrost` is a high-performance graph encoding library for planning states and transitions.
It combines C++ encoder engines (via nanobind) with Python-facing APIs that produce
PyTorch Geometric (`Data` / `HeteroData`) outputs.

## What it does

- Encodes planning states into graph structures for GNN pipelines.
- Supports single, batch, and stream-oriented encoding workflows.
- Exposes multiple encoder families:
  - `HGraphEncoder`
  - `HorizonEncoder`
  - `TransitionHGraphEncoder` / `TransitionEffectsHGraphEncoder`
  - `ColorEncoder`
  - `ILGEncoder`
- Returns either:
  - ready-to-use PyG objects, or
  - normalized encoding "parts" for custom assembly.

## Requirements

- Python `>= 3.10`
- A working C++ toolchain
- `pymimir` available in the Python environment used for build/runtime
- For Python-side graph assembly: `torch` and `torch-geometric`
- For source builds: Conan (or `CONAN_COMMAND`/`CONAN_CMD` pointing to it)

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


## Quick start

```python
import mifrost

# domain: pymimir wrapper or advanced domain
encoder = mifrost.HGraphEncoder(domain)

# state: pymimir wrapper or advanced state
data = encoder.encode(state)             # HeteroData
batch = encoder.encode_batch([state1, state2, state3])  # batched HeteroData

parts = encoder.export_batch_encoding([state1, state2])    # normalized parts payload
```

Example output (trimmed):

```text
type(data): HeteroData
node_type_count: 28
node_types_head:
  ['[+]on[g]', '_symbol_', 'clear', 'object', 'ontable',
   '[+]clear[g]', '[+]clear[g][sat]', '[+]handempty[g]']
edge_type_count: 54

type(batch): HeteroDataBatch
num_graphs: 3
node_type_count: 28

parts keys:
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
batch = stream.flush(as_batch=True)
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

### C++ streaming (`HGraphStreamEncoder` via `StreamEncoderBase`)
Cached stream with ids (supports edits). Encodes each step into a single-graph cache entry and **merges on flush**.

- `append(...) -> id` (state-only or state+goals+actions, optionally with history)
- `update(id, ...)` (same overload set as `append`)
- `remove(id)`
- `flush_batch_encoding_py()` / `flush_batch_encoding()` / `flush()` (PyG)
- `reset()`, `set_reuse_removed(bool)`

---

### Python (`HGraphEncoder`)
- `export_encoding(state, *, goals=None, actions=None, subgoal_layers=None, history_subgoals=None, history_max_steps=None) -> parts`
  - One graph, normalized tensor payload (no PyG assembly).

- `encode(state, *, ...) -> HeteroData`
  - One graph, assembled PyG object.

- `export_batch_encoding(states, *, goals=None, actions=..., subgoal_layers=None, history_subgoals=..., history_max_steps=None) -> parts`
  - Many graphs, one-pass builder accumulation (fast batch path).

- `encode_batch(states, *, ...) -> HeteroData`
  - Many graphs, assembled PyG batch.

- `stream() -> HGraphEncoderStream`
  - Create a streaming encoder backed by the same C++ engine.

### Python streaming (`HGraphEncoderStream`)
Cached stream wrapper (ids + edits), merges on flush.

- `append(state, *, goals=None, actions=None, subgoal_layers=None, history_subgoals=None, history_max_steps=None) -> id`
- `update(id, state, *, ...)`
- `remove(id)`
- `flush(...)` (from `StreamEncoderBase`, returns PyG; supports `as_batch=True` in the base API)

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

## Development

### Configure and build C++ targets

```bash
python configure.py --config Release --build_dir build
python build.py build
```

Build benchmarks:

```bash
python configure.py --config Release --build_dir build_bench --with_benchmarks
python build.py build_bench --bench
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
python scripts/profile_encoding.py --domain blocks --problem probBLOCKS-4-0
python scripts/profile_encoding.py --profile cprofile --include-goals
```

## Output assembly notes

- `encode(...)` / `encode_batch(...)` assemble PyG objects directly.
- `export_encoding(...)` / `export_batch_encoding(...)` return normalized payloads:
  - `schema`
  - `tensors`
  - optional metadata (`node_names`, `object_names`, `graph_attrs`)
- Use `mifrost.parts_to_tensors(parts)` when feeding a custom downstream pipeline.

## Encoder architecture note

- See `src/ENCODER_INHERITANCE_ARCHITECTURE.md` for the C++/Python inheritance
  and config layering used by hetero encoders.

## License

GPL-3.0-only
