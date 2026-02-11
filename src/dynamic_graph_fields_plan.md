# Dynamic Graph Fields (PyG-style collation) — TDD Implementation Plan

## Goal

Support **optional, dynamically defined per-graph fields** (e.g. `goal_distance`, `target_indices`) that:
- are assigned with **PyG-like ergonomics** (`enc_graph.goal_distance = 45`)
- collate/batch **natively in C++** (including index offsets like PyG `__inc__`)
- add **effectively zero overhead** when unused

Non-goals (MVP):
- arbitrary nested Python objects across the boundary
- full PyG slicing semantics (`batch[i]` reconstruction)


## Design summary (what we will build)

### A) Typed collation contract (like PyG, explicit)
Each dynamic field key has a stable **spec**:
- `dtype`: `F32 | I64`
- `mode`: `STACK | RAGGED_CAT | CAT | CONST` (MVP uses `STACK`, `RAGGED_CAT`, `CONST`)
- `dim`: feature dim (default `1`)
- `inc`: `NONE` or `NODE_OFFSET(node_type)` (PyG `__inc__` analogue for node indices)

### B) Native representation (ragged uses ptr)
For each field:
- `values` numeric buffer (float32 or int64)
- optional `ptr` (`len = num_graphs+1`) for `RAGGED_CAT`

### C) Schema-driven export to Python assembly
Extend schema (or `schema["extensions"]`) with `graph_tensors` that tells Python how to attach root attrs:
- values key: `__graph__/<attr>`
- ptr key (ragged only): `__graph__/<attr>/ptr`

### D) Python UX (PyG ergonomic)
Introduce `EncodedGraph` wrapper:
- returned by `HGraphEncoder.encode_graph(...)`
- wraps a native `BatchEncoding` (`num_graphs == 1`)
- intercepts `__setattr__` for registered dynamic fields and stores raw values
- converts raw values → typed C++ payload *only on finalize* (batching/export)

Batching entrypoint:
- `HGraphEncoder.batch_graphs(list[EncodedGraph]) -> BatchEncoding`
  - finalizes each graph
  - calls `_core.batch_encodings(...)` (C++ applies collation + `inc`)


## Hard requirements

### R1) “No dynamic attrs” overhead must be negligible
Implementation must be fast-path guarded:
- dynamic-field store is **lazily allocated** (e.g. `unique_ptr` in C++; `None`/empty in Python)
- no extra schema entries or tensor keys are emitted when no fields exist
- Python `_encoding_dict_to_pyg` does nothing extra unless schema includes `graph_tensors`

### R2) Strictness
If a field spec exists, it is enforced:
- dtype/mode/dim mismatches raise with a clear message
- `inc` only allowed for integer dtype
- `STACK` missing policy: error by default (MVP)
- `RAGGED_CAT` missing policy: treat as length 0 (MVP)


## MVP decisions (lock these before coding)

1) **Key naming in flat tensors**:
   - values: `__graph__/<attr>`
   - ptr (ragged): `__graph__/<attr>/ptr`
2) **Tensor shapes** (MVP):
   - values are exported as:
     - 1D if `dim == 1` and mode is `STACK` or `RAGGED_CAT`
     - 2D `[N, dim]` if `dim > 1`
   - ptr is always 1D
3) **Schema location**:
   - add `Schema.graph_tensors` (preferred), serialized to `schema["graph_tensors"]`


## TDD workflow (mandatory)

Implement using red → green → refactor in this order:
1) C++: `STACK` batching
2) C++: `RAGGED_CAT` batching (+ptr)
3) C++: `inc=NODE_OFFSET(node_type)` batching for I64 ragged fields
4) Python: `EncodedGraph` attribute assignment + finalize + `batch_graphs(...)`
5) Python: `as_pyg()` / `_encoding_dict_to_pyg` attaches root attrs + ptrs

Run Python tests (do not bypass rebuild behavior):
- `conda run -n rgtest python -m pytest ...`


## Tests to write first (red)

### T1 — C++: STACK field collation
Purpose: pin `[B]`/`[B,dim]` semantics and ensure fields survive native batching.

Setup:
- create 2 single-graph `BatchEncoding` objects (or use `BatchBuilder` to build them) with:
  - `goal_distance` as `STACK`, `F32`, `dim=1` values: 1.0 and 2.0
- call `_core.batch_encodings([enc0, enc1])`

Assert:
- batched encoding exports `__graph__/goal_distance` tensor with length 2 and values `[1.0, 2.0]`
- schema contains `graph_tensors` entry for `goal_distance`

### T2 — C++: RAGGED_CAT + ptr
Setup:
- graph0: `target_indices=[5,6]`
- graph1: `target_indices=[7]`

Assert:
- values: `[5,6,7]`
- ptr: `[0,2,3]`

### T3 — C++: INC node offset (ragged I64)
Setup:
- two graphs with known `"symbol"` node counts:
  - graph0 has `N0` symbols, graph1 has `N1` symbols
- `target_positions` are local-to-graph indices into `"symbol"` nodes:
  - graph0 positions `[2, 7]`, graph1 positions `[0, 3]`
- spec: `RAGGED_CAT`, `I64`, `inc=NODE_OFFSET("symbol")`

Assert:
- values become `[2,7, N0+0, N0+3]`
- ptr correct

### T4 — Python integration: EncodedGraph assignment + batch_graphs + as_pyg
Setup:
- register specs on encoder
- `g0 = enc.encode_graph(state0)`, `g1 = enc.encode_graph(state1)`
- assign:
  - `g0.goal_distance = 45`
  - `g0.target_indices = [0,3,5]`
  - `g1.goal_distance = 12`
  - `g1.target_indices = [1]`
- `batch_enc = enc.batch_graphs([g0, g1])`
- `data = batch_enc.as_pyg()`

Assert:
- `data.goal_distance` exists with shape `[2]`
- `data.target_indices` exists and `data.target_indices_ptr == [0,3,4]`


## C++ implementation plan (green)

### C1) Data structures
Files:
- `src/_core/mifrost/core/batch_builder.hpp`

Add:
- `enum class GraphFieldDType { F32, I64 };`
- `enum class GraphFieldMode { STACK, CAT, RAGGED_CAT, CONST };`
- `struct GraphFieldInc { enum Kind { NONE, NODE_OFFSET }; Kind kind; std::string node_type; };`
- `struct GraphFieldSpec { GraphFieldDType dtype; GraphFieldMode mode; int dim=1; int cat_dim=0; GraphFieldInc inc; };`
- `struct GraphField { GraphFieldSpec spec; ColumnData values; std::vector<int64_t> ptr; };`

Storage:
- `std::unique_ptr< hash_map<std::string, GraphField> > graph_fields;` in `BatchBuilder`
- `hash_map<std::string, GraphField> graph_fields;` in `BatchEncoding` (encodings are materialized objects)

### C2) Builder ingestion API (minimal)
Files:
- `src/_core/mifrost/core/batch_builder.hpp`
- `src/_core/mifrost/core/batch_builder.cpp`

Add methods:
- `register_graph_field(key, spec)` (creates field entry + validates if exists)
- `set_graph_field(key, scalar/ndarray/list)`:
  - requires registered spec in MVP (avoid inference in C++ initially)
  - appends value(s) into a per-graph staging slot
- `commit_graph_fields()` called from `next_graph()`:
  - for `STACK`: exactly one item required → append to `values`
  - for `RAGGED_CAT`: append items, update `ptr`
  - missing handling per requirements

Implementation detail (keep simple):
- store per-graph staging in a `std::optional`-like flag inside `GraphField` (or a separate staging map keyed by field name) so `next_graph()` can validate “set exactly once” for STACK.

### C3) Build/export and native batching
Files:
- `src/_core/mifrost/core/batch_builder.cpp`

Update:
- `reset()` clears graph_fields only if allocated (R1)
- `next_graph()` calls `commit_graph_fields()` only if store exists (R1)
- `build()` moves committed `graph_fields` into `BatchEncoding`

Update batching:
- `BatchBuilder::append_batch_encoding(const BatchEncoding&)`:
  - merge specs (must match)
  - append values according to mode
  - for ragged: merge ptr by offsetting incoming ptr by `dest.ptr.back()`
  - apply `inc=NODE_OFFSET(node_type)` for I64 values:
    - add `node_offsets[node_type]` to incoming values *before* append
  - handle missing keys:
    - `RAGGED_CAT`: treat missing as empty for that graph
    - `STACK`: error (MVP)


## Schema + Python assembly (green)

### S1) Schema extension
Files:
- `src/_core/mifrost/core/schema.hpp`
- `src/_core/mifrost/core/schema.cpp`

Add:
- `struct GraphTensorSpec { std::string attr; std::string key; std::string ptr_key; GraphFieldMode mode; GraphFieldDType dtype; int dim; GraphFieldInc inc; };`
- `std::vector<GraphTensorSpec> graph_tensors;`

Serialize:
- `schema["graph_tensors"] = [...]` only if non-empty (R1)

### S2) BatchEncoding.as_dict exports graph tensors
Files:
- `src/_core/mifrost/init_hgraph_encoder.cpp`

In `batch_encoding_as_dict(...)`:
- if `encoding.graph_fields` non-empty:
  - add `__graph__/<attr>` tensors to `out["tensors"]`
  - add ptr tensors for ragged fields
  - include `schema.graph_tensors`
- ensure 1D export for `dim==1` graph tensors and for ptr tensors (MVP decision #2)

### S3) Python `_encoding_dict_to_pyg` attaches root attrs
Files:
- `src/mifrost/encoders/common.py`

When schema contains `graph_tensors`:
- for each spec:
  - `setattr(data, attr, tensor(values_key))`
  - if ragged: `setattr(data, f"{attr}_ptr", tensor(ptr_key))`


## Python user API (green)

### P1) Typed specs module
Files:
- `src/mifrost/graph_fields.py` (new)

Provide:
- `Enum Mode, DType`
- `Inc.none()`, `Inc.node_offset(node_type: str)`
- `@dataclass(frozen=True) GraphFieldSpec(...); def to_core_dict(self) -> dict`

### P2) Encoder wrapper UX: EncodedGraph
Files:
- `src/mifrost/encoders/hgraph.py` (or `src/mifrost/encoders/base.py` if generalized)

Add:
- `HGraphEncoder.register_graph_fields(specs: dict[str, GraphFieldSpec])`
- `HGraphEncoder.encode_graph(state, ...) -> EncodedGraph`
- `HGraphEncoder.batch_graphs(graphs: Sequence[EncodedGraph]) -> BatchEncoding`

`EncodedGraph` behavior:
- **lazy encoding wrapper** that stores:
  - the original `state` (and any encode kwargs like goals/actions/history),
  - a reference to the encoder/engine,
  - `_dynamic_values: dict[str, Any]` populated via attribute assignment,
  - an optional cached native `BatchEncoding` produced on first finalize
- `__setattr__` stores values for registered keys into `_dynamic_values`
- `finalize()` (must be pure-native; no PyG construction):
  - creates a fresh `BatchBuilder`, `builder.set_graph_kind("hetero")` (or derive kind from encoder)
  - registers graph field specs onto the builder
  - calls the C++ engine encode into the builder for exactly one graph
  - calls `builder.set_graph_fields(_dynamic_values)` once
  - calls `builder.next_graph()` then `builder.build()` → returns `BatchEncoding(num_graphs==1)` with embedded graph fields
  - caches and returns the encoding

Rationale: attaching fields to an already-materialized `BatchEncoding` without re-encoding would
require new mutation APIs and careful graph-boundary bookkeeping. Lazy encoding keeps the path
simple, correct, and consistent with builder semantics.


## Migration target (after MVP tests pass)

Horizon engine currently uses `graph_attrs` for collatable lists (`target_positions`, `target_indices`, `target_depths`).
After MVP, migrate those to dynamic graph fields:
- `target_positions`: `RAGGED_CAT`, `I64`, `inc=NODE_OFFSET(symbol_type_id)`
- `target_indices`, `target_depths`: `RAGGED_CAT`, `I64`, `inc=NONE`
- keep constants (`target_symbol_prefix`, `parent_relation`) in metadata (`graph_attrs` or CONST field)

Add a regression test that batches two horizon encodings and validates these fields.


## Acceptance checklist (done when all true)
- T1–T4 tests pass.
- `batch_encodings([...])` collates graph fields correctly (STACK + ragged + inc).
- `BatchEncoding.as_dict()` emits no graph-field keys when none exist (R1).
- `common._encoding_dict_to_pyg()` attaches graph fields only when schema includes `graph_tensors` (R1).
- `EncodedGraph` assignment workflow works exactly as in the example below:

```python
import mifrost
from mifrost.graph_fields import GraphFieldSpec, Mode, DType, Inc

enc = mifrost.HGraphEncoder(domain)
enc.register_graph_fields({
    "goal_distance": GraphFieldSpec(mode=Mode.STACK, dtype=DType.F32),
    "target_indices": GraphFieldSpec(mode=Mode.RAGGED_CAT, dtype=DType.I64),
    "target_positions": GraphFieldSpec(
        mode=Mode.RAGGED_CAT, dtype=DType.I64,
        inc=Inc.node_offset(node_type=enc.symbol_type_id),
    ),
})

g0 = enc.encode_graph(state0)
g1 = enc.encode_graph(state1)
g0.goal_distance = 45
g0.target_indices = [0, 3, 5]
g1.goal_distance = 12
g1.target_indices = [1]

batch_enc = enc.batch_graphs([g0, g1])
data = batch_enc.as_pyg()
assert data.goal_distance.shape[0] == 2
assert data.target_indices_ptr.tolist() == [0, 3, 4]
```
