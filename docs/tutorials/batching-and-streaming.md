# Batching and Streaming

## Batch Encoding

Use `encode_batch` when all states are available in memory.

--8<-- "_includes/snippets/hgraph_batch.md"

## Append-Only Stream

Use `stream()` when samples arrive incrementally and only append semantics are needed.

--8<-- "_includes/snippets/hgraph_stream_append_only.md"

## Mutable Stream

Use `mutable_stream()` when you need stable IDs and update/remove semantics.

--8<-- "_includes/snippets/hgraph_stream_mutable.md"

## Semantics

- `encode_batch(states, *, ...)` treats `states` as the only batch axis.
- For batch-capable kwargs, pass either:
  - one shared payload (reused for all states), or
  - a per-state sequence with `len(...) == len(states)` and optional `None` entries.
- Encoder-specific batch kwargs follow the same pattern:
  - transition encoders: `successors=successor`, `successors=[succ0, succ1]`,
    `successors=BatchParam.shared(successor)`, or
    `successors=BatchParam.separate([succ0, succ1])`
  - horizon encoder: `dags=dag`, `dags=[dag0, None]`,
    `dags=BatchParam.shared(dag)`, or `dags=BatchParam.separate([dag0, None])`
    where each DAG leaf may be either `TransitionDAG` or `rustworkx.PyDiGraph`
- High-level `encode_batch(...)` performs Python-side wrapper/adapter conversion, then
  executes C++-backed batch parsing.
- Low-level `_core._parse_*` batch helpers remain strict advanced-only and reject
  adapter-backed objects.
- Example:
  - shared: `encoder.encode_batch(states, goals=goals, actions=actions)`
  - per-state: `encoder.encode_batch(states, goals=[g0, g1], actions=[[a0], None])`
  - transition shared successor: `encoder.encode_batch(states, successors=BatchParam.shared(succ))`
  - horizon per-entry DAGs: `encoder.encode_batch(states, dags=BatchParam.separate([dag0, None]))`
- `rustworkx` support is optional and stays on the Python side:
  `mifrost.transition_dag_from_rustworkx(...)` performs the same conversion explicitly,
  and direct `PyDiGraph` inputs are normalized to `TransitionDAG` before they reach C++.
- The interop path assumes the `PyDiGraph` is already the correct single-root horizon
  DAG/tree. `mifrost` does not run a semantic DAG verification pass during conversion.
- Append-only stream intentionally does not provide `update/remove`.
- Mutable stream provides ID-based mutation and merges cached entries at flush.
- Both stream variants return native `BatchEncoding` via `flush()`.
