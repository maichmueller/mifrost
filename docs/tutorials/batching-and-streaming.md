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
- High-level `encode_batch(...)` performs Python-side wrapper/adapter conversion, then
  executes C++-backed batch parsing.
- Low-level `_core._parse_*` batch helpers remain strict advanced-only and reject
  adapter-backed objects.
- Example:
  - shared: `encoder.encode_batch(states, goals=goals, actions=actions)`
  - per-state: `encoder.encode_batch(states, goals=[g0, g1], actions=[[a0], None])`
- Append-only stream intentionally does not provide `update/remove`.
- Mutable stream provides ID-based mutation and merges cached entries at flush.
- Both stream variants return native `BatchEncoding` via `flush()`.
