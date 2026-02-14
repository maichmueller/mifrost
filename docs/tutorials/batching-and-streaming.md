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

- Append-only stream intentionally does not provide `update/remove`.
- Mutable stream provides ID-based mutation and merges cached entries at flush.
- Both stream variants return native `BatchEncoding` via `flush()`.
