# Streaming Model

## Append-Only Streams

- Backed by a persistent builder
- `append(...) -> id`, `flush()`, `flush_pyg()`, `reset()`
- No `update/remove`

## Mutable Streams

- Maintain ID-addressable cache entries
- Support `append`, `update`, `remove`, optional removed-slot reuse
- Flush merges cached single-graph encodings into one batch

## Output Contract

Both stream categories follow the same native-first rule:

- `flush()` returns native `BatchEncoding`
- `flush_pyg()` returns PyG convenience output
