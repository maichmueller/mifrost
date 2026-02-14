# First Encoding

This walkthrough uses `HGraphEncoder`, the default heterogeneous state encoder.

The following is a standalone program and its captured output for this version:

--8<-- "_includes/snippets/hgraph_basic.md"

## Batch Workflow

--8<-- "_includes/snippets/hgraph_batch.md"

## Direct PyG Convenience

The native-first examples above are intentionally stable and compact. For PyG conversion convenience helpers, see:

- `HGraphEncoder.encode_pyg(...)`
- `HGraphEncoder.encode_batch_pyg(...)`

## Native-First Recommendation

Use `encode` / `encode_batch` if you want to keep a fast native representation for as long as possible. Convert to PyG only when needed by downstream models.
