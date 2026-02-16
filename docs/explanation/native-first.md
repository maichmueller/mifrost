# Native-First Design

`mifrost` is designed so encoding paths produce native `BatchEncoding` first, with explicit PyG conversion.

Why this helps:

- Keeps the fast path in native columnar buffers.
- Avoids immediate graph-object materialization when not needed.
- Makes conversion an explicit boundary (`as_pyg`, `encode_pyg`, `encode_batch_pyg`, `flush_pyg`).
- Allows model-boundary tensor access via `as_hetero()` / `as_homo()` facades without creating PyG objects.

Recommended pattern:

1. Encode to native `BatchEncoding`.
2. Perform any native batching/serialization operations.
3. Use `as_hetero()` / `as_homo()` for tensor-first model code when possible.
4. Convert to PyG only for APIs that require real `Data` / `HeteroData`.
