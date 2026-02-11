# Native-First Design

`mifrost` is designed so encoding paths produce native `BatchEncoding` first, with explicit PyG conversion.

Why this helps:

- Keeps the fast path in native columnar buffers.
- Avoids immediate graph-object materialization when not needed.
- Makes conversion an explicit boundary (`as_pyg`, `encode_pyg`, `encode_batch_pyg`, `flush_pyg`).

Recommended pattern:

1. Encode to native `BatchEncoding`.
2. Perform any native batching/serialization operations.
3. Convert to PyG only at model input boundaries.
