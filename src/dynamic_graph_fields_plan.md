# Dynamic Attr Collation Design (Current)

This note reflects the current design after removing the old `EncodedGraph` wrapper path.

## Status

The earlier wrapper-based plan (`EncodedGraph`, `encode_graph(...)`, `batch_graphs(...)`, `register_fields(...)`) is obsolete and no longer matches the implementation.

Current behavior is based on two explicit batching entrypoints:

1. `mifrost.batch_encodings(encodings, collate_spec=None) -> BatchEncoding`
2. `encoder.encode_batch(..., batch_attrs=None, collate_spec=None) -> BatchEncoding`

## Core model

There are now two distinct concepts:

1. Native graph fields
- Declared with `BatchBuilder.register_field(...)`
- Stored in native schema-backed graph tensor storage
- Exported through `BatchEncoding.as_dict()["tensors"]`
- Batched natively by `BatchBuilder.append_batch_encoding(...)`

2. Dynamic Python attrs
- Assigned directly as Python attrs on `BatchEncoding`
- Not promoted into native graph-field schema
- Collated only when an explicit Python batching path is used
- Exposed on the resulting `BatchEncoding` as Python attrs

These are intentionally separate. Python attr collation metadata is not a substitute for native graph-field schema.

## Public batching APIs

### `mifrost.batch_encodings(...)`

Use this to merge single-graph encodings.

- `collate_spec=None`:
  - default collation is applied for non-native Python attrs
  - if all values for a key are mappings: shallow dict value-list collation
  - otherwise: list collation
  - missing keys are errors
  - mixed dict/non-dict values are errors
  - no recursive dict collation

- `collate_spec={...}`:
  - explicit typed collation for selected keys
  - supports `dtype in {pyobj, str, f32, i64}`
  - supports `mode in {stack, cat, ragged_cat, const}`
  - numeric modes honor `dim`, `cat_dim`, and `inc`

Important:
- `collate_spec` must be passed explicitly for typed collation.
- Stored `BatchEncoding.collate_spec()` metadata is not implicitly reused when re-batching.

### `encoder.encode_batch(...)`

Use this when the encoder itself builds the batch.

- `batch_attrs={...}` injects already-collated Python attrs onto the result
- `collate_spec={...}` records read-only metadata describing those injected attrs

This metadata exists for output introspection only. There is no public `register_collate_spec(...)` mutator on `BatchEncoding`.

## `BatchEncoding.collate_spec()`

`BatchEncoding.collate_spec()` returns read-only metadata for dynamic Python attrs that were explicitly collated or explicitly annotated during batch construction.

It does not:
- declare native fields
- validate future writes
- participate implicitly in later batching operations

## Reserved-key rules

Dynamic Python attrs and Python attr collation reject keys that would collide with:

- internal `__mifrost_*` keys
- PyG structural keys such as `x`, `edge_index`, `edge_attr`, `batch`, `ptr`
- native tensor field keys already present in the encoding

## Rationale

This design keeps the boundary explicit:

- native graph fields are schema and storage
- Python attr collation is batching policy

That avoids the main problem in the older wrapper plan: a registration-like API that looked like schema declaration but actually only stored deferred Python-side collation hints.
