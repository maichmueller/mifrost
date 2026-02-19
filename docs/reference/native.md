# Native Reference

`mifrost` exposes native encoding objects and helpers that support a native-first pipeline.

## BatchEncoding

Primary methods/properties include:

- `num_graphs`, `num_nodes`, `num_edges`
- `graph_kind`, `node_types`, `edge_types`
- `as_dict()`
- `as_pyg(as_batch=...)`
- `as_hetero()`, `as_homo()` (lazy tensor facades without PyG materialization)
- `has_field(key)`, `get_field(key)`
- `set_field(key, value)`, `set_fields({...})`
- `field_specs()`, `register_field_specs({...})`
- `keys()`, `items()` (lightweight runtime introspection)
- `schema_fingerprint()`
- `dumps(include_metadata=True)`, `loads(payload)`
- `save(path, include_metadata=False)`
- `BatchEncoding.load(path)`

Native graph fields are also exposed as attributes (`encoding.target_indices`).
If a key is native, assignment is routed to native graph-field storage rather than
stored as a python shadow attribute.

Ragged field notes:

- Ragged values are keyed at `<key>`, with ptr metadata at `<key>_ptr`.
- Direct assignment to `<key>_ptr` is rejected.
- Assign ragged values as `(values, ptr)` via `set_field` / `set_fields`.
- In-place mutation on returned value tensors is write-through; ptr tensors are returned as snapshots.

Collision policy:

- During `as_pyg(...)`, native graph fields still win over python attrs for the same key.
- Registering python collation specs that collide with native graph-field keys now raises `ValueError`
  (both on `register_field_specs(...)` and `batch_encodings(..., field_specs=...)`).

## Native Helpers

- `mifrost.batch_encodings([...])`: merge single-graph encodings with schema checks.
- `mifrost.encoding_to_tensors(encoding.as_dict())`: convert flat tensor payload to torch tensors.

Tensor payload note:

- `BatchEncoding.as_dict()["tensors"]` exports DLPack-backed tensor values.
- Convert with `mifrost.encoding_to_tensors(...)`, `torch.utils.dlpack.from_dlpack(...)`, or any consumer supporting `__dlpack__`.

## Config Surfaces

Key config classes (bindings-defined):

- `HGraphEncoderConfig`
- `HorizonEncoderConfig`
- `SuccessorEncoderConfig`
- `ColorEncoderConfig`
