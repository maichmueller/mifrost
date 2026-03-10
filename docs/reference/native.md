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
- `collate_spec()` (read-only batching metadata)
- `keys()`, `items()` (lightweight runtime introspection)
- `schema_fingerprint()`
- `dumps(include_metadata=True)`, `loads(payload)`
- `save(path, include_metadata=False)`
- `BatchEncoding.load(path)`

Native graph fields are also exposed as attributes (`encoding.target_indices`).
If a key is native, assignment is routed to native graph-field storage rather than
stored as a python shadow attribute.

`target_indices` semantics are encoder-dependent:
- `HorizonEncoder`: `TransitionDAG` node indices.
- `HGraphEncoder` with `target_sources=["action"]`: per-graph action input positions.
- `FlatRelationEncoder` with `target_sources=["action"]`: per-graph action input positions.

`target_candidate_ids` is row-aligned with `target_indices` and `target_positions`:
- `HorizonEncoder`: explicit `TransitionDAG` node `candidate_id` values when provided for all
  emitted targets; otherwise falls back to DAG node indices.
- `HGraphEncoder` with `target_sources=["action"]`: per-graph action input positions.
- `FlatRelationEncoder` with `target_sources=["action"]`: per-graph action input positions.

`BatchEncoding.as_pyg(...)` exposes native graph attrs as top-level PyG attrs
(`data.target_names`, `data.target_symbol_prefix`, etc.). Collisions with
reserved/structural PyG keys raise an error.

Ragged field notes:

- Ragged values are keyed at `<key>`, with ptr metadata at `<key>_ptr`.
- Direct assignment to `<key>_ptr` is rejected.
- Assign ragged values as `(values, ptr)` via `set_field` / `set_fields`.
- In-place mutation on returned value tensors is write-through; ptr tensors are returned as snapshots.

Collision policy:

- During `as_pyg(...)`, native graph fields still win over python attrs for the same key.
- Registering python collation specs that collide with native graph-field keys now raises `ValueError`
  when passed via `batch_encodings(..., collate_spec=...)`.

Python-side collation notes:

- `collate_spec()` is informational metadata stored on already-batched outputs.
- It is not a public registration surface and is not implicitly reused when re-batching.
- To control dynamic-attr collation, pass `collate_spec=...` explicitly to:
  - `mifrost.batch_encodings([...], collate_spec=...)`
  - `encoder.encode_batch(..., batch_attrs=..., collate_spec=...)`

## Native Helpers

- `mifrost.batch_encodings([...])`: merge single-graph encodings with schema checks.
- `mifrost.encoding_to_tensors(encoding.as_dict())`: convert flat tensor payload to torch tensors.
- `mifrost.transition_dag_from_rustworkx(graph, *, fallback_missing_candidate_id_to_node_index=False)`:
  convert `rustworkx.PyDiGraph` to native `TransitionDAG` (delegates to classmethod).
- `mifrost.TransitionDAG.from_rustworkx(graph, *, fallback_missing_candidate_id_to_node_index=False)`:
  native classmethod implementing the rustworkx-to-`TransitionDAG` conversion path.

`TransitionDAG.register_transition(...)` accepts:
- `register_transition(parent, child, action=None, candidate_id=None)`
- `register_transitions(records)` where each record may be either
  `(parent, child, action)` or `(parent, child, action, candidate_id)`.

Tensor payload note:

- `BatchEncoding.as_dict()["tensors"]` exports DLPack-backed tensor values.
- Convert with `mifrost.encoding_to_tensors(...)`, `torch.utils.dlpack.from_dlpack(...)`, or any consumer supporting `__dlpack__`.

## Config Surfaces

Key config classes (bindings-defined):

- `HGraphEncoderConfig`
- `HorizonEncoderConfig`
- `SuccessorEncoderConfig`
- `ColorEncoderConfig`

LGAN config note:

- Main state lanes (`HGraphEncoder`, `FlatRelationEncoder`) separate
  `target_sources` from `lgan_anchor_sources`.
- Horizon and transition lanes use candidate state rows as LGAN anchors and do
  not expose `lgan_anchor_sources`.
