# Native Reference

`mifrost` exposes native encoding objects and helpers that support a native-first pipeline.

## BatchEncoding

Primary methods/properties include:

- `num_graphs`, `num_nodes`, `num_edges`
- `graph_kind`, `node_types`, `edge_types`
- `as_dict()`
- `as_pyg(as_batch=...)`
- `schema_fingerprint()`
- `save(path, include_metadata=False)`
- `BatchEncoding.load(path)`

## Native Helpers

- `mifrost.batch_encodings([...])`: merge single-graph encodings with schema checks.
- `mifrost.encoding_to_tensors(encoding.as_dict())`: convert flat tensor payload to torch tensors.

## Config Surfaces

Key config classes (bindings-defined):

- `HGraphEncoderConfig`
- `HorizonEncoderConfig`
- `SuccessorEncoderConfig`
- `ColorEncoderConfig`
