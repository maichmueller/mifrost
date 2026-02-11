# Schema and Tensor Contract

`BatchEncoding.as_dict()` exposes a normalized dictionary with:

- `schema`: structural description of node/edge/graph tensors
- `tensors`: flat tensor payload keyed by schema keys
- optional metadata such as `node_names`, `object_names`, `graph_attrs`

PyG conversion uses the schema to reconstruct node stores, edge stores, edge indices, and graph-level attributes.

Dynamic graph fields appear under schema `graph_tensors` and are attached as root attributes in PyG output (`attr`, plus optional `attr_ptr` for ragged fields).
