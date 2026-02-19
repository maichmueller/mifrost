# Schema and Tensor Contract

`BatchEncoding.as_dict()` exposes a normalized dictionary with:

- `schema`: structural description of node/edge/graph tensors
- `tensors`: flat tensor payload keyed by schema keys
- optional metadata such as `node_names`, `object_names`, `graph_attrs`

`tensors` values are exported as DLPack-backed Python values (`__dlpack__`),
so downstream conversion should use DLPack-aware paths (for example
`mifrost.encoding_to_tensors(...)` or `torch.utils.dlpack.from_dlpack(...)`).

PyG conversion uses the schema to reconstruct node stores, edge stores, edge indices, and graph-level attributes.

Dynamic graph fields appear under schema `graph_tensors` and are attached as root attributes in PyG output (`attr`, plus optional `attr_ptr` for ragged fields).

If a python attr and a native graph field share the same key during `as_pyg()`,
the native graph field value is kept and the python attr is skipped. For ragged
native fields, `<attr>_ptr` is reserved by the native field as well.

For native `BatchEncoding` access (without PyG conversion), native graph fields are
also reachable as attributes and through `get_field(key)`. The attribute path
is routed to native storage for known keys.

Python graph-field collation specs (`dtype="pyobj"`) are now validated against native
graph-field keys. Collisions are rejected explicitly instead of being silently shadowed.
