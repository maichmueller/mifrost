# Dynamic Graph Fields

Use dynamic graph fields when you need per-graph attributes collated during native batching.

This is a standalone program and its captured output for this version:

--8<-- "_includes/snippets/dynamic_graph_fields.md"

Ragged fields expose `<attr>_ptr` tensors (for example `target_indices_ptr`) in PyG conversion.

## Current Native Behavior

- Native graph fields on `BatchEncoding` are accessible as attributes (`encoding.goal_distance`, `encoding.target_indices`).
- For ragged fields, `<key>_ptr` is reserved and direct assignment to it is rejected.
- In-place mutation on value tensors returned from native access is write-through.
- Ptr tensors are returned as snapshots (mutating the returned ptr tensor does not mutate native storage).

## Collision Rules

Python-side graph-field collation specs (`dtype="pyobj"`) may not reuse native graph-field keys.
Attempting to register a colliding key now raises an explicit `ValueError`.
