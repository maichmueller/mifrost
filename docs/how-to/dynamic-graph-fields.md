# Dynamic Graph Fields

Use dynamic graph fields when you need per-graph attributes collated during native batching.

This is a standalone program and its captured output for this version:

--8<-- "_includes/snippets/dynamic_graph_fields.md"

Ragged fields expose `<attr>_ptr` tensors (for example `target_indices_ptr`) in PyG conversion.
