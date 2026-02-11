# Dynamic Graph Fields

Use dynamic graph fields when you need per-graph attributes collated during native batching.

## Register Specs

```python
from mifrost.graph_fields import GraphFieldSpec, Mode, DType

encoder.register_graph_fields({
    "goal_distance": GraphFieldSpec(mode=Mode.STACK, dtype=DType.F32),
    "target_indices": GraphFieldSpec(mode=Mode.RAGGED_CAT, dtype=DType.I64),
})
```

## Assign Values on EncodedGraph

```python
graph = encoder.encode_graph(state)
graph.goal_distance = 12
graph.target_indices = [0, 3, 5]
```

## Batch Graphs

```python
batch_encoding = encoder.batch_graphs([graph1, graph2])
data = batch_encoding.as_pyg(as_batch=True)
```

Ragged fields expose `<attr>_ptr` tensors (for example `target_indices_ptr`) in PyG conversion.

## Increment Semantics

Use `Inc.node_offset(node_type)` for integer fields that represent node-local indices and require offset adjustment during batching.
