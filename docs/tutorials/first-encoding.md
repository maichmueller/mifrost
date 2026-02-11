# First Encoding

This walkthrough uses `HGraphEncoder`, the default heterogeneous state encoder.

```python
import mifrost

encoder = mifrost.HGraphEncoder(domain)
encoding = encoder.encode(state)      # native BatchEncoding
data = encoding.as_pyg()              # HeteroData
```

## Batch Workflow

```python
batch_encoding = encoder.encode_batch([state1, state2, state3])
batch = batch_encoding.as_pyg(as_batch=True)
```

## Direct PyG Convenience

```python
data = encoder.encode_pyg(state)
batch = encoder.encode_batch_pyg([state1, state2])
```

## Native-First Recommendation

Use `encode` / `encode_batch` if you want to keep a fast native representation for as long as possible. Convert to PyG only when needed by downstream models.
