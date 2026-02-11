# Batching and Streaming

## Batch Encoding

Use `encode_batch` when all states are available in memory.

```python
enc = encoder.encode_batch(states)
pyg_batch = enc.as_pyg(as_batch=True)
```

## Append-Only Stream

Use `stream()` when samples arrive incrementally and only append semantics are needed.

```python
stream = encoder.stream()
stream.append(state1)
stream.append(state2)
enc = stream.flush()
```

## Mutable Stream

Use `mutable_stream()` when you need stable IDs and update/remove semantics.

```python
mstream = encoder.mutable_stream()
sid = mstream.append(state1)
mstream.update(sid, state2)
mstream.remove(sid)
enc = mstream.flush()
```

## Semantics

- Append-only stream intentionally does not provide `update/remove`.
- Mutable stream provides ID-based mutation and merges cached entries at flush.
- Both stream variants return native `BatchEncoding` via `flush()`.
