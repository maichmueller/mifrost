# Performance Guidance

## Prefer Native-First

Use native `BatchEncoding` as long as possible:

- `encode` / `encode_batch` / `flush` for native flow
- `as_pyg(...)` only when needed by model training/inference code

## Tune Metadata Export

- `export_node_names=False` can reduce metadata overhead.
- `include_metadata=False` for conversion paths can reduce payload size in PyG objects.

## Stream for Incremental Workloads

- Use append-only streams for pure append pipelines.
- Use mutable streams only when update/remove semantics are required.

## Profile and Benchmark

Use:

- `scripts/profile_encoding.py` for profiling paths
- `scripts/benchmark_encoder_suite.py` for comparative encoder and batch-size benchmarks
