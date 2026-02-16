# mifrost Documentation

`mifrost` is a high-performance graph encoding library for planning states and transitions.
It provides native `BatchEncoding` outputs and optional conversion to PyTorch Geometric (`Data`/`HeteroData`).

## Start Here

- New users: [Installation](tutorials/installation.md)
- First usable graph: [First Encoding](tutorials/first-encoding.md)
- Throughput workflows: [Batching and Streaming](tutorials/batching-and-streaming.md)

## Choose an Encoder Quickly

- Use `HGraphEncoder` for state-centric heterogeneous encodings (goals/actions/history optional).
- Use `HorizonEncoder` for root state + lookahead DAG candidate structure.
- Use `TransitionHGraphEncoder` for full current-to-successor transition structure.
- Use `TransitionEffectsHGraphEncoder` for delta/effects-focused transition features.
- Use `ColorEncoder` for compact homogeneous graph baselines.
- Use `ILGEncoder` for Python-native ILG topology/features.

See [How to Choose an Encoder](how-to/choose-an-encoder.md) for scenario-specific guidance.

## What Is Covered

- Native-first API and conversion model
- Native hetero/homo tensor facades (`as_hetero`, `as_homo`) for model-boundary use
- Single, batch, append-only stream, and mutable stream workflows
- Dynamic graph fields (`GraphFieldSpec`) and batching semantics
- Adapter registration for custom wrapper types
