# How to Choose an Encoder

## Scenario Guide

| Scenario | Recommended Encoder | Why |
| --- | --- | --- |
| Encode current planning state with goals/actions/history | `HGraphEncoder` | Flexible heterogeneous state graph with optional goal/action/history features |
| Encode root state plus lookahead DAG structure | `HorizonEncoder` | Adds candidate-target DAG semantics and horizon-specific attributes |
| Learn full transition structure (state -> successor) | `TransitionHGraphEncoder` | Represents both source and successor graph structure |
| Learn transition effects/deltas only | `TransitionEffectsHGraphEncoder` | Focuses on change signals instead of full duplicated structure |
| Build compact homogeneous baselines | `ColorEncoder` | Lower-dimensional homogeneous encoding for simple baselines |
| Use ILG topology/features from Python | `ILGEncoder` | Pure-Python implementation with ILG atom/action/object construction |

## Minimal Runnable Examples

--8<-- "_includes/snippets/hgraph_basic.md"

--8<-- "_includes/snippets/horizon_basic.md"

--8<-- "_includes/snippets/transition_full_vs_delta.md"

--8<-- "_includes/snippets/color_basic.md"

--8<-- "_includes/snippets/ilg_basic.md"

## Stream Support

- `HGraphEncoder`, `HorizonEncoder`, transition encoders, and `ColorEncoder` provide stream wrappers.
- `HGraphEncoder` also exposes a mutable stream with `update/remove`.

## Action Contract

- `HGraphEncoder` expects flat action inputs (single action list or per-state flat lists).
- Nested/tuple action payloads are rejected by design.
- For IW lookahead/macro-transition outputs, use `HorizonEncoder` with `TransitionDAG`.

## Batch Contract

- `encode_batch(states, *, ...)` is kwargs-based (no tuple/sample batch payloads).
- Batch-capable kwargs accept either shared payloads or per-state sequences.
- For specialized encoder kwargs, the same shared/per-entry rule applies:
  - transition encoders: `successors` accepts a single shared successor, an aligned iterable,
    `BatchParam.shared(...)`, or `BatchParam.separate([...])`
  - horizon encoder: `dags` accepts a shared DAG, an aligned iterable of `TransitionDAG | None`,
    `BatchParam.shared(...)`, or `BatchParam.separate([...])`
- High-level encoder batch paths preprocess wrapper/adapter inputs in Python, then
  dispatch to strict advanced-only C++ batch parsing.
- Direct low-level `_core._parse_*` batch helpers stay advanced-only and reject
  adapter-backed objects.
- Encoder support:
  - `HGraphEncoder`: per-state `goals`, `actions`, `subgoal_layers`, `history_subgoals`
  - `ColorEncoder`: per-state `goals`, `subgoal_layers` (`actions` are accepted as inputs but non-empty payloads are rejected)
  - `ILGEncoder`: per-state `goals`, `actions`, `subgoal_layers`
  - `Transition*Encoder`: shared or per-entry `successors`, per-state `goals`, `subgoal_layers` (`actions`/history are explicit inputs but currently rejected when non-empty)
  - `HorizonEncoder`: shared or per-entry `dags`, per-state `goals`, `subgoal_layers` (`actions`/history are explicit inputs but currently rejected when non-empty)
