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
- Batch paths are C++-parsed and native-only (adapter-backed objects are rejected).
- Encoder support:
  - `HGraphEncoder`: per-state `goals`, `actions`, `subgoal_layers`, `history_subgoals`
  - `ColorEncoder`: per-state `goals`, `subgoal_layers` (`actions` rejected)
  - `ILGEncoder`: per-state `goals`, `actions`, `subgoal_layers`
  - `Transition*Encoder`: aligned `successors`, per-state `goals`, `subgoal_layers` (`actions` rejected)
  - `HorizonEncoder`: per-state `dags`, `goals`, `subgoal_layers` (`actions`/history rejected)
