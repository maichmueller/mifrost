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
