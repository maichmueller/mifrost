# How to Choose an Encoder

## Scenario Guide

| Scenario | Recommended Encoder | Why |
| --- | --- | --- |
| Encode current planning state with goals/actions/history | `HGraphEncoder` | Flexible heterogeneous state graph with optional goal/action/history features |
| Encode current planning state as packed flat relations | `FlatRelationEncoder` | Compact packed relation tensors with target-entity metadata and flat inspectability |
| Encode root state plus lookahead DAG structure | `HorizonEncoder` | Adds candidate-target DAG semantics and horizon-specific attributes |
| Encode root state plus lookahead DAG structure as flat packed relations | `FlatHorizonEncoder` | Compact state-target carrier rows over a `TransitionDAG` without relation nodes |
| Learn full transition structure (state -> successor) | `TransitionHGraphEncoder` | Represents both source and successor graph structure |
| Learn full transition structure (state -> successor) in the flat lane | `FlatTransitionEncoder` | Reuses the flat horizon/state-target substrate for compact successor encoding |
| Learn transition effects/deltas only | `TransitionEffectsHGraphEncoder` | Focuses on change signals instead of full duplicated structure |
| Learn transition effects/deltas only in the flat lane | `FlatTransitionEffectsEncoder` | Emits only changed successor structure on the compact flat carrier |
| Build compact homogeneous baselines | `ColorEncoder` | Lower-dimensional homogeneous encoding for simple baselines |
| Use ILG topology/features from Python | `ILGEncoder` | Pure-Python implementation with ILG atom/action/object construction |

## Minimal Runnable Examples

--8<-- "_includes/snippets/hgraph_basic.md"

--8<-- "_includes/snippets/horizon_basic.md"

--8<-- "_includes/snippets/transition_full_vs_delta.md"

--8<-- "_includes/snippets/color_basic.md"

--8<-- "_includes/snippets/ilg_basic.md"

## Stream Support

- `HGraphEncoder`, `HorizonEncoder`, transition encoders, `ColorEncoder`, and the flat encoder family provide stream wrappers.
- `HGraphEncoder` also exposes a mutable stream with `update/remove`.
- `FlatRelationEncoder.stream()` is append-only.
- `FlatRelationEncoder.mutable_stream()` supports `update/remove`.
- `FlatHorizonEncoder.stream()` is mutable.
- `FlatTransitionEncoder.stream()` and `FlatTransitionEffectsEncoder.stream()` are mutable wrapper-level streams built on flat horizon streaming.

## Action Contract

- `HGraphEncoder` expects flat action inputs (single action list or per-state flat lists).
- `FlatRelationEncoder` also accepts flat action inputs and materializes grounded-action target entities on the packed carrier.
- Nested/tuple action payloads are rejected by design.
- For IW lookahead/macro-transition outputs, use `HorizonEncoder` with `TransitionDAG`
  or a `rustworkx.PyDiGraph`.
- `FlatHorizonEncoder` and `FlatTransition*Encoder` derive candidate/state structure from
  `dag` or `successor(s)` and currently reject explicit non-empty `actions` or
  `history_subgoals` payloads.

## Batch Contract

- `encode_batch(states, *, ...)` is kwargs-based (no tuple/sample batch payloads).
- Batch-capable kwargs accept either shared payloads or per-state sequences.
- For specialized encoder kwargs, the same shared/per-entry rule applies:
  - transition encoders: `successors` accepts a single shared successor, an aligned iterable,
    `BatchParam.shared(...)`, or `BatchParam.separate([...])`
  - horizon encoder: `dags` accepts a shared DAG, an aligned iterable of
    `TransitionDAG | rustworkx.PyDiGraph | None`,
    `BatchParam.shared(...)`, or `BatchParam.separate([...])`
- High-level encoder batch paths preprocess wrapper/adapter inputs in Python, then
  dispatch to strict advanced-only C++ batch parsing.
- Direct low-level `_core._parse_*` batch helpers stay advanced-only and reject
  adapter-backed objects.
- `mifrost.transition_dag_from_rustworkx(...)` is the explicit conversion helper when
  you want to pre-normalize a `PyDiGraph` yourself. The integration is Python-level;
  there is no raw/native graph handoff into the C++ core.
- `transition_dag_from_rustworkx(...)` reads per-node candidate identity from payload
  `candidate_id` (mapping key or attribute). If candidate IDs are only partially present,
  conversion fails by default; pass
  `fallback_missing_candidate_id_to_node_index=True` to fill missing values from
  rustworkx node indices.
- `PyDiGraph` interop assumes the graph is already a valid horizon DAG/tree. The
  converter imports it directly into `TransitionDAG` and treats malformed graph
  structure as an input error.
- Encoder support:
  - `HGraphEncoder`: per-state `goals`, `actions`, `subgoal_layers`, `history_subgoals`
  - `FlatRelationEncoder`: per-state `goals`, `actions`, `subgoal_layers`, `history_subgoals`
  - `ColorEncoder`: per-state `goals`, `subgoal_layers` (`actions` are accepted as inputs but non-empty payloads are rejected)
  - `ILGEncoder`: per-state `goals`, `actions`, `subgoal_layers`
  - `Transition*Encoder`: shared or per-entry `successors`, per-state `goals`, `subgoal_layers` (`actions`/history are explicit inputs but currently rejected when non-empty)
  - `HorizonEncoder`: shared or per-entry `dags`, per-state `goals`, `subgoal_layers` (`actions`/history are explicit inputs but currently rejected when non-empty)
  - `FlatHorizonEncoder`: shared or per-entry `dags`, per-state `goals`, `subgoal_layers` (`actions`/history are explicit inputs but currently rejected when non-empty)
  - `FlatTransition*Encoder`: shared or per-entry `successors`, per-state `goals`, `subgoal_layers` (`actions`/history are explicit inputs but currently rejected when non-empty)
