# Encode Goals, Actions, and History

## Goals and Subgoal Layers

For heterogeneous encoders, goals default to the state's problem goals if omitted. You can pass explicit goals and optional layered subgoals:

```python
enc = encoder.encode(
    state,
    goals=goal_literals,
    subgoal_layers=[layer1, layer2],
)
```

## Actions

Action inputs are supported in `HGraphEncoder` state-step workflows:

```python
enc = encoder.encode(state, goals=goals, actions=actions)
```

If you want Horizon-like target metadata for encoded action targets, enable
`export_action_targets` on `HGraphEncoder`:

```python
encoder = mifrost.HGraphEncoder(domain, ignore_actions=False, export_action_targets=True)
enc = encoder.encode_batch(states, actions=per_state_actions)
data = enc.as_pyg(as_batch=True)
```

When enabled, HGraph emits:
- `target_positions` (symbol-node positions with batch node-offset semantics)
- `target_indices` (input action positions `0..n-1` per graph)
- `target_names` and `target_symbol_prefix`

`HGraphEncoder` action inputs must be flat. Nested or tuple/macro action payloads are rejected.
If your planner emits lookahead or macro-transition structures (for example from IW), use
`HorizonEncoder` with a `TransitionDAG` instead of passing nested action payloads to `HGraphEncoder`.

## History Subgoals

`HGraphEncoder` also supports history-aware subgoals:

```python
enc = encoder.encode(
    state,
    goals=goals,
    history_subgoals=[(1, literals_t_minus_1), (2, literals_t_minus_2)],
    history_max_steps=4,
)
```

## Notes

- `HorizonEncoder` takes a root plus `TransitionDAG` (or `rustworkx.PyDiGraph`).
  Use `mifrost.transition_dag_from_rustworkx(...)` if you want an explicit conversion step.
  In batch mode, `dags` may be passed as:
  - one shared DAG,
  - an aligned iterable of `TransitionDAG | rustworkx.PyDiGraph | None`,
  - `BatchParam.shared(dag)`, or
  - `BatchParam.separate([dag0, None, ...])`
- `rustworkx` interop is Python-level only. There is no raw/native graph handoff:
  `PyDiGraph` inputs are converted into `TransitionDAG` before encoding.
- `PyDiGraph` inputs are treated as already-valid horizon DAGs. `mifrost` does not
  pre-verify the incoming graph shape beyond the minimal import constraints needed to
  build a `TransitionDAG`; malformed inputs are treated as input errors and may fail
  during import or downstream encoding.
- Horizon `target_indices` refer to `TransitionDAG` insertion-order node indices, not
  the original `rustworkx` node IDs.
- Transition encoders require successor inputs (`successor` or `successors`). In batch mode,
  `successors` may be passed as:
  - one shared successor state,
  - an aligned iterable of successor states,
  - `BatchParam.shared(successor)`, or
  - `BatchParam.separate([succ0, succ1, ...])`
- `HorizonEncoder` and transition encoders currently expose `actions` / `history_subgoals`
  parameters for API consistency, but non-empty payloads are rejected by their current encoding
  implementations.
