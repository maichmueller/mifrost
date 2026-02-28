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

- `HorizonEncoder` takes a root plus `TransitionDAG`. In batch mode, `dags` may be passed as:
  - one shared DAG,
  - an aligned iterable of `TransitionDAG | None`,
  - `BatchParam.shared(dag)`, or
  - `BatchParam.separate([dag0, None, ...])`
- Transition encoders require successor inputs (`successor` or `successors`). In batch mode,
  `successors` may be passed as:
  - one shared successor state,
  - an aligned iterable of successor states,
  - `BatchParam.shared(successor)`, or
  - `BatchParam.separate([succ0, succ1, ...])`
- `HorizonEncoder` and transition encoders currently expose `actions` / `history_subgoals`
  parameters for API consistency, but non-empty payloads are rejected by their current encoding
  implementations.
