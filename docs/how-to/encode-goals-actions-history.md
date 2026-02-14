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

- `HorizonEncoder` takes a root plus `TransitionDAG`; it does not consume action lists directly in its encoding path.
- Transition encoders require successor inputs (`successor` or `successors`).
