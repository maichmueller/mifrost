# Encode backend-neutral semantic flat inputs

Use `SemanticFlatRelationEncoderEngine` when planning data already has an owned
semantic representation and should not be tied to a Mimir domain, problem, or
state object. The engine implements the same flat relation, target, history,
predicate-virtual, LGAN, and batch contract as `FlatRelationEncoderEngine`, but
its input contains only strings and integer indices.

The engine is initialized once from predicate and action schema:

```python
import mifrost

category = mifrost.SemanticPredicateCategory
predicates = [
    mifrost.SemanticPredicateSpec(category.fluent, "at", 2),
]
actions = [
    mifrost.SemanticActionSpec("move", 3),
]
engine = mifrost.SemanticFlatRelationEncoderEngine(predicates, actions)
```

Each graph owns its object table. Predicate indices refer to the engine's
predicate vector, action indices refer to its action vector, and object indices
refer to that graph's object vector:

```python
value = mifrost.SemanticFlatRelationInput()
value.objects = ["robot", "room-a", "room-b"]
value.state_facts = [mifrost.SemanticAtom(0, [0, 1])]
value.goals = [
    mifrost.SemanticLiteral(mifrost.SemanticAtom(0, [0, 2]), True),
]
value.actions = [mifrost.SemanticGroundAction(0, [0, 1, 2])]

encoding = engine.encode(value)
tensors = encoding.as_pyg()
```

These indices are compact call-local references, not persistent semantic IDs.
For process, repository, replay, or checkpoint boundaries, serialize names and
reconstruct the local indices deterministically. Never serialize planning
repository indices or assume independently constructed repositories share
identity.

`encode_batch` accepts any number of compatible owned inputs, including an
empty list. It uses the same node offsets, LGAN relation-instance offsets,
target-local candidate IDs, and optional relation-major packing as the Mimir
engine:

```python
config = mifrost.FlatRelationEncoderConfig(
    pack_relation_args_relation_major=True,
)
engine = mifrost.SemanticFlatRelationEncoderEngine(predicates, actions, config)
batch = engine.encode_batch([value, value])
```

Input validation rejects duplicate schema/object names, invalid arities,
out-of-scope indices, non-negative history deltas, unsupported target sources,
and subgoal layers beyond `max_goal_level` before emitting a graph.
