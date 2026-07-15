from __future__ import annotations

import pytest

import mifrost


def _schema():
    category = mifrost.SemanticPredicateCategory
    return (
        [
            mifrost.SemanticPredicateSpec(getattr(category, "static"), "object", 1),
            mifrost.SemanticPredicateSpec(category.fluent, "flag", 0),
            mifrost.SemanticPredicateSpec(category.fluent, "on", 2),
            mifrost.SemanticPredicateSpec(category.derived, "ready", 1),
        ],
        [mifrost.SemanticActionSpec("move", 2)],
    )


def _config(*, relation_major: bool = False):
    return mifrost.FlatRelationEncoderConfig(
        ignore_zero_arity_relations=False,
        use_predicate_virtual_nodes=True,
        include_lgan_edges=True,
        lgan_anchor_sources={mifrost.TargetSource.history},
        max_goal_level=1,
        target_sources={
            mifrost.TargetSource.actions,
            mifrost.TargetSource.goals,
            mifrost.TargetSource.subgoals,
            mifrost.TargetSource.history,
        },
        goal_derivations={
            mifrost.GoalDerivation.plain,
            mifrost.GoalDerivation.satisfied,
            mifrost.GoalDerivation.unsatisfied,
        },
        pack_relation_args_relation_major=relation_major,
    )


def _input():
    value = mifrost.SemanticFlatRelationInput()
    value.objects = ["a", "b"]
    value.state_facts = [
        mifrost.SemanticAtom(0, [0]),
        mifrost.SemanticAtom(0, [1]),
        mifrost.SemanticAtom(1, []),
        mifrost.SemanticAtom(2, [0, 1]),
        mifrost.SemanticAtom(3, [0]),
    ]
    value.goals = [
        mifrost.SemanticLiteral(mifrost.SemanticAtom(2, [1, 0]), True),
        mifrost.SemanticLiteral(mifrost.SemanticAtom(1, []), False),
    ]
    value.actions = [
        mifrost.SemanticGroundAction(0, [0, 1]),
        mifrost.SemanticGroundAction(0, [0, 1]),
    ]
    value.subgoal_layers = [
        [mifrost.SemanticLiteral(mifrost.SemanticAtom(3, [0]), True)]
    ]
    value.history = [
        mifrost.SemanticHistoryEntry(
            -1,
            [
                mifrost.SemanticLiteral(mifrost.SemanticAtom(2, [0, 1]), True),
                mifrost.SemanticLiteral(mifrost.SemanticAtom(2, [0, 1]), True),
            ],
        )
    ]
    return value


def test_semantic_engine_encodes_all_flat_lanes_without_planning_views() -> None:
    predicates, actions = _schema()
    engine = mifrost.SemanticFlatRelationEncoderEngine(predicates, actions, _config())

    native = engine.encode(_input())
    data = native.as_pyg()

    assert native.num_graphs == 1
    assert tuple(engine.relation_names) == tuple(data.relation_names)
    assert data.object_names == ["a", "b"]
    assert data.node_sizes.tolist() == [len(data.node_names)]
    assert data.object_indices.tolist() == [0, 1]
    assert data.history_entity_dt.tolist() == [-1]
    assert data.target_sizes.tolist() == [7]
    assert data.target_positions[3].item() == data.target_positions[4].item()
    assert data.target_names[3] == data.target_names[4]
    assert data.relation_counts.shape == (1, len(engine.relation_names))
    assert data.relation_instance_sizes.item() == data.relation_counts.sum().item()
    assert data.lgan_tn_sizes.item() == len(data.lgan_tn_relation_indices)
    assert data.lgan_nn_sizes.item() == len(data.lgan_nn_relation_indices)
    assert data.lgan_rr_sizes.item() == len(data.lgan_rr_src_relation_indices)


def test_semantic_engine_batches_and_packs_relation_major() -> None:
    predicates, actions = _schema()
    engine = mifrost.SemanticFlatRelationEncoderEngine(
        predicates, actions, _config(relation_major=True)
    )

    native = engine.encode_batch([_input(), _input()])
    data = native.as_pyg(as_batch=True)

    assert native.num_graphs == 2
    assert data.relation_args_layout == "relation_major"
    assert data.ptr.tolist() == [
        0,
        data.node_sizes[0].item(),
        data.node_sizes.sum().item(),
    ]
    assert data.batch.tolist() == [
        graph
        for graph, count in enumerate(data.node_sizes.tolist())
        for _ in range(count)
    ]
    first_relation_count = data.relation_instance_sizes[0].item()
    assert min(data.lgan_tn_relation_indices[data.lgan_tn_sizes[0] :]).item() >= (
        first_relation_count
    )


def test_semantic_engine_empty_batch_retains_schema_without_nodes() -> None:
    predicates, actions = _schema()
    engine = mifrost.SemanticFlatRelationEncoderEngine(predicates, actions, _config())

    native = engine.encode_batch([])
    data = native.as_pyg(as_batch=True)

    assert native.num_graphs == 0
    assert tuple(data.relation_names) == tuple(engine.relation_names)
    assert data.x is None
    assert data.batch is None
    assert getattr(data, "ptr", None) is None


def test_semantic_engine_rejects_out_of_scope_indices() -> None:
    predicates, actions = _schema()
    engine = mifrost.SemanticFlatRelationEncoderEngine(predicates, actions)
    value = mifrost.SemanticFlatRelationInput()
    value.objects = ["a"]
    value.state_facts = [mifrost.SemanticAtom(2, [0, 1])]

    with pytest.raises(ValueError, match="object index out of range"):
        engine.encode(value)


def test_semantic_input_accepts_unbounded_history() -> None:
    value = mifrost.SemanticFlatRelationInput()
    value.history_max_steps = 2

    value.history_max_steps = None

    assert value.history_max_steps is None
