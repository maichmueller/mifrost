from __future__ import annotations

from typing import Any

import mifrost
import pytest
from mifrost.backends.flat import FlatSemanticAdapter
from mifrost.backends.pymimir import PymimirSnapshotReader

from .test_semantic_hgraph_encoder import _assert_hgraph_parity
from .test_utils import adv_action, adv_domain, adv_state, goal_inputs_from_problem


def _first_distinct_transition(space: Any, root: Any) -> tuple[Any, Any]:
    root_name = str(adv_state(root))
    for action, target in space.get_forward_transitions(root):
        if (
            action is not None
            and target is not None
            and str(adv_state(target)) != root_name
        ):
            return action, target
    pytest.skip("fixture has no distinct successor")
    raise AssertionError("pytest.skip must not return")


def _semantic_action(adapter: FlatSemanticAdapter, action: Any) -> Any:
    action_index, arguments = adapter._action(adapter.reader.action_key(action))
    return mifrost._neutral_core.SemanticGroundAction(action_index, arguments)


@pytest.mark.parametrize("mode_name", ["full", "delta", "action"])
@pytest.mark.parametrize("root_policy_name", ["include", "encode_only", "exclude"])
@pytest.mark.parametrize("feature_profile", ["default", "rich"])
def test_semantic_horizon_matches_native_modes_and_root_policies(
    small_blocks: tuple[Any, Any, Any],
    mode_name: str,
    root_policy_name: str,
    feature_profile: str,
) -> None:
    space, domain, problem = small_blocks
    root = problem.get_initial_state()
    action, successor = _first_distinct_transition(space, root)

    native_dag = mifrost.TransitionDAG(adv_state(root))
    native_dag.register_transition(
        adv_state(root),
        adv_state(successor),
        adv_action(action),
    )
    native_config = mifrost.HorizonEncoderConfig()
    native_config.goal_derivations = {mifrost.GoalDerivation.plain}
    native_config.transition_mode = getattr(mifrost.HorizonEncoderMode, mode_name)
    native_config.root_policy = getattr(mifrost.RootPolicy, root_policy_name)
    if mode_name == "action":
        native_config.ignore_actions = False
    if feature_profile == "rich":
        native_config.include_static = False
        native_config.add_nullary_predicates = True
        native_config.include_empty_edge_types = False
        native_config.support_literals = True
        native_config.include_lgan_edges = True
    native_engine = mifrost.HorizonHGraphEncoderEngine(
        adv_domain(domain), native_config
    )
    native = native_engine.encode(
        adv_state(root), native_dag, goal_inputs_from_problem(problem)
    ).as_pyg()

    adapter = FlatSemanticAdapter(PymimirSnapshotReader(problem))
    core = mifrost._neutral_core
    semantic_dag = core.SemanticTransitionDAG(
        adapter.engine.predicates,
        adapter.engine.actions,
        [
            core.SemanticTransitionNode(
                adapter.make_input(root),
                0,
                0,
                display_name=str(adv_state(root)),
            ),
            core.SemanticTransitionNode(
                adapter.make_input(successor),
                1,
                1,
                incoming_action=_semantic_action(adapter, action),
                display_name=str(adv_state(successor)),
            ),
        ],
        [(0, 1)],
    )
    semantic_config = core.SemanticHorizonHGraphEncoderConfig()
    semantic_config.goal_derivations = {core.GoalDerivation.plain}
    semantic_config.transition_mode = getattr(
        core.SemanticHorizonEncoderMode, mode_name
    )
    semantic_config.root_policy = getattr(core.RootPolicy, root_policy_name)
    if mode_name == "action":
        semantic_config.ignore_actions = False
    if feature_profile == "rich":
        semantic_config.include_static = False
        semantic_config.add_nullary_predicates = True
        semantic_config.include_empty_edge_types = False
        semantic_config.support_literals = True
        semantic_config.include_lgan_edges = True
    semantic_engine = core.SemanticHorizonHGraphEncoderEngine(
        adapter.engine.predicates,
        adapter.engine.actions,
        semantic_config,
    )
    semantic = semantic_engine.encode(semantic_dag).as_pyg()

    _assert_hgraph_parity(
        native,
        semantic,
        native_relation_arities=dict(native_engine.relation_dict.items()),
        semantic_relation_arities=dict(semantic_engine.relation_arities),
    )
    assert semantic.target_indices.tolist() == native.target_indices.tolist()
    assert semantic.target_depths.tolist() == native.target_depths.tolist()
    assert (
        semantic.target_candidate_ids.tolist() == native.target_candidate_ids.tolist()
    )


def test_semantic_horizon_corrects_legacy_cross_category_satisfaction_collision(
    small_blocks: tuple[Any, Any, Any],
) -> None:
    space, domain, problem = small_blocks
    root = problem.get_initial_state()
    action, successor = _first_distinct_transition(space, root)
    reader = PymimirSnapshotReader(problem)
    adapter = FlatSemanticAdapter(reader)
    core = mifrost._neutral_core

    actual_facts = {
        (fact.predicate, tuple(fact.arguments))
        for fact in adapter.make_input(root).state_facts
    }
    semantic_goals = list(adapter.make_input(root).goals)
    if not semantic_goals or any(
        (goal.atom.predicate, tuple(goal.atom.arguments)) in actual_facts
        for goal in semantic_goals
    ):
        pytest.skip("fixture does not expose an actually-unsatisfied root goal")

    native_dag = mifrost.TransitionDAG(adv_state(root))
    native_dag.register_transition(
        adv_state(root), adv_state(successor), adv_action(action)
    )
    derivations = {
        mifrost.GoalDerivation.plain,
        mifrost.GoalDerivation.satisfied,
        mifrost.GoalDerivation.unsatisfied,
    }
    native_config = mifrost.HorizonEncoderConfig()
    native_config.goal_derivations = derivations
    native = (
        mifrost.HorizonHGraphEncoderEngine(adv_domain(domain), native_config)
        .encode(adv_state(root), native_dag, goal_inputs_from_problem(problem))
        .as_pyg()
    )
    native_root_satisfied = sum(
        str(name).startswith("_root|") and "[g][sat]" in str(name)
        for node_type in native.node_types
        for name in native[node_type].node_names
    )
    if native_root_satisfied == 0:
        pytest.skip("fixture does not expose the legacy repository-index collision")

    semantic_dag = core.SemanticTransitionDAG(
        adapter.engine.predicates,
        adapter.engine.actions,
        [
            core.SemanticTransitionNode(
                adapter.make_input(root), 0, 0, display_name=str(adv_state(root))
            ),
            core.SemanticTransitionNode(
                adapter.make_input(successor),
                1,
                1,
                incoming_action=_semantic_action(adapter, action),
                display_name=str(adv_state(successor)),
            ),
        ],
        [(0, 1)],
    )
    semantic_config = core.SemanticHorizonHGraphEncoderConfig()
    semantic_config.goal_derivations = derivations
    semantic = (
        core.SemanticHorizonHGraphEncoderEngine(
            adapter.engine.predicates, adapter.engine.actions, semantic_config
        )
        .encode(semantic_dag)
        .as_pyg()
    )
    semantic_root_satisfied = sum(
        str(name).startswith("_root|") and "[g][sat]" in str(name)
        for node_type in semantic.node_types
        for name in semantic[node_type].node_names
    )
    semantic_root_unsatisfied = sum(
        str(name).startswith("_root|") and "[g][unsat]" in str(name)
        for node_type in semantic.node_types
        for name in semantic[node_type].node_names
    )

    assert native_root_satisfied > 0
    assert semantic_root_satisfied == 0
    assert semantic_root_unsatisfied == len(semantic_goals)


def _empty_input(objects: list[str] | None = None) -> Any:
    return mifrost._neutral_core.SemanticFlatRelationInput.from_compact(
        objects=objects or [],
        state_facts=[],
        goals=[],
        actions=[],
        subgoal_layers=[],
        history=[],
    )


def _synthetic_dag(
    edges: list[tuple[int, int]],
    *,
    objects: list[str] | None = None,
    candidate_ids: dict[int, int] | None = None,
    missing_display: set[int] | None = None,
    incoming_actions: dict[int, Any] | None = None,
) -> Any:
    core = mifrost._neutral_core
    node_count = 1 + max((max(edge) for edge in edges), default=0)
    depths = [0] * node_count
    for parent, child in edges:
        depths[child] = max(depths[child], depths[parent] + 1)
    nodes = [
        core.SemanticTransitionNode(
            _empty_input(objects),
            index,
            depths[index],
            incoming_action=(incoming_actions or {}).get(index),
            candidate_id=(candidate_ids or {}).get(index),
            display_name=(
                None if index in (missing_display or set()) else f"state-{index}"
            ),
        )
        for index in range(node_count)
    ]
    return core.SemanticTransitionDAG([], [], nodes, edges)


def test_semantic_horizon_topology_and_empty_schema() -> None:
    core = mifrost._neutral_core
    dag = _synthetic_dag([(0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (2, 6)])
    config = core.SemanticHorizonHGraphEncoderConfig()
    config.root_policy = core.RootPolicy.include
    config.enable_parent_relation = True
    config.enable_sibling_relation = True
    config.enable_cousin_relation = True
    engine = core.SemanticHorizonHGraphEncoderEngine([], [], config)
    data = engine.encode(dag).as_pyg()

    assert engine.relation_arities == {
        "_parent_": 2,
        "_sibling_": 2,
        "_cousin_": 2,
    }
    assert data["_parent_"].num_nodes == 6
    assert data["_sibling_"].num_nodes == 6
    assert data["_cousin_"].num_nodes == 8
    assert data.target_indices.tolist() == list(range(7))
    assert data.target_depths.tolist() == [0, 1, 1, 2, 2, 2, 2]


def test_semantic_horizon_zero_arity_action_and_symbol_name_collision() -> None:
    core = mifrost._neutral_core
    action = core.SemanticGroundAction(0, [])
    inputs = [_empty_input(["target:0"]), _empty_input(["target:0"])]
    nodes = [
        core.SemanticTransitionNode(inputs[0], 0, 0, display_name="root"),
        core.SemanticTransitionNode(
            inputs[1], 1, 1, incoming_action=action, display_name="successor"
        ),
    ]
    dag = core.SemanticTransitionDAG(
        [], [core.SemanticActionSpec("finish", 0)], nodes, [(0, 1)]
    )
    config = core.SemanticHorizonHGraphEncoderConfig()
    config.transition_mode = core.SemanticHorizonEncoderMode.action
    config.ignore_actions = False
    engine = core.SemanticHorizonHGraphEncoderEngine(
        [], [core.SemanticActionSpec("finish", 0)], config
    )
    data = engine.encode(dag).as_pyg()

    assert engine.relation_arities["finish"] == 1
    assert list(data["finish"].node_names) == ["target:1|(finish )"]
    assert list(data.object_names) == ["target:0"]
    assert list(data.target_names) == ["successor"]


def test_semantic_horizon_candidate_rows_names_batch_and_owned_builder() -> None:
    core = mifrost._neutral_core
    dag = _synthetic_dag([(0, 1)], candidate_ids={1: 42})
    engine = core.SemanticHorizonHGraphEncoderEngine([], [])
    encoding = engine.encode(dag)
    data = encoding.as_pyg()
    assert data.target_indices.tolist() == [1]
    assert data.target_candidate_ids.tolist() == [42]
    assert data.target_depths.tolist() == [1]
    assert list(data.target_names) == ["state-1"]

    batch = engine.encode_batch([dag, dag])
    assert batch.num_graphs == 2
    builder = core.BatchBuilder()
    engine.encode(dag, builder)
    builder.next_graph()
    engine.encode(dag, builder)
    builder.next_graph()
    composed = builder.build()
    assert composed.num_graphs == 2
    assert composed.as_pyg(as_batch=True).target_indices.tolist() == (
        batch.as_pyg(as_batch=True).target_indices.tolist()
    )


def test_semantic_horizon_policy_and_schema_errors() -> None:
    core = mifrost._neutral_core
    action_config = core.SemanticHorizonHGraphEncoderConfig()
    action_config.transition_mode = core.SemanticHorizonEncoderMode.action
    action_engine = core.SemanticHorizonHGraphEncoderEngine([], [], action_config)
    with pytest.raises(ValueError, match="requires ignore_actions=false"):
        action_engine.encode(_synthetic_dag([(0, 1)]))

    with pytest.raises(ValueError, match="display_name"):
        core.SemanticHorizonHGraphEncoderEngine([], []).encode(
            _synthetic_dag([(0, 1)], missing_display={1})
        )

    partial_ids = _synthetic_dag([(0, 1), (0, 2)], candidate_ids={1: 7})
    with pytest.raises(ValueError, match="missing candidate_id"):
        core.SemanticHorizonHGraphEncoderEngine([], []).encode(partial_ids)

    predicate = core.SemanticPredicateSpec(
        core.SemanticPredicateCategory.fluent, "p", 0
    )
    with pytest.raises(ValueError, match="schema"):
        core.SemanticHorizonHGraphEncoderEngine([predicate], []).encode(
            _synthetic_dag([(0, 1)])
        )


def test_semantic_horizon_export_names_false_allows_missing_display_names() -> None:
    core = mifrost._neutral_core
    config = core.SemanticHorizonHGraphEncoderConfig()
    config.export_node_names = False
    data = (
        core.SemanticHorizonHGraphEncoderEngine([], [], config)
        .encode(_synthetic_dag([(0, 1)], missing_display={0, 1}))
        .as_pyg()
    )
    assert not hasattr(data, "target_names")
    assert data.target_indices.tolist() == [1]


def test_semantic_horizon_supplied_delta_static_nullary_goals_and_subgoals() -> None:
    core = mifrost._neutral_core
    predicates = [
        core.SemanticPredicateSpec(core.SemanticPredicateCategory.static, "flag", 0),
        core.SemanticPredicateSpec(core.SemanticPredicateCategory.fluent, "p", 1),
    ]
    root = core.SemanticFlatRelationInput.from_compact(
        objects=["a"],
        state_facts=[(0, []), (1, [0])],
        goals=[(1, [0], True)],
        actions=[],
        subgoal_layers=[[(1, [0], True)]],
        history=[],
    )
    successor = core.SemanticFlatRelationInput.from_compact(
        objects=["a"],
        state_facts=[(0, [])],
        goals=[(1, [0], True)],
        actions=[],
        subgoal_layers=[[(1, [0], True)]],
        history=[],
    )
    removed_p = core.SemanticLiteral(core.SemanticAtom(1, [0]), False)
    removed_flag = core.SemanticLiteral(core.SemanticAtom(0, []), False)
    dag = core.SemanticTransitionDAG(
        predicates,
        [],
        [
            core.SemanticTransitionNode(root, 0, 0, display_name="root"),
            core.SemanticTransitionNode(
                successor,
                1,
                1,
                delta_literals=[removed_p, removed_flag],
                display_name="successor",
            ),
        ],
        [(0, 1)],
    )
    config = core.SemanticHorizonHGraphEncoderConfig()
    config.transition_mode = core.SemanticHorizonEncoderMode.delta
    config.root_policy = core.RootPolicy.include
    config.add_nullary_predicates = True
    config.max_goal_level = 1
    config.goal_derivations = {
        core.GoalDerivation.plain,
        core.GoalDerivation.satisfied,
        core.GoalDerivation.unsatisfied,
        core.GoalDerivation.added_satisfied,
        core.GoalDerivation.added_unsatisfied,
    }
    engine = core.SemanticHorizonHGraphEncoderEngine(predicates, [], config)
    data = engine.encode(dag).as_pyg()

    assert engine.config.support_literals
    assert "![nullary_symbol]!" in data.object_names
    assert data["flag"].num_nodes == 1
    assert data["[-]flag"].num_nodes == 1
    assert data["[-]p"].num_nodes == 1
    assert any("[g][sat]" in node_type for node_type in data.node_types)
    assert any("[sg][sat]" in node_type for node_type in data.node_types)
    assert any("[g][sat-]" in node_type for node_type in data.node_types)


def test_semantic_horizon_derived_delta_and_explicit_empty_override() -> None:
    core = mifrost._neutral_core
    predicates = [
        core.SemanticPredicateSpec(core.SemanticPredicateCategory.fluent, "p", 1)
    ]
    root = core.SemanticFlatRelationInput.from_compact(
        objects=["a"],
        state_facts=[(0, [0])],
        goals=[],
        actions=[],
        subgoal_layers=[],
        history=[],
    )
    successor = _empty_input(["a"])
    config = core.SemanticHorizonHGraphEncoderConfig()
    config.transition_mode = core.SemanticHorizonEncoderMode.delta
    engine = core.SemanticHorizonHGraphEncoderEngine(predicates, [], config)

    fallback = core.SemanticTransitionDAG(
        predicates,
        [],
        [
            core.SemanticTransitionNode(root, 0, 0, display_name="root"),
            core.SemanticTransitionNode(successor, 1, 1, display_name="successor"),
        ],
        [(0, 1)],
    )
    explicit_empty = core.SemanticTransitionDAG(
        predicates,
        [],
        [
            core.SemanticTransitionNode(root, 0, 0, display_name="root"),
            core.SemanticTransitionNode(
                successor, 1, 1, delta_literals=[], display_name="successor"
            ),
        ],
        [(0, 1)],
    )

    assert engine.encode(fallback).as_pyg()["[-]p"].num_nodes == 1
    assert engine.encode(explicit_empty).as_pyg()["[-]p"].num_nodes == 0
