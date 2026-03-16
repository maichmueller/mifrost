from __future__ import annotations

import mifrost
import pymimir as mimir
import pytest

from .test_utils import (
    adv_action,
    adv_domain,
    adv_state,
    hetero_data_equal,
    goal_inputs_from_problem,
    encoding_dict_to_pyg,
)


def _first_distinct_changed_transitions(space, root, count: int = 2):
    out = []
    seen_targets: set[str] = set()
    root_repr = str(adv_state(root))
    for action, target in space.get_forward_transitions(root):
        if action is None or target is None:
            continue
        target_repr = str(adv_state(target))
        if target_repr == root_repr or target_repr in seen_targets:
            continue
        seen_targets.add(target_repr)
        out.append((action, target))
        if len(out) >= count:
            return out
    pytest.skip("Fixture should yield enough distinct changed transitions")


def _delta_literals(root, target, problem):
    root_atoms = set(root.get_atoms(ignore_static=True))
    target_atoms = set(target.get_atoms(ignore_static=True))
    out = [
        mimir.GroundLiteral.new(atom, True, problem)
        for atom in (target_atoms - root_atoms)
    ]
    out.extend(
        mimir.GroundLiteral.new(atom, False, problem)
        for atom in (root_atoms - target_atoms)
    )
    return out


def test_horizon_encoder_target_mapping_and_order(small_blocks):
    space, domain, problem = small_blocks
    root = problem.get_initial_state()

    transitions = list(space.get_forward_transitions(root))[:3]
    if not transitions:
        import pytest

        pytest.skip("Fixture should yield at least 1 transition")

    dag = mifrost.TransitionDAG(adv_state(root))
    for action, target in transitions:
        dag.register_transition(
            adv_state(root),
            adv_state(target),
            adv_action(action),
        )

    config = mifrost.HorizonEncoderConfig()
    config.transition_mode = mifrost.HorizonEncoderMode.full
    config.ignore_actions = False
    encoder = mifrost.HorizonHGraphEncoderEngine(adv_domain(domain), config)
    goals = goal_inputs_from_problem(problem)
    encoding_dict = encoder.encode(adv_state(root), dag, goals)
    data = encoding_dict_to_pyg(encoding_dict)

    symbol_type = config.symbol_type_id
    symbol_node_names = list(getattr(data[symbol_type], "node_names", []))
    prefix = config.target_symbol_prefix
    target_positions = [
        idx
        for idx, name in enumerate(symbol_node_names)
        if str(name).startswith(prefix)
    ]
    assert target_positions == list(range(len(target_positions))), (
        f"Target positions should be contiguous from 0..n-1, got {target_positions}"
    )

    target_nodes = {
        int(str(name)[len(prefix) :]): name
        for name in symbol_node_names
        if str(name).startswith(prefix)
    }

    # Verify per-transition action mapping (edge from target symbol -> action node at pos 0)
    formatter = mifrost.RelationFormatter
    for action, target in transitions:
        action = adv_action(action)
        action_type = formatter.format_action_schema(action.get_action())
        target_idx = dag.index(adv_state(target))
        if target_idx == 0:
            # Root actions are not encoded in the horizon graph.
            continue
        target_name = target_nodes[target_idx]
        action_node_name = f"{target_name}|{formatter.format_action(action)}"

        node_names = list(getattr(data[action_type], "node_names", []))
        assert action_node_name in node_names, (
            f"Expected action node '{action_node_name}' in node names for type '{action_type}'."
        )
        action_idx = node_names.index(action_node_name)
        target_symbol_idx = symbol_node_names.index(target_name)

        edge_type = (symbol_type, "0", action_type)
        edge_index = data[edge_type].edge_index
        src_indices = edge_index[0].tolist()
        dst_indices = edge_index[1].tolist()
        assert (target_symbol_idx, action_idx) in zip(src_indices, dst_indices), (
            f"No ({target_name} -> {action_node_name}) edge at position 0."
        )


def test_horizon_as_pyg_exposes_target_graph_attrs(small_blocks):
    space, domain, problem = small_blocks
    root = problem.get_initial_state()
    transitions = list(space.get_forward_transitions(root))[:2]
    if not transitions:
        pytest.skip("Fixture should yield at least 1 transition")

    dag = mifrost.TransitionDAG(adv_state(root))
    for action, target in transitions:
        dag.register_transition(
            adv_state(root),
            adv_state(target),
            adv_action(action),
        )

    encoder = mifrost.HorizonEncoder(domain)
    goals = list(problem.get_goal_condition().get_literals())
    data = encoder.encode(root, dag=dag, goals=goals).as_pyg(as_batch=True)

    assert hasattr(data, "target_names")
    assert hasattr(data, "target_symbol_prefix")
    assert hasattr(data, "target_candidate_ids")
    assert data.target_symbol_prefix == encoder.target_symbol_prefix
    assert len(list(data.target_names)) == len(data.target_indices.tolist())
    assert len(data.target_candidate_ids.tolist()) == len(data.target_indices.tolist())
    assert data.target_candidate_ids.tolist() == data.target_indices.tolist()


def test_horizon_export_node_names_false_skips_target_names(small_blocks):
    space, domain, problem = small_blocks
    root = problem.get_initial_state()
    transitions = list(space.get_forward_transitions(root))[:1]
    if not transitions:
        pytest.skip("Fixture should yield at least 1 transition")

    dag = mifrost.TransitionDAG(adv_state(root))
    for action, target in transitions:
        dag.register_transition(
            adv_state(root),
            adv_state(target),
            adv_action(action),
        )

    encoder = mifrost.HorizonEncoder(domain, export_node_names=False)
    goals = list(problem.get_goal_condition().get_literals())
    data = encoder.encode(root, dag=dag, goals=goals).as_pyg(as_batch=True)

    assert not hasattr(data, "target_names")


def test_horizon_batch_target_names_cover_all_candidates(small_blocks):
    space, domain, problem = small_blocks
    root = problem.get_initial_state()
    transitions = _first_distinct_changed_transitions(space, root, count=2)

    dag = mifrost.TransitionDAG(adv_state(root))
    for action, target in transitions[:2]:
        dag.register_transition(
            adv_state(root),
            adv_state(target),
            adv_action(action),
        )

    second_root = adv_state(transitions[0][1])
    empty_dag = mifrost.TransitionDAG(second_root)
    encoder = mifrost.HorizonEncoder(domain)
    data = encoder.encode_batch(
        [root, second_root],
        dags=[dag, empty_dag],
        goals=[
            list(problem.get_goal_condition().get_literals()),
            list(problem.get_goal_condition().get_literals()),
        ],
    ).as_pyg(as_batch=True)

    assert hasattr(data, "target_names")
    assert len(list(data.target_names)) == len(data.target_indices.tolist())


def test_horizon_target_candidate_ids_from_explicit_dag_ids(small_blocks):
    space, domain, problem = small_blocks
    root = problem.get_initial_state()
    (action0, target0), (action1, target1) = _first_distinct_changed_transitions(
        space, root, count=2
    )

    dag = mifrost.TransitionDAG(adv_state(root))
    dag.register_transition(
        adv_state(root),
        adv_state(target0),
        adv_action(action0),
        candidate_id=101,
    )
    dag.register_transition(
        adv_state(root),
        adv_state(target1),
        adv_action(action1),
        candidate_id=202,
    )

    data = (
        mifrost.HorizonEncoder(domain)
        .encode(root, dag=dag, goals=list(problem.get_goal_condition().get_literals()))
        .as_pyg(as_batch=True)
    )
    index_to_candidate = dict(
        zip(
            data.target_indices.tolist(),
            data.target_candidate_ids.tolist(),
            strict=True,
        )
    )
    assert index_to_candidate[dag.index(adv_state(target0))] == 101
    assert index_to_candidate[dag.index(adv_state(target1))] == 202


def test_horizon_delta_registers_state_literal_relations_without_plain_goals(
    small_blocks,
):
    _space, domain, _problem = small_blocks

    config = mifrost.HorizonEncoderConfig()
    config.transition_mode = mifrost.HorizonEncoderMode.delta
    config.root_policy = mifrost.RootPolicy.exclude
    config.ignore_actions = True
    config.goal_derivations = {
        mifrost.GoalDerivation.satisfied,
        mifrost.GoalDerivation.unsatisfied,
        mifrost.GoalDerivation.added_satisfied,
        mifrost.GoalDerivation.added_unsatisfied,
    }
    encoder = mifrost.HorizonHGraphEncoderEngine(adv_domain(domain), config)
    relation_names = {str(name) for name in encoder.relation_dict.keys()}
    assert any(name.startswith("[+]") for name in relation_names)
    assert not any("[state]" in name for name in relation_names)


def test_horizon_excluded_root_skips_root_parent_edges(small_blocks):
    space, domain, problem = small_blocks
    root = problem.get_initial_state()
    transitions = _first_distinct_changed_transitions(space, root, count=1)
    action, target = transitions[0]

    dag = mifrost.TransitionDAG(adv_state(root))
    dag.register_transition(
        adv_state(root),
        adv_state(target),
        adv_action(action),
        candidate_id=101,
    )

    config = mifrost.HorizonEncoderConfig()
    config.transition_mode = mifrost.HorizonEncoderMode.delta
    config.root_policy = mifrost.RootPolicy.exclude
    config.enable_parent_relation = True
    config.enable_sibling_relation = False
    config.enable_cousin_relation = False
    config.ignore_actions = True
    encoder = mifrost.HorizonHGraphEncoderEngine(adv_domain(domain), config)
    data = encoding_dict_to_pyg(
        encoder.encode(
            adv_state(root),
            dag,
            goal_inputs_from_problem(problem),
        )
    )

    parent_edge_type = ("_symbol_", "0", "_parent_")
    assert parent_edge_type in data.edge_types
    assert data[parent_edge_type].edge_index.numel() == 0


def test_horizon_rejects_partial_explicit_candidate_ids(small_blocks):
    space, domain, problem = small_blocks
    root = problem.get_initial_state()
    (action0, target0), (action1, target1) = _first_distinct_changed_transitions(
        space, root, count=2
    )

    dag = mifrost.TransitionDAG(adv_state(root))
    dag.register_transition(
        adv_state(root),
        adv_state(target0),
        adv_action(action0),
        candidate_id=7,
    )
    dag.register_transition(
        adv_state(root),
        adv_state(target1),
        adv_action(action1),
    )

    encoder = mifrost.HorizonEncoder(domain)
    goals = list(problem.get_goal_condition().get_literals())
    with pytest.raises(ValueError, match="missing candidate_id for target node index"):
        encoder.encode(root, dag=dag, goals=goals)


def test_horizon_rejects_duplicate_explicit_candidate_ids(small_blocks):
    space, domain, problem = small_blocks
    root = problem.get_initial_state()
    (action0, target0), (action1, target1) = _first_distinct_changed_transitions(
        space, root, count=2
    )

    dag = mifrost.TransitionDAG(adv_state(root))
    dag.register_transition(
        adv_state(root),
        adv_state(target0),
        adv_action(action0),
        candidate_id=9,
    )
    dag.register_transition(
        adv_state(root),
        adv_state(target1),
        adv_action(action1),
        candidate_id=9,
    )

    encoder = mifrost.HorizonEncoder(domain)
    goals = list(problem.get_goal_condition().get_literals())
    with pytest.raises(ValueError, match="duplicate candidate_id"):
        encoder.encode(root, dag=dag, goals=goals)


def test_horizon_to_networkx_preserves_object_symbol_nodes(small_blocks):
    space, domain, problem = small_blocks
    root = problem.get_initial_state()
    transitions = list(space.get_forward_transitions(root))[:2]
    if not transitions:
        import pytest

        pytest.skip("Fixture should yield at least 1 transition")

    dag = mifrost.TransitionDAG(adv_state(root))
    for action, target in transitions:
        dag.register_transition(
            adv_state(root),
            adv_state(target),
            adv_action(action),
        )

    encoder = mifrost.HorizonEncoder(domain)
    goals = list(problem.get_goal_condition().get_literals())
    data = encoder.encode_pyg(root, dag=dag, goals=goals)
    graph = encoder.to_networkx(data)

    object_names = [str(name) for name in getattr(data, "object_names", [])]
    symbol_nodes = {
        str(node)
        for node, attrs in graph.nodes(data=True)
        if attrs.get("type") == encoder.symbol_type_id
        and attrs.get("target_index") is None
    }
    for name in object_names:
        assert name in symbol_nodes, (
            f"Missing object symbol node in networkx graph: {name}"
        )


def test_horizon_encode_accepts_rustworkx_digraph(small_blocks):
    rx = pytest.importorskip("rustworkx")
    space, domain, problem = small_blocks
    root = problem.get_initial_state()
    transitions = [
        (action, target)
        for action, target in space.get_forward_transitions(root)
        if target is not None and target.get_index() != root.get_index()
    ][:1]
    if not transitions:
        pytest.skip("Fixture should yield at least 1 changed transition")

    action, target = transitions[0]
    dag = mifrost.TransitionDAG(adv_state(root))
    dag.register_transition(
        adv_state(root),
        adv_state(target),
        adv_action(action),
    )

    graph = rx.PyDiGraph()
    root_idx = graph.add_node(root)
    target_idx = graph.add_node(target)
    graph.add_edge(root_idx, target_idx, action)

    encoder = mifrost.HorizonEncoder(domain)
    goals = list(problem.get_goal_condition().get_literals())

    assert hetero_data_equal(
        encoder.encode(root, dag=graph, goals=goals),
        encoder.encode(root, dag=dag, goals=goals),
    )

    single_root_graph = rx.PyDiGraph()
    single_root_graph.add_node(root)
    assert hetero_data_equal(
        encoder.encode(root, dag=single_root_graph, goals=goals),
        encoder.encode(root, goals=goals),
    )


def test_horizon_delta_literals_match_state_diff_fallback(small_blocks):
    space, domain, problem = small_blocks
    root = problem.get_initial_state()
    (action, target) = _first_distinct_changed_transitions(space, root, count=1)[0]

    baseline_dag = mifrost.TransitionDAG(adv_state(root))
    baseline_dag.register_transition(
        adv_state(root),
        adv_state(target),
        adv_action(action),
        candidate_id=101,
    )

    annotated_dag = mifrost.TransitionDAG(adv_state(root))
    annotated_dag.register_transition(
        adv_state(root),
        adv_state(target),
        adv_action(action),
        candidate_id=101,
        delta_literals=_delta_literals(root, target, problem),
    )

    encoder = mifrost.HorizonEncoder(
        domain,
        transition_mode=mifrost.HorizonEncoderMode.delta,
        ignore_actions=False,
    )
    actual = encoder.encode(root, dag=annotated_dag).as_pyg(as_batch=True)
    expected = encoder.encode(root, dag=baseline_dag).as_pyg(as_batch=True)

    assert hetero_data_equal(actual, expected)
    assert (
        actual.target_candidate_ids.tolist() == expected.target_candidate_ids.tolist()
    )


def test_horizon_encode_rejects_mismatched_dag_roots(small_blocks):
    rx = pytest.importorskip("rustworkx")

    space, domain, problem = small_blocks
    root = problem.get_initial_state()
    transitions = [
        (_action, target)
        for _action, target in space.get_forward_transitions(root)
        if target is not None and target.get_index() != root.get_index()
    ]
    if not transitions:
        pytest.skip("Fixture should yield at least 1 changed successor")

    target = transitions[0][1]
    goals = list(problem.get_goal_condition().get_literals())
    encoder = mifrost.HorizonEncoder(domain)

    mismatched_dag = mifrost.TransitionDAG(adv_state(target))
    with pytest.raises(ValueError, match="dag root must match root state"):
        encoder.encode(root, dag=mismatched_dag, goals=goals)

    mismatched_graph = rx.PyDiGraph()
    mismatched_graph.add_node(target)
    with pytest.raises(ValueError, match="dag root must match root state"):
        encoder.encode(root, dag=mismatched_graph, goals=goals)


def test_horizon_encode_batch_rejects_actions_and_history(small_blocks):
    space, domain, problem = small_blocks
    root = problem.get_initial_state()
    transitions = list(space.get_forward_transitions(root))[:1]
    if not transitions:
        pytest.skip("Fixture should yield at least 1 transition")

    action, target = transitions[0]
    dag = mifrost.TransitionDAG(adv_state(root))
    dag.register_transition(
        adv_state(root),
        adv_state(target),
        adv_action(action),
    )

    encoder = mifrost.HorizonEncoder(domain)
    goals = list(problem.get_goal_condition().get_literals())

    with pytest.raises(
        ValueError,
        match="Horizon batch encoding does not support explicit action payloads",
    ):
        encoder.encode_batch(
            [root],
            dags=[dag],
            goals=[goals],
            actions=[action],
        )

    if goals:
        with pytest.raises(
            ValueError,
            match="Horizon batch encoding does not support history_subgoals payloads",
        ):
            encoder.encode_batch(
                [root],
                dags=[dag],
                goals=[goals],
                history_subgoals=[(-1, [goals[0]])],
                history_max_steps=3,
            )
