from __future__ import annotations

import mifrost
import pytest

from .test_utils import (
    adv_action,
    adv_domain,
    adv_state,
    hetero_data_equal,
    goal_inputs_from_problem,
    encoding_dict_to_pyg,
)


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
    config.transition_mode = mifrost.HorizonEncoderMode.Full
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
