from __future__ import annotations

from collections import deque

import networkx as nx
import networkx.algorithms.isomorphism as iso
import pytest

import mifrost

from .test_utils import (
    adv_action,
    adv_domain,
    adv_state,
    goal_inputs_from_problem,
    parts_to_pyg,
    to_named_networkx,
)


def _build_dag(
    space,
    root,
    *,
    max_depth: int = 2,
    branch_factor: int = 2,
    with_actions: bool = False,
):
    dag = mifrost.TransitionDAG(adv_state(root))
    queue: deque[tuple[object, int]] = deque([(root, 0)])
    seen = {root}
    while queue:
        state, depth = queue.popleft()
        if depth >= max_depth:
            continue
        successors = list(space.get_forward_transitions(state))
        for action, target in successors[:branch_factor]:
            action = adv_action(action) if with_actions else None
            dag.register_transition(
                adv_state(state),
                adv_state(target),
                action,
            )
            if target not in seen:
                seen.add(target)
                queue.append((target, depth + 1))
    return dag


def _encode_graph(
    domain, root, dag, *, mode, config_override=None, goals=None, drop_lgan=False
):
    config = mifrost.HorizonEncoderConfig()
    config.transition_mode = mode
    if config_override:
        for key, value in config_override.items():
            setattr(config, key, value)
    encoder = mifrost.HorizonHGraphEncoderEngine(adv_domain(domain), config)
    parts = encoder.encode(adv_state(root), dag, goals)
    return config, to_named_networkx(parts_to_pyg(parts), drop_lgan=drop_lgan)


def _target_nodes(graph, config):
    prefix = config.target_symbol_prefix
    nodes = {}
    for node, data in graph.nodes(data=True):
        if data.get("type") != config.symbol_type_id:
            continue
        if not str(node).startswith(prefix):
            continue
        idx = int(str(node)[len(prefix) :])
        nodes[idx] = (node, data)
    return nodes


def test_transition_tree_encoder_full_semantics(small_blocks):
    space, domain, problem = small_blocks
    root = problem.get_initial_state()
    dag = _build_dag(space, root)
    goals = goal_inputs_from_problem(problem)

    config, graph = _encode_graph(
        domain,
        root,
        dag,
        mode=mifrost.HorizonEncoderMode.Full,
        config_override={"enable_parent_relation": True},
        goals=goals,
    )

    target_nodes = _target_nodes(graph, config)
    expected_indices = {node.index for node in dag.nodes()}
    assert set(target_nodes.keys()) == expected_indices

    expected_transition_nodes = {
        f"{config.parent_relation}({parent}->{child})"
        for parent, child in dag.transitions()
    }
    actual_transition_nodes = {
        node
        for node, data in graph.nodes(data=True)
        if data["type"] == config.parent_relation
    }
    assert actual_transition_nodes == expected_transition_nodes

    for parent_idx, child_idx in dag.transitions():
        transition_node = f"{config.parent_relation}({parent_idx}->{child_idx})"
        parent_node = target_nodes[parent_idx][0]
        child_node = target_nodes[child_idx][0]
        assert graph.has_edge(parent_node, transition_node)
        assert graph.has_edge(child_node, transition_node)


def test_transition_tree_encoder_delta_changes(small_blocks):
    space, domain, problem = small_blocks
    root = problem.get_initial_state()
    dag = _build_dag(space, root)
    goals = goal_inputs_from_problem(problem)

    config, graph = _encode_graph(
        domain,
        root,
        dag,
        mode=mifrost.HorizonEncoderMode.Delta,
        goals=goals,
    )
    target_nodes = _target_nodes(graph, config)
    root_node = target_nodes[0][0]

    # root should contain all facts (delta mode retains root facts)
    neighbors = {
        neighbor
        for neighbor in graph.neighbors(root_node)
        if graph.nodes[neighbor]["type"] != config.symbol_type_id
    }
    assert neighbors


def test_transition_tree_encoder_custom_prefix(small_blocks):
    space, domain, problem = small_blocks
    root = problem.get_initial_state()
    dag = _build_dag(space, root)
    goals = goal_inputs_from_problem(problem)

    config, graph = _encode_graph(
        domain,
        root,
        dag,
        mode=mifrost.HorizonEncoderMode.Full,
        config_override={
            "target_symbol_prefix": "S@",
            "enable_parent_relation": True,
            "parent_relation": "custom_transition",
        },
        goals=goals,
    )

    target_nodes = [
        node
        for node, data in graph.nodes(data=True)
        if data["type"] == config.symbol_type_id and str(node).startswith("S@")
    ]
    assert target_nodes

    transition_nodes = [
        node
        for node, data in graph.nodes(data=True)
        if data["type"] == config.parent_relation
    ]
    assert transition_nodes
    for node in transition_nodes:
        positions = {
            edge_data["position"]
            for edge_dict in graph[node].values()
            for edge_data in edge_dict.values()
        }
        assert positions == {0, 1}


def test_transition_tree_encoder_roundtrip(small_blocks):
    space, domain, problem = small_blocks
    root = problem.get_initial_state()
    dag = _build_dag(space, root)
    goals = goal_inputs_from_problem(problem)

    _config, graph = _encode_graph(
        domain,
        root,
        dag,
        mode=mifrost.HorizonEncoderMode.Delta,
        goals=goals,
    )
    reconstructed = graph

    node_match = iso.categorical_node_match(["type"], [None])
    edge_match = iso.categorical_edge_match(["position"], [None])
    assert nx.is_isomorphic(
        graph, reconstructed, node_match=node_match, edge_match=edge_match
    )


def test_horizon_encoder_connects_actions_from_transitions_to_target_node(small_blocks):
    space, domain, problem = small_blocks
    root = problem.get_initial_state()
    action, target = next(space.get_forward_transitions(root))
    dag = mifrost.TransitionDAG(adv_state(root))
    dag.register_transition(
        adv_state(root),
        adv_state(target),
        adv_action(action),
    )
    goals = goal_inputs_from_problem(problem)

    config, graph = _encode_graph(
        domain,
        root,
        dag,
        mode=mifrost.HorizonEncoderMode.Full,
        goals=goals,
    )

    target_nodes = _target_nodes(graph, config)
    successor_node = target_nodes[1][0]
    action_node = f"{successor_node}|" + mifrost.RelationFormatter.format_action(
        adv_action(action)
    )
    assert graph.has_node(action_node)
    assert graph.has_edge(successor_node, action_node)
    positions = {
        edge_data["position"]
        for edge_data in graph[successor_node][action_node].values()
    }
    assert 0 in positions


def test_horizon_encoder_encodes_sibling_relations(medium_blocks):
    space, domain, problem = medium_blocks
    root = problem.get_initial_state()
    successors = list(space.get_forward_transitions(root))
    if len(successors) < 2:
        pytest.skip("Fixture does not provide two sibling successors.")
    child_a = successors[0][1]
    child_b = successors[1][1]

    dag = mifrost.TransitionDAG(adv_state(root))
    dag.register_transition(adv_state(root), adv_state(child_a))
    dag.register_transition(adv_state(root), adv_state(child_b))
    goals = goal_inputs_from_problem(problem)

    config, graph = _encode_graph(
        domain,
        root,
        dag,
        mode=mifrost.HorizonEncoderMode.Full,
        config_override={
            "enable_parent_relation": True,
            "enable_sibling_relation": True,
        },
        goals=goals,
    )

    target_nodes = _target_nodes(graph, config)
    idx_a = dag.index(adv_state(child_a))
    idx_b = dag.index(adv_state(child_b))
    sib_nodes_for_pair = []
    for node, data in graph.nodes(data=True):
        if data["type"] != config.sibling_relation:
            continue
        neighbors = set(graph.neighbors(node))
        if {target_nodes[idx_a][0], target_nodes[idx_b][0]} <= neighbors:
            sib_nodes_for_pair.append(node)
    assert len(sib_nodes_for_pair) >= 2


def test_horizon_encoder_encodes_cousin_relations(medium_blocks):
    space, domain, problem = medium_blocks
    root = problem.get_initial_state()
    level1 = list(space.get_forward_transitions(root))
    if len(level1) < 2:
        pytest.skip("Fixture does not provide two first-level branches.")
    p1 = level1[0][1]
    p2 = level1[1][1]
    level2a = list(space.get_forward_transitions(p1))
    level2b = list(space.get_forward_transitions(p2))
    if not level2a or not level2b:
        pytest.skip("Fixture does not provide grandchildren for cousin test.")
    u = level2a[0][1]
    v = level2b[0][1]

    dag = mifrost.TransitionDAG(adv_state(root))
    dag.register_transition(adv_state(root), adv_state(p1))
    dag.register_transition(adv_state(root), adv_state(p2))
    dag.register_transition(adv_state(p1), adv_state(u))
    dag.register_transition(adv_state(p2), adv_state(v))
    goals = goal_inputs_from_problem(problem)

    config, graph = _encode_graph(
        domain,
        root,
        dag,
        mode=mifrost.HorizonEncoderMode.Full,
        config_override={
            "enable_parent_relation": True,
            "enable_cousin_relation": True,
            "enable_sibling_relation": True,
        },
        goals=goals,
    )

    target_nodes = _target_nodes(graph, config)
    idx_u = dag.index(adv_state(u))
    idx_v = dag.index(adv_state(v))
    name_u = target_nodes[idx_u][0]
    name_v = target_nodes[idx_v][0]

    def _dir_exists(src_name: str, dst_name: str) -> bool:
        for node, data in graph.nodes(data=True):
            if data["type"] != config.cousin_relation:
                continue
            nbs = set(graph.neighbors(node))
            if {src_name, dst_name} <= nbs:
                src_pos = next(iter(graph[node][src_name].values())).get("position")
                dst_pos = next(iter(graph[node][dst_name].values())).get("position")
                if src_pos == 0 and dst_pos == 1:
                    return True
        return False

    assert _dir_exists(name_u, name_v)
    assert _dir_exists(name_v, name_u)
