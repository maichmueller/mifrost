from __future__ import annotations

from collections import deque

import mifrost

from .test_utils import goal_inputs_from_problem, parts_to_pyg, to_named_networkx


def _adv(obj):
    return getattr(obj, "_advanced_state", obj)


def _adv_domain(obj):
    return getattr(obj, "_advanced_domain", obj)


def _build_dag(space, root, *, max_depth: int = 2, branch_factor: int = 2):
    dag = mifrost.TransitionDAG(_adv(root))
    queue = deque([(root, 0)])
    seen = {root}
    while queue:
        state, depth = queue.popleft()
        if depth >= max_depth:
            continue
        successors = list(space.get_forward_transitions(state))
        for action, target in successors[:branch_factor]:
            child = target
            dag.register_transition(_adv(state), _adv(child))
            if child not in seen:
                seen.add(child)
                queue.append((child, depth + 1))
    return dag


def test_horizon_encoder_parent_relations(small_blocks):
    space, domain, problem = small_blocks
    root = problem.get_initial_state()
    dag = _build_dag(space, root)

    config = mifrost.HorizonEncoderConfig()
    config.transition_mode = mifrost.HorizonEncoderMode.Full
    config.enable_parent_relation = True
    encoder = mifrost.HorizonHGraphEncoderEngine(_adv_domain(domain), config)

    goals = goal_inputs_from_problem(problem)
    parts = encoder.encode(_adv(root), dag, goals)
    graph = to_named_networkx(parts_to_pyg(parts))

    prefix = config.target_symbol_prefix
    target_nodes = {
        n
        for n, data in graph.nodes(data=True)
        if data.get("type") == config.symbol_type_id and str(n).startswith(prefix)
    }
    expected_targets = {f"{prefix}{node.index}" for node in dag.nodes()}
    assert target_nodes == expected_targets

    parent_relation = config.parent_relation
    parent_nodes = {
        n for n, data in graph.nodes(data=True) if data.get("type") == parent_relation
    }
    expected_parents = {
        f"{parent_relation}({parent}->{child})" for parent, child in dag.transitions()
    }
    assert parent_nodes == expected_parents

    for parent, child in dag.transitions():
        rel_node = f"{parent_relation}({parent}->{child})"
        parent_node = f"{prefix}{parent}"
        child_node = f"{prefix}{child}"
        assert graph.has_edge(parent_node, rel_node)
        assert graph.has_edge(child_node, rel_node)
