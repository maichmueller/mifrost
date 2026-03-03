from __future__ import annotations

from collections import deque

import mifrost

from .test_utils import (
    goal_inputs_from_problem,
    encoding_dict_to_pyg,
    to_named_networkx,
)


def _adv(obj):
    return getattr(obj, "_advanced_state", obj)


def _adv_domain(obj):
    return getattr(obj, "_advanced_domain", obj)


def _adv_action(obj):
    return getattr(obj, "_advanced_ground_action", obj)


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


def test_horizon_encoder_parent_relations(horizon_cases):
    space, domain, problem = horizon_cases
    root = problem.get_initial_state()
    dag = _build_dag(space, root)

    config = mifrost.HorizonEncoderConfig()
    config.transition_mode = mifrost.HorizonEncoderMode.Full
    config.enable_parent_relation = True
    encoder = mifrost.HorizonHGraphEncoderEngine(_adv_domain(domain), config)

    goals = goal_inputs_from_problem(problem)
    encoding_dict = encoder.encode(_adv(root), dag, goals)
    graph = to_named_networkx(encoding_dict_to_pyg(encoding_dict))

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


def test_horizon_encoder_exclude_root_candidate_controls_targets(horizon_cases):
    space, domain, problem = horizon_cases
    root = problem.get_initial_state()
    dag = _build_dag(space, root)
    if len(dag.nodes()) < 2:
        import pytest

        pytest.skip("Need at least one non-root node to test candidate filtering")

    goals = goal_inputs_from_problem(problem)

    excluded = mifrost.HorizonEncoderConfig()
    excluded.exclude_root_candidate = True
    encoder_excluded = mifrost.HorizonHGraphEncoderEngine(_adv_domain(domain), excluded)
    data_excluded = encoding_dict_to_pyg(
        encoder_excluded.encode(_adv(root), dag, goals)
    )
    indices_excluded = list(getattr(data_excluded, "target_indices", []))
    assert 0 not in indices_excluded
    assert len(indices_excluded) == len(dag.nodes()) - 1

    included = mifrost.HorizonEncoderConfig()
    included.exclude_root_candidate = False
    encoder_included = mifrost.HorizonHGraphEncoderEngine(_adv_domain(domain), included)
    data_included = encoding_dict_to_pyg(
        encoder_included.encode(_adv(root), dag, goals)
    )
    indices_included = list(getattr(data_included, "target_indices", []))
    assert 0 in indices_included
    assert len(indices_included) == len(dag.nodes())


def test_horizon_encoder_batch_collates_target_fields_with_ptrs(horizon_cases):
    space, domain, problem = horizon_cases
    root = problem.get_initial_state()
    dag = _build_dag(space, root)

    encoder = mifrost.HorizonEncoder(domain)
    goals = list(problem.get_goal_condition().get_literals())
    batch_enc = encoder.encode_batch([root, root], dags=[dag, dag], goals=goals)
    data = batch_enc.as_pyg(as_batch=True)

    target_indices = getattr(data, "target_indices", None)
    target_indices_ptr = getattr(data, "target_indices_ptr", None)
    target_positions = getattr(data, "target_positions", None)
    target_positions_ptr = getattr(data, "target_positions_ptr", None)

    assert target_indices is not None
    assert target_indices_ptr is not None
    assert target_positions is not None
    assert target_positions_ptr is not None

    ptr = target_indices_ptr.tolist()
    assert ptr[0] == 0
    assert len(ptr) == 3
    assert ptr[1] > ptr[0]
    assert ptr[2] > ptr[1]

    symbol_ptr = data[encoder.symbol_type_id].ptr.tolist()
    symbol_n0 = symbol_ptr[1] - symbol_ptr[0]

    pos_ptr = target_positions_ptr.tolist()
    batched_positions = target_positions.tolist()
    second_graph_positions = batched_positions[pos_ptr[1] : pos_ptr[2]]
    assert all(int(value) >= int(symbol_n0) for value in second_graph_positions)


def test_horizon_relation_dict_arities_match_emitted_positions(horizon_cases):
    space, domain, problem = horizon_cases
    root = problem.get_initial_state()
    transitions = list(space.get_forward_transitions(root))
    if not transitions:
        import pytest

        pytest.skip("Need at least one transition to validate horizon arities")

    action, target = transitions[0]
    dag = mifrost.TransitionDAG(_adv(root))
    dag.register_transition(_adv(root), _adv(target), _adv_action(action))

    config = mifrost.HorizonEncoderConfig()
    config.transition_mode = mifrost.HorizonEncoderMode.Full
    config.ignore_actions = False
    encoder = mifrost.HorizonHGraphEncoderEngine(_adv_domain(domain), config)

    goals = goal_inputs_from_problem(problem)
    data = encoding_dict_to_pyg(encoder.encode(_adv(root), dag, goals))
    arity_by_relation = encoder.relation_dict.arity
    symbol_type = config.symbol_type_id

    checked = 0
    for node_type in data.node_types:
        if node_type == symbol_type:
            continue
        observed_positions: set[int] = set()
        for src_type, pos, dst_type in data.edge_types:
            if src_type != symbol_type or dst_type != node_type:
                continue
            try:
                pos_idx = int(pos)
            except (TypeError, ValueError):
                continue
            edge_index = data[(src_type, pos, dst_type)].edge_index
            if edge_index.numel() == 0:
                continue
            observed_positions.add(pos_idx)

        if not observed_positions:
            continue
        assert node_type in arity_by_relation
        assert arity_by_relation[node_type] == max(observed_positions) + 1
        checked += 1

    assert checked > 0


def test_horizon_lgan_targets_and_rr_directions(horizon_cases):
    space, domain, problem = horizon_cases
    root = problem.get_initial_state()
    dag = _build_dag(space, root, max_depth=2, branch_factor=2)

    config = mifrost.HorizonEncoderConfig()
    config.transition_mode = mifrost.HorizonEncoderMode.Full
    config.include_lgan_edges = True
    config.ignore_actions = False
    config.enable_parent_relation = True
    encoder = mifrost.HorizonHGraphEncoderEngine(_adv_domain(domain), config)

    goals = goal_inputs_from_problem(problem)
    data = encoding_dict_to_pyg(encoder.encode(_adv(root), dag, goals))

    target_positions = set(getattr(data, "target_positions", []).tolist())
    assert target_positions

    symbol_type = config.symbol_type_id
    saw_tn_or_nn = False
    saw_rr = False
    for edge_type, edge_index in data.edge_index_dict.items():
        src_type, rel, dst_type = edge_type
        if rel in {config.lgan_tn_edge_pos, config.lgan_nn_edge_pos}:
            saw_tn_or_nn = True
            assert src_type != symbol_type
            assert dst_type == symbol_type
            assert set(edge_index[1].tolist()).issubset(target_positions)
        elif rel == config.lgan_rr_edge_pos:
            saw_rr = True
            assert src_type != symbol_type
            assert dst_type != symbol_type

    assert saw_tn_or_nn
    assert saw_rr
