from __future__ import annotations

import pytest

import mifrost
from mifrost.encoders import (
    HGraphEncoder,
    HGraphEncoderStream,
    HGraphMutableEncoderStream,
    HorizonEncoder,
    HorizonEncoderStream,
)

from .test_utils import adv_action, adv_state, hetero_data_equal, encoding_dict_to_pyg


def _first_successor(space, state):
    transitions = list(space.get_forward_transitions(state))
    for _action, target in transitions:
        if target is None:
            continue
        if target.get_index() != state.get_index():
            return target
    pytest.skip("No successor state available for stream tests.")


def _first_actions(space, state, count: int = 2):
    transitions = list(space.get_forward_transitions(state))
    actions = [act for act, _ in transitions if act is not None]
    if len(actions) < count:
        pytest.skip("Fixture does not provide enough actions for stream tests.")
    return actions[:count]


def _horizon_dags(space, root):
    transitions = list(space.get_forward_transitions(root))
    for action, target in transitions:
        if target is None:
            continue
        if target.get_index() == root.get_index():
            continue
        empty_dag = mifrost.TransitionDAG(adv_state(root))
        successor_dag = mifrost.TransitionDAG(adv_state(root))
        successor_dag.register_transition(
            adv_state(root), adv_state(target), adv_action(action)
        )
        return target, empty_dag, successor_dag
    pytest.skip("No successor transition available for horizon stream tests.")


def _horizon_pygraph(root, action, target):
    rx = pytest.importorskip("rustworkx")

    graph = rx.PyDiGraph()
    root_idx = graph.add_node(root)
    target_idx = graph.add_node(target)
    graph.add_edge(root_idx, target_idx, action)
    return graph


def test_stream_remove_matches_direct_encode(small_blocks):
    space, domain, problem = small_blocks
    root = problem.get_initial_state()
    succ = _first_successor(space, root)

    encoder = HGraphEncoder(domain)
    stream = HGraphMutableEncoderStream(encoder.engine)

    root_id = stream.append(root)
    _succ_id = stream.append(succ)
    stream.remove(root_id)

    encoding_dict = stream.flush()
    data = encoding_dict_to_pyg(encoding_dict)
    expected = encoder.encode(succ)
    assert hetero_data_equal(data, expected)


def test_stream_update_replaces_graph(small_blocks):
    space, domain, problem = small_blocks
    root = problem.get_initial_state()
    succ = _first_successor(space, root)

    encoder = HGraphEncoder(domain)
    stream = HGraphMutableEncoderStream(encoder.engine)

    root_id = stream.append(root)
    succ_id = stream.append(succ)
    stream.update(root_id, succ)
    stream.remove(succ_id)

    encoding_dict = stream.flush()
    data = encoding_dict_to_pyg(encoding_dict)
    expected = encoder.encode(succ)
    assert hetero_data_equal(data, expected)


def test_stream_reuse_removed_slot_keeps_ids_and_order(small_blocks):
    space, domain, problem = small_blocks
    root = problem.get_initial_state()
    succ = _first_successor(space, root)

    encoder = HGraphEncoder(domain)
    stream = HGraphMutableEncoderStream(encoder.engine)
    stream.set_reuse_removed(True)

    root_id = stream.append(root)
    _succ_id = stream.append(succ)
    stream.remove(root_id)
    reused_id = stream.append(root)

    assert reused_id == root_id

    encoding_dict = stream.flush()
    data = encoding_dict_to_pyg(encoding_dict)
    expected = encoder.encode_batch([root, succ])
    assert hetero_data_equal(data, expected)


def test_horizon_stream_remove_matches_direct_encode(small_blocks):
    space, domain, problem = small_blocks
    root = problem.get_initial_state()
    _succ, empty_dag, successor_dag = _horizon_dags(space, root)
    goals = list(problem.get_goal_condition().get_literals())

    encoder = HorizonEncoder(domain)
    stream = HorizonEncoderStream(encoder.engine)

    empty_id = stream.append(root, empty_dag, goals=goals)
    _full_id = stream.append(root, successor_dag, goals=goals)
    stream.remove(empty_id)

    encoding_dict = stream.flush()
    data = encoding_dict_to_pyg(encoding_dict)
    expected = encoder.encode(root, successor_dag, goals=goals)
    assert hetero_data_equal(data, expected)


def test_horizon_stream_update_replaces_graph(small_blocks):
    space, domain, problem = small_blocks
    root = problem.get_initial_state()
    _succ, empty_dag, successor_dag = _horizon_dags(space, root)
    goals = list(problem.get_goal_condition().get_literals())

    encoder = HorizonEncoder(domain)
    stream = HorizonEncoderStream(encoder.engine)

    empty_id = stream.append(root, empty_dag, goals=goals)
    full_id = stream.append(root, successor_dag, goals=goals)
    stream.update(empty_id, root, successor_dag, goals=goals)
    stream.remove(full_id)

    encoding_dict = stream.flush()
    data = encoding_dict_to_pyg(encoding_dict)
    expected = encoder.encode(root, successor_dag, goals=goals)
    assert hetero_data_equal(data, expected)


def test_horizon_stream_accepts_rustworkx_digraphs(small_blocks):
    pytest.importorskip("rustworkx")

    space, domain, problem = small_blocks
    root = problem.get_initial_state()
    transitions = [
        (action, target)
        for action, target in space.get_forward_transitions(root)
        if action is not None
        and target is not None
        and target.get_index() != root.get_index()
    ]
    if not transitions:
        pytest.skip("No successor transition available for horizon stream tests.")

    action, target = transitions[0]
    successor_dag = mifrost.TransitionDAG(adv_state(root))
    successor_dag.register_transition(
        adv_state(root), adv_state(target), adv_action(action)
    )
    successor_graph = _horizon_pygraph(root, action, target)
    goals = list(problem.get_goal_condition().get_literals())

    encoder = HorizonEncoder(domain)
    stream = HorizonEncoderStream(encoder.engine)
    stream.append(root, successor_graph, goals=goals)

    encoding_dict = stream.flush()
    data = encoding_dict_to_pyg(encoding_dict)
    expected = encoder.encode(root, successor_dag, goals=goals)
    assert hetero_data_equal(data, expected)

    stream = HorizonEncoderStream(encoder.engine)
    empty_id = stream.append(root, goals=goals)
    full_id = stream.append(root, successor_dag, goals=goals)
    stream.update(empty_id, root, successor_graph, goals=goals)
    stream.remove(full_id)

    encoding_dict = stream.flush()
    data = encoding_dict_to_pyg(encoding_dict)
    assert hetero_data_equal(data, expected)


def test_horizon_stream_rejects_mismatched_rustworkx_dag_root(small_blocks):
    pytest.importorskip("rustworkx")

    space, domain, problem = small_blocks
    root = problem.get_initial_state()
    transitions = [
        (action, target)
        for action, target in space.get_forward_transitions(root)
        if action is not None
        and target is not None
        and target.get_index() != root.get_index()
    ]
    if not transitions:
        pytest.skip("No successor transition available for horizon stream tests.")

    action, target = transitions[0]
    goals = list(problem.get_goal_condition().get_literals())
    mismatched_graph = _horizon_pygraph(target, action, root)

    encoder = HorizonEncoder(domain)
    stream = HorizonEncoderStream(encoder.engine)

    with pytest.raises(ValueError, match="dag root must match root state"):
        stream.append(root, mismatched_graph, goals=goals)


def test_hgraph_append_only_stream_matches_encode_batch(small_blocks):
    space, domain, problem = small_blocks
    root = problem.get_initial_state()
    succ = _first_successor(space, root)

    encoder = HGraphEncoder(domain)
    stream = HGraphEncoderStream(encoder.engine)
    stream.append(root)
    stream.append(succ)

    encoding_dict = stream.flush()
    data = encoding_dict_to_pyg(encoding_dict)
    expected = encoder.encode_batch([root, succ])
    assert hetero_data_equal(data, expected)


def test_hgraph_append_only_stream_has_no_update_remove(small_blocks):
    _space, domain, _problem = small_blocks
    encoder = HGraphEncoder(domain)
    stream = HGraphEncoderStream(encoder.engine)

    with pytest.raises(NotImplementedError):
        stream.remove(0)
    with pytest.raises(NotImplementedError):
        stream.update(0, object())


def test_hgraph_stream_append_rejects_nested_actions_with_horizon_guidance(
    small_blocks,
):
    space, domain, problem = small_blocks
    root = problem.get_initial_state()
    action0, action1 = _first_actions(space, root, count=2)

    encoder = HGraphEncoder(domain, ignore_actions=False)
    stream = HGraphEncoderStream(encoder.engine)

    with pytest.raises(ValueError, match="use HorizonEncoder for IW lookahead"):
        stream.append(root, actions=[(action0, action1)])


def test_hgraph_mutable_stream_update_rejects_nested_actions_with_horizon_guidance(
    small_blocks,
):
    space, domain, problem = small_blocks
    root = problem.get_initial_state()
    succ = _first_successor(space, root)
    action0, action1 = _first_actions(space, root, count=2)

    encoder = HGraphEncoder(domain, ignore_actions=False)
    stream = HGraphMutableEncoderStream(encoder.engine)
    stream_id = stream.append(root)

    with pytest.raises(ValueError, match="use HorizonEncoder for IW lookahead"):
        stream.update(stream_id, succ, actions=[(action0, action1)])
