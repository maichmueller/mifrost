from __future__ import annotations

import pytest

import mifrost
from mifrost.encoders import (
    HGraphEncoder,
    HGraphEncoderStream,
    HorizonEncoder,
    HorizonEncoderStream,
)

from .test_utils import adv_action, adv_state, hetero_data_equal, parts_to_pyg


def _first_successor(space, state):
    transitions = list(space.get_forward_transitions(state))
    for _action, target in transitions:
        if target is None:
            continue
        if target.get_index() != state.get_index():
            return target
    pytest.skip("No successor state available for stream tests.")


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


def test_stream_remove_matches_direct_encode(small_blocks):
    space, domain, problem = small_blocks
    root = problem.get_initial_state()
    succ = _first_successor(space, root)

    encoder = HGraphEncoder(domain)
    stream = HGraphEncoderStream(encoder.engine)

    root_id = stream.append(root)
    _succ_id = stream.append(succ)
    stream.remove(root_id)

    parts = stream.flush_parts()
    data = parts_to_pyg(parts)
    expected = encoder.encode(succ)
    assert hetero_data_equal(data, expected)


def test_stream_update_replaces_graph(small_blocks):
    space, domain, problem = small_blocks
    root = problem.get_initial_state()
    succ = _first_successor(space, root)

    encoder = HGraphEncoder(domain)
    stream = HGraphEncoderStream(encoder.engine)

    root_id = stream.append(root)
    succ_id = stream.append(succ)
    stream.update(root_id, succ)
    stream.remove(succ_id)

    parts = stream.flush_parts()
    data = parts_to_pyg(parts)
    expected = encoder.encode(succ)
    assert hetero_data_equal(data, expected)


def test_stream_reuse_removed_slot_keeps_ids_and_order(small_blocks):
    space, domain, problem = small_blocks
    root = problem.get_initial_state()
    succ = _first_successor(space, root)

    encoder = HGraphEncoder(domain)
    stream = HGraphEncoderStream(encoder.engine)
    stream.set_reuse_removed(True)

    root_id = stream.append(root)
    _succ_id = stream.append(succ)
    stream.remove(root_id)
    reused_id = stream.append(root)

    assert reused_id == root_id

    parts = stream.flush_parts()
    data = parts_to_pyg(parts)
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

    parts = stream.flush_parts()
    data = parts_to_pyg(parts)
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

    parts = stream.flush_parts()
    data = parts_to_pyg(parts)
    expected = encoder.encode(root, successor_dag, goals=goals)
    assert hetero_data_equal(data, expected)
