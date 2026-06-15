from __future__ import annotations

import pytest

import mifrost
from mifrost.encoders import (
    FlatHorizonEncoder,
    FlatHorizonMutableEncoderStream,
    FlatHorizonEncoderStream,
    FlatRelationData,
    FlatRelationEncoder,
    FlatRelationEncoderStream,
    FlatRelationMutableEncoderStream,
    FlatTransitionEffectsEncoder,
    FlatTransitionEffectsEncoderStream,
    FlatTransitionEncoder,
    FlatTransitionEncoderStream,
)

from .test_flat_horizon_encoder import (
    _first_distinct_changed_transitions,
    _single_step_dag,
)
from .test_flat_relation_encoder import _assert_flat_batch_equal, _history_inputs
from .test_stream_encoder import _first_successor, _horizon_pygraph


def test_flat_relation_append_only_stream_matches_encode_batch(small_blocks):
    space, domain, problem = small_blocks
    root = problem.get_initial_state()
    succ = _first_successor(space, root)

    encoder = FlatRelationEncoder(domain)
    stream = encoder.stream()

    assert isinstance(stream, FlatRelationEncoderStream)
    stream.append(root)
    stream.append(succ)

    actual = stream.flush_pyg()
    expected = encoder.encode_batch([root, succ]).as_pyg(as_batch=True)

    assert isinstance(actual, FlatRelationData)
    _assert_flat_batch_equal(actual, expected)


def test_flat_relation_append_only_stream_has_no_update_remove(small_blocks):
    _space, domain, _problem = small_blocks
    stream = FlatRelationEncoder(domain).stream()

    with pytest.raises(NotImplementedError):
        stream.remove(0)
    with pytest.raises(NotImplementedError):
        stream.update(0, object())


def test_flat_relation_mutable_stream_remove_update_and_reuse_match_direct_encode(
    small_blocks,
):
    space, domain, problem = small_blocks
    root = problem.get_initial_state()
    succ = _first_successor(space, root)

    encoder = FlatRelationEncoder(domain)
    stream = encoder.mutable_stream()

    assert isinstance(stream, FlatRelationMutableEncoderStream)
    root_id = stream.append(root)
    succ_id = stream.append(succ)
    stream.update(root_id, succ)
    stream.remove(succ_id)

    actual = stream.flush_pyg()
    expected = encoder.encode_batch([succ]).as_pyg(as_batch=True)
    _assert_flat_batch_equal(actual, expected)

    stream = encoder.mutable_stream()
    stream.set_reuse_removed(True)
    root_id = stream.append(root)
    _succ_id = stream.append(succ)
    stream.remove(root_id)
    reused_id = stream.append(root)

    assert reused_id == root_id

    actual = stream.flush_pyg()
    expected = encoder.encode_batch([root, succ]).as_pyg(as_batch=True)
    _assert_flat_batch_equal(actual, expected)


def test_flat_relation_mutable_stream_preserves_actions_goals_and_history(small_blocks):
    space, domain, problem = small_blocks
    state = problem.get_initial_state()
    transitions = list(space.get_forward_transitions(state))
    actions = [action for action, _target in transitions if action is not None][:2]
    if len(actions) < 2:
        pytest.skip("Fixture does not provide enough actions.")
    goals, history_subgoals = _history_inputs(problem)

    encoder = FlatRelationEncoder(
        domain,
        target_sources={"goal", "action", "history"},
    )
    stream = encoder.mutable_stream()
    stream.append(
        state,
        goals=goals,
        actions=actions,
        history_subgoals=history_subgoals,
        history_max_steps=2,
    )

    actual = stream.flush_pyg()
    expected = encoder.encode_batch(
        [state],
        goals=[goals],
        actions=[actions],
        history_subgoals=[history_subgoals],
        history_max_steps=2,
    ).as_pyg(as_batch=True)

    _assert_flat_batch_equal(actual, expected)


def test_flat_horizon_stream_matches_direct_encode_and_accepts_rustworkx(small_blocks):
    pytest.importorskip("rustworkx")

    space, domain, problem = small_blocks
    root = problem.get_initial_state()
    transitions = _first_distinct_changed_transitions(space, root, count=1)
    if not transitions:
        pytest.skip("Fixture should yield at least one distinct changed transition")
    dag = _single_step_dag(root, transitions, candidate_ids=[101])
    goals = list(problem.get_goal_condition().get_literals())

    encoder = FlatHorizonEncoder(domain, ignore_actions=False)
    stream = encoder.stream()

    assert isinstance(stream, FlatHorizonEncoderStream)
    stream.append(root, goals=goals)
    stream.append(root, dag=dag, goals=goals)

    actual = stream.flush_pyg()
    expected = encoder.encode_batch(
        [root, root], dags=[None, dag], goals=[goals, goals]
    ).as_pyg(as_batch=True)

    _assert_flat_batch_equal(actual, expected)
    assert actual.graph_target_depths(1).tolist() == [1]


def test_flat_horizon_append_only_stream_has_no_update_remove(small_blocks):
    _space, domain, _problem = small_blocks
    stream = FlatHorizonEncoder(domain).stream()

    with pytest.raises(NotImplementedError):
        stream.remove(0)
    with pytest.raises(NotImplementedError):
        stream.update(0, object())


def test_flat_horizon_mutable_stream_matches_direct_encode_and_accepts_rustworkx(
    small_blocks,
):
    pytest.importorskip("rustworkx")

    space, domain, problem = small_blocks
    root = problem.get_initial_state()
    transitions = _first_distinct_changed_transitions(space, root, count=1)
    action, successor = transitions[0]
    dag = _single_step_dag(root, transitions, candidate_ids=[101])
    goals = list(problem.get_goal_condition().get_literals())
    graph = _horizon_pygraph(root, action, successor)

    encoder = FlatHorizonEncoder(domain, ignore_actions=False)
    stream = encoder.mutable_stream()

    assert isinstance(stream, FlatHorizonMutableEncoderStream)
    empty_id = stream.append(root, goals=goals)
    full_id = stream.append(root, dag=dag, goals=goals)
    stream.update(empty_id, root, dag=graph, goals=goals)
    stream.remove(full_id)

    actual = stream.flush_pyg()
    expected = encoder.encode_batch([root], dags=[graph], goals=[goals]).as_pyg(
        as_batch=True
    )

    _assert_flat_batch_equal(actual, expected)
    assert actual.graph_target_depths(0).tolist() == [1]


def test_flat_horizon_stream_rejects_mismatched_rustworkx_dag_root(small_blocks):
    pytest.importorskip("rustworkx")

    space, domain, problem = small_blocks
    root = problem.get_initial_state()
    transitions = _first_distinct_changed_transitions(space, root, count=1)
    _action, successor = transitions[0]
    goals = list(problem.get_goal_condition().get_literals())
    rx = pytest.importorskip("rustworkx")
    mismatched_graph = rx.PyDiGraph()
    mismatched_graph.add_node(successor)

    encoder = FlatHorizonEncoder(domain, ignore_actions=False)
    stream = encoder.stream()

    with pytest.raises(ValueError, match="dag root must match root state"):
        stream.append(root, dag=mismatched_graph, goals=goals)


def test_flat_transition_stream_matches_direct_encode(small_blocks):
    space, domain, problem = small_blocks
    state = problem.get_initial_state()
    transitions = _first_distinct_changed_transitions(space, state, count=2)
    if len(transitions) < 2:
        pytest.skip("Fixture should yield at least 2 distinct changed transitions")

    successor0 = transitions[0][1]
    successor1 = transitions[1][1]
    goals = list(problem.get_goal_condition().get_literals())
    encoder = FlatTransitionEncoder(domain)
    stream = encoder.stream()

    assert isinstance(stream, FlatTransitionEncoderStream)
    first_id = stream.append(state, successor0, goals=goals)
    second_id = stream.append(state, successor1, goals=goals)
    stream.update(first_id, state, successor1, goals=goals)
    stream.remove(second_id)

    actual = stream.flush_pyg()
    expected = encoder.encode_batch(
        [state],
        successors=[successor1],
        goals=[goals],
    ).as_pyg(as_batch=True)

    _assert_flat_batch_equal(actual, expected)
    assert actual.target_entity_groups == ["state"]


def test_flat_transition_effects_stream_matches_direct_encode(small_blocks):
    space, domain, problem = small_blocks
    state = problem.get_initial_state()
    transitions = _first_distinct_changed_transitions(space, state, count=1)
    successor = transitions[0][1]
    encoder = FlatTransitionEffectsEncoder(domain)
    stream = encoder.stream()

    assert isinstance(stream, FlatTransitionEffectsEncoderStream)
    stream.append(state, successor, goals=[])

    actual = stream.flush_pyg()
    expected = encoder.encode_batch([state], successors=[successor], goals=[[]]).as_pyg(
        as_batch=True
    )

    _assert_flat_batch_equal(actual, expected)


def test_flat_stream_classes_export_from_root_module():
    assert hasattr(mifrost, "FlatRelationEncoderStream")
    assert hasattr(mifrost, "FlatRelationMutableEncoderStream")
    assert hasattr(mifrost, "FlatHorizonEncoderStream")
    assert hasattr(mifrost, "FlatHorizonMutableEncoderStream")
    assert hasattr(mifrost, "FlatTransitionEncoderStream")
    assert hasattr(mifrost, "FlatTransitionEffectsEncoderStream")
