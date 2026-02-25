from __future__ import annotations

import pytest

from mifrost.encoders import (
    ColorEncoder,
    HGraphEncoder,
    HorizonEncoder,
    ILGEncoder,
    TransitionHGraphEncoder,
)

import mifrost

from .test_utils import adv_action, adv_state


def _first_transitions(space, state, count: int = 2):
    transitions = [
        (action, target)
        for action, target in space.get_forward_transitions(state)
        if action is not None and target is not None
    ]
    if len(transitions) < count:
        pytest.skip("Fixture does not provide enough transitions.")
    return transitions[:count]


def _problem_goals(problem):
    return list(problem.get_goal_condition().get_literals())


def _single_transition_dag(root, action, successor):
    dag = mifrost.TransitionDAG(adv_state(root))
    dag.register_transition(
        adv_state(root),
        adv_state(successor),
        adv_action(action),
    )
    return dag


def test_hgraph_batch_accepts_shared_and_per_state_payloads(small_blocks):
    space, domain, problem = small_blocks
    state = problem.get_initial_state()
    goals = _problem_goals(problem)
    (action0, _succ0), (action1, _succ1) = _first_transitions(space, state, count=2)
    encoder = HGraphEncoder(domain, ignore_actions=False)

    shared = encoder.encode_batch(
        [state, state],
        goals=goals,
        actions=[action0],
    )
    assert shared.num_graphs == 2

    per_state = encoder.encode_batch(
        [state, state],
        goals=[goals, goals],
        actions=[[action0], [action1]],
        subgoal_layers=[[[goals[0]]], None] if goals else [None, None],
        history_subgoals=[[(-1, [goals[0]])], None] if goals else [None, None],
    )
    assert per_state.num_graphs == 2


def test_hgraph_batch_rejects_per_state_length_mismatch(small_blocks):
    space, domain, problem = small_blocks
    state = problem.get_initial_state()
    (action0, _succ0), _ = _first_transitions(space, state, count=2)
    encoder = HGraphEncoder(domain, ignore_actions=False)

    with pytest.raises(ValueError, match="actions length must match states length"):
        encoder.encode_batch([state, state], actions=[[action0]])

    with pytest.raises(ValueError, match="goals length must match states length"):
        encoder.encode_batch([state, state], goals=[_problem_goals(problem)])


def test_color_batch_rejects_actions(small_blocks):
    _space, domain, problem = small_blocks
    state = problem.get_initial_state()
    encoder = ColorEncoder(domain)

    with pytest.raises(
        TypeError,
        match="ColorEncoder does not accept 'actions' in encode_batch",
    ):
        encoder.encode_batch([state], actions=[])


def test_color_batch_accepts_per_state_goals_and_subgoal_layers(small_blocks):
    _space, domain, problem = small_blocks
    state = problem.get_initial_state()
    goals = _problem_goals(problem)
    encoder = ColorEncoder(domain)

    encoding = encoder.encode_batch(
        [state, state],
        goals=[goals, goals],
        subgoal_layers=[None, None],
    )
    assert encoding.num_graphs == 2


def test_transition_batch_requires_successors(small_blocks):
    _space, domain, problem = small_blocks
    state = problem.get_initial_state()
    encoder = TransitionHGraphEncoder(domain)

    with pytest.raises(
        ValueError,
        match="successors must be provided for transition batch encoding",
    ):
        encoder.encode_batch([state], goals=_problem_goals(problem))


def test_transition_batch_rejects_actions(small_blocks):
    space, domain, problem = small_blocks
    state = problem.get_initial_state()
    (_action0, successor), _ = _first_transitions(space, state, count=2)
    encoder = TransitionHGraphEncoder(domain)

    with pytest.raises(
        TypeError,
        match="TransitionHGraphEncoder does not accept 'actions' in encode_batch",
    ):
        encoder.encode_batch([state], successors=[successor], actions=[])

    with pytest.raises(
        TypeError,
        match="TransitionHGraphEncoder does not accept 'history_subgoals' in encode_batch",
    ):
        encoder.encode_batch([state], successors=[successor], history_subgoals=[])


def test_transition_batch_accepts_per_state_goals_subgoal_layers(small_blocks):
    space, domain, problem = small_blocks
    state = problem.get_initial_state()
    goals = _problem_goals(problem)
    (_a0, succ0), (_a1, succ1) = _first_transitions(space, state, count=2)
    encoder = TransitionHGraphEncoder(domain)

    encoding = encoder.encode_batch(
        [state, state],
        successors=[succ0, succ1],
        goals=[goals, goals],
        subgoal_layers=[None, None],
    )
    assert encoding.num_graphs == 2


def test_transition_batch_rejects_successor_length_mismatch(small_blocks):
    space, domain, problem = small_blocks
    state = problem.get_initial_state()
    (_a0, succ0), _ = _first_transitions(space, state, count=2)
    encoder = TransitionHGraphEncoder(domain)

    with pytest.raises(ValueError, match="successors length must match states length"):
        encoder.encode_batch([state, state], successors=[succ0])


def test_horizon_batch_accepts_per_state_goals_and_dags(small_blocks):
    space, domain, problem = small_blocks
    root = problem.get_initial_state()
    goals = _problem_goals(problem)
    (action0, succ0), _ = _first_transitions(space, root, count=2)
    dag = _single_transition_dag(root, action0, succ0)
    encoder = HorizonEncoder(domain, ignore_actions=False)

    encoding = encoder.encode_batch(
        [root, root],
        dags=[dag, None],
        goals=[goals, None],
        subgoal_layers=[None, None],
    )
    assert encoding.num_graphs == 2


def test_horizon_batch_rejects_unsupported_fields(small_blocks):
    space, domain, problem = small_blocks
    root = problem.get_initial_state()
    (action0, succ0), _ = _first_transitions(space, root, count=2)
    dag = _single_transition_dag(root, action0, succ0)
    encoder = HorizonEncoder(domain)

    with pytest.raises(
        TypeError,
        match="HorizonEncoder does not accept 'actions' in encode_batch",
    ):
        encoder.encode_batch([root], dags=[dag], actions=[])

    with pytest.raises(
        TypeError,
        match="HorizonEncoder does not accept 'history_subgoals' in encode_batch",
    ):
        encoder.encode_batch([root], dags=[dag], history_subgoals=[])

    with pytest.raises(
        TypeError,
        match="HorizonEncoder does not accept 'history_max_steps' in encode_batch",
    ):
        encoder.encode_batch([root], dags=[dag], history_max_steps=1)


def test_horizon_batch_rejects_dag_length_mismatch(small_blocks):
    space, domain, problem = small_blocks
    root = problem.get_initial_state()
    (action0, succ0), _ = _first_transitions(space, root, count=2)
    dag = _single_transition_dag(root, action0, succ0)
    encoder = HorizonEncoder(domain)

    with pytest.raises(ValueError, match="dags length must match states length"):
        encoder.encode_batch([root, root], dags=[dag])


def test_ilg_batch_accepts_per_state_goals_actions_and_subgoal_layers(small_blocks):
    space, domain, problem = small_blocks
    state = problem.get_initial_state()
    goals = _problem_goals(problem)
    (action0, _succ0), _ = _first_transitions(space, state, count=2)
    encoder = ILGEncoder(domain)

    encoding = encoder.encode_batch(
        [state, state],
        goals=[goals, goals],
        actions=[[action0], []],
        subgoal_layers=[None, None],
    )
    data = encoding.as_pyg(as_batch=True)
    assert encoding.num_graphs == 2
    assert "action" in data.node_types
    assert data["action"].num_nodes == 1
