from __future__ import annotations

import pytest

from mifrost.encoders import BatchParam, HGraphEncoder


def _problem_goals(problem):
    return list(problem.get_goal_condition().get_literals())


def _first_action(space, state):
    transitions = list(space.get_forward_transitions(state))
    if not transitions:
        pytest.skip("Fixture does not provide forward transitions.")
    return transitions[0][0]


def test_batch_param_shared_and_per_state(small_blocks):
    space, domain, problem = small_blocks
    state = problem.get_initial_state()
    goals = _problem_goals(problem)
    action = _first_action(space, state)

    encoder = HGraphEncoder(domain, ignore_actions=False)

    shared = encoder.encode_batch(
        [state, state],
        goals=BatchParam.shared(goals),
        actions=BatchParam.shared([action]),
        subgoal_layers=BatchParam.shared([[goals[0]]] if goals else []),
    )
    assert shared.num_graphs == 2

    separate = encoder.encode_batch(
        [state, state],
        goals=BatchParam.separate([goals, goals]),
        actions=BatchParam.separate([[action], None]),
    )
    assert separate.num_graphs == 2


def test_batch_param_none_passthrough(small_blocks):
    space, domain, problem = small_blocks
    state = problem.get_initial_state()
    action = _first_action(space, state)

    encoder = HGraphEncoder(domain, ignore_actions=False)
    encoding = encoder.encode_batch(
        [state],
        goals=BatchParam.none(),
        actions=BatchParam.shared([action]),
    )
    assert encoding.num_graphs == 1


def test_batch_param_requires_sequence_for_per_state(small_blocks):
    _space, domain, problem = small_blocks
    state = problem.get_initial_state()
    encoder = HGraphEncoder(domain)

    with pytest.raises(
        TypeError, match="BatchParam\\(separate\\) value must be a sequence"
    ):
        encoder.encode_batch(
            [state],
            goals=BatchParam(kind="separate", value=(x for x in [None])),
        )
