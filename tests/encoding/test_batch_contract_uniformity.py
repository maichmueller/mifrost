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


def test_color_batch_ignores_actions(small_blocks):
    _space, domain, problem = small_blocks
    state = problem.get_initial_state()
    encoder = ColorEncoder(domain)

    encoding = encoder.encode_batch([state], actions=[object()])
    assert encoding.num_graphs == 1


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


def test_transition_batch_ignores_unsupported_kwargs(small_blocks):
    space, domain, problem = small_blocks
    state = problem.get_initial_state()
    (_action0, successor), _ = _first_transitions(space, state, count=2)
    encoder = TransitionHGraphEncoder(domain)

    encoding = encoder.encode_batch(
        [state],
        successors=[successor],
        actions=[object()],
        history_subgoals=[object()],
        history_max_steps=7,
    )
    assert encoding.num_graphs == 1


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


def test_horizon_batch_ignores_unsupported_fields(small_blocks):
    space, domain, problem = small_blocks
    root = problem.get_initial_state()
    (action0, succ0), _ = _first_transitions(space, root, count=2)
    dag = _single_transition_dag(root, action0, succ0)
    encoder = HorizonEncoder(domain)

    encoding = encoder.encode_batch(
        [root],
        dags=[dag],
        actions=[object()],
        history_subgoals=[object()],
        history_max_steps=1,
    )
    assert encoding.num_graphs == 1


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


def test_hgraph_advanced_batch_derives_default_goals_with_aux_payload(small_blocks):
    space, domain, problem = small_blocks
    state = problem.get_initial_state()
    transitions = _first_transitions(space, state, count=1)
    if not transitions:
        pytest.skip("Fixture does not provide forward transitions.")
    action0, _ = transitions[0]

    encoder = HGraphEncoder(domain, ignore_actions=False)
    encoding = encoder.encode_batch(
        [adv_state(state)],
        actions=[adv_action(action0)],
    )
    assert encoding.num_graphs == 1
    assert encoding.num_nodes > 0


def test_wrapper_and_advanced_batch_parity_across_encoders(small_blocks):
    space, domain, problem = small_blocks
    state = problem.get_initial_state()
    goals = _problem_goals(problem)
    if not goals:
        pytest.skip("Fixture has no goals.")
    goals_adv = [getattr(goal, "_advanced_ground_literal", goal) for goal in goals]
    (action0, succ0), _ = _first_transitions(space, state, count=2)

    hgraph = HGraphEncoder(domain, ignore_actions=False)
    hgraph_wrapper = hgraph.encode_batch([state], goals=goals, actions=[action0])
    hgraph_advanced = hgraph.encode_batch(
        [adv_state(state)],
        goals=goals_adv,
        actions=[adv_action(action0)],
    )
    assert hgraph_wrapper.num_nodes == hgraph_advanced.num_nodes
    assert hgraph_wrapper.num_edges == hgraph_advanced.num_edges

    color = ColorEncoder(domain)
    color_wrapper = color.encode_batch([state], goals=goals)
    color_advanced = color.encode_batch([adv_state(state)], goals=goals_adv)
    assert color_wrapper.num_nodes == color_advanced.num_nodes
    assert color_wrapper.num_edges == color_advanced.num_edges

    transition = TransitionHGraphEncoder(domain)
    transition_wrapper = transition.encode_batch(
        [state],
        successors=[succ0],
        goals=[goals],
        subgoal_layers=[None],
    )
    transition_advanced = transition.encode_batch(
        [adv_state(state)],
        successors=[adv_state(succ0)],
        goals=[goals_adv],
        subgoal_layers=[None],
    )
    assert transition_wrapper.num_nodes == transition_advanced.num_nodes
    assert transition_wrapper.num_edges == transition_advanced.num_edges

    horizon = HorizonEncoder(domain, ignore_actions=False)
    dag = _single_transition_dag(state, action0, succ0)
    horizon_wrapper = horizon.encode_batch([state], dags=[dag], goals=[goals])
    horizon_advanced = horizon.encode_batch(
        [adv_state(state)],
        dags=[dag],
        goals=[goals_adv],
    )
    assert horizon_wrapper.num_nodes == horizon_advanced.num_nodes
    assert horizon_wrapper.num_edges == horizon_advanced.num_edges

    ilg = ILGEncoder(domain)
    ilg_wrapper = ilg.encode_batch([state], goals=[goals], actions=[[action0]])
    ilg_advanced = ilg.encode_batch(
        [state],
        goals=[goals_adv],
        actions=[[adv_action(action0)]],
    )
    assert ilg_wrapper.num_nodes == ilg_advanced.num_nodes
    assert ilg_wrapper.num_edges == ilg_advanced.num_edges


def test_batch_accepts_state_adapters(small_blocks):
    _space, domain, problem = small_blocks
    state = problem.get_initial_state()
    encoder = HGraphEncoder(domain)

    class WrappedState:
        pass

    mifrost.register_state_adapter(
        WrappedState,
        lambda _: adv_state(state),
    )
    try:
        encoding = encoder.encode_batch([WrappedState()])
        assert encoding.num_graphs == 1
    finally:
        mifrost.unregister_state_adapter(WrappedState)


def test_batch_accepts_literal_adapters(small_blocks):
    _space, domain, problem = small_blocks
    state = problem.get_initial_state()
    goals = _problem_goals(problem)
    if not goals:
        pytest.skip("Fixture has no goals.")

    encoder = HGraphEncoder(domain)

    class WrappedLiteral:
        pass

    mifrost.register_literal_adapter(
        WrappedLiteral,
        lambda _: getattr(goals[0], "_advanced_ground_literal", goals[0]),
    )
    try:
        encoding = encoder.encode_batch([state], goals=[WrappedLiteral()])
        assert encoding.num_graphs == 1
    finally:
        mifrost.unregister_literal_adapter(WrappedLiteral)


def test_batch_accepts_action_adapters(small_blocks):
    space, domain, problem = small_blocks
    state = problem.get_initial_state()
    transitions = _first_transitions(space, state, count=1)
    if not transitions:
        pytest.skip("Fixture does not provide forward transitions.")
    action0, _ = transitions[0]

    encoder = HGraphEncoder(domain, ignore_actions=False)

    class WrappedAction:
        pass

    mifrost.register_action_adapter(
        WrappedAction,
        lambda _: adv_action(action0),
    )
    try:
        encoding = encoder.encode_batch([state], actions=[WrappedAction()])
        assert encoding.num_graphs == 1
    finally:
        mifrost.unregister_action_adapter(WrappedAction)


def test_core_parse_states_rejects_state_adapters(small_blocks):
    _space, domain, problem = small_blocks
    state = problem.get_initial_state()

    class WrappedState:
        pass

    mifrost.register_state_adapter(
        WrappedState,
        lambda _: adv_state(state),
    )
    try:
        with pytest.raises(
            TypeError,
            match=(
                "Batch parsing does not support state adapters"
                "|encode_batch expects a state or an iterable of states"
            ),
        ):
            mifrost._core._parse_states_batch([WrappedState()])
    finally:
        mifrost.unregister_state_adapter(WrappedState)


def test_core_parse_goals_rejects_literal_adapters(small_blocks):
    _space, domain, problem = small_blocks
    state = problem.get_initial_state()
    goals = _problem_goals(problem)
    if not goals:
        pytest.skip("Fixture has no goals.")

    class WrappedLiteral:
        pass

    mifrost.register_literal_adapter(
        WrappedLiteral,
        lambda _: getattr(goals[0], "_advanced_ground_literal", goals[0]),
    )
    try:
        with pytest.raises(
            TypeError,
            match="Batch parsing does not support literal adapters",
        ):
            mifrost._core._parse_goals_batch_param([WrappedLiteral()], 1)
    finally:
        mifrost.unregister_literal_adapter(WrappedLiteral)


def test_core_parse_actions_rejects_action_adapters(small_blocks):
    space, domain, problem = small_blocks
    state = problem.get_initial_state()
    transitions = _first_transitions(space, state, count=1)
    if not transitions:
        pytest.skip("Fixture does not provide forward transitions.")
    action0, _ = transitions[0]

    class WrappedAction:
        pass

    mifrost.register_action_adapter(
        WrappedAction,
        lambda _: adv_action(action0),
    )
    try:
        with pytest.raises(
            TypeError,
            match="Batch parsing does not support action adapters",
        ):
            mifrost._core._parse_actions_batch_param([WrappedAction()], 1)
    finally:
        mifrost.unregister_action_adapter(WrappedAction)


def test_core_parse_states_rejects_wrapper_states(small_blocks):
    _space, _domain, problem = small_blocks
    state = problem.get_initial_state()

    with pytest.raises(
        TypeError,
        match=(
            "states entry at index 0 has invalid type"
            "|encode_batch expects a state or an iterable of states"
        ),
    ):
        mifrost._core._parse_states_batch([state])


def test_core_parse_goals_rejects_invalid_literal_types():
    with pytest.raises(
        TypeError,
        match="has invalid goal literal type",
    ):
        mifrost._core._parse_goals_batch_param([object()], 1)
