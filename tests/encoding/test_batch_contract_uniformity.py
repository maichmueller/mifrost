from __future__ import annotations

from pathlib import Path

import pymimir
import pytest

from mifrost.encoders import (
    ColorEncoder,
    FlatHorizonEncoder,
    FlatRelationEncoder,
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


def _load_grounded_problem(domain: str, problem: str):
    root = Path(__file__).resolve().parents[2]
    domain_path = root / "data" / "pddl" / domain / "domain.pddl"
    problem_path = root / "data" / "pddl" / domain / f"{problem}.pddl"
    domain_obj = pymimir.Domain(domain_path)
    problem_obj = pymimir.Problem(domain_obj, problem_path, mode="grounded")
    return domain_obj, problem_obj


def _action_arity(action) -> int:
    return len(list(action.get_objects()))


def _find_distinct_action_arities(space, root):
    queue = [root]
    seen = {str(adv_state(root))}
    by_arity: dict[int, tuple[object, object]] = {}

    while queue and len(seen) < 32 and len(by_arity) < 2:
        state = queue.pop(0)
        for action, successor in space.get_forward_transitions(state):
            if action is None or successor is None:
                continue
            by_arity.setdefault(_action_arity(action), (state, action))
            succ_key = str(adv_state(successor))
            if succ_key not in seen:
                seen.add(succ_key)
                queue.append(successor)
            if len(by_arity) >= 2:
                break

    if len(by_arity) < 2:
        pytest.skip("Fixture does not provide actions with distinct arities.")

    return [by_arity[arity] for arity in sorted(by_arity)[:2]]


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


def test_color_encode_rejects_actions(small_blocks):
    space, domain, problem = small_blocks
    state = problem.get_initial_state()
    (action0, _succ0), _ = _first_transitions(space, state, count=2)
    encoder = ColorEncoder(domain)

    with pytest.raises(
        ValueError,
        match="ColorEncoderEngine does not support action encoding",
    ):
        encoder.encode(state, actions=[action0])


def test_color_batch_rejects_actions(small_blocks):
    space, domain, problem = small_blocks
    state = problem.get_initial_state()
    (action0, _succ0), _ = _first_transitions(space, state, count=2)
    encoder = ColorEncoder(domain)

    with pytest.raises(
        ValueError,
        match="Color batch encoding does not support explicit action payloads",
    ):
        encoder.encode_batch([state], actions=[action0])


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


@pytest.mark.parametrize(
    ("encoder_cls", "encoder_kwargs"),
    [
        (FlatRelationEncoder, {"max_goal_level": 1}),
        (FlatHorizonEncoder, {"max_goal_level": 1}),
    ],
)
def test_flat_batch_rejects_singleton_subgoal_layers(
    small_blocks,
    encoder_cls,
    encoder_kwargs,
):
    _space, domain, problem = small_blocks
    state = problem.get_initial_state()
    goals = _problem_goals(problem)
    if len(goals) < 2:
        pytest.skip("Fixture does not provide enough goal literals.")

    encoder = encoder_cls(domain, **encoder_kwargs)

    with pytest.raises(
        ValueError,
        match=(
            r"subgoal_layers entry at state index 0 looks like singleton layers "
            r"at positions 0, 1"
        ),
    ):
        encoder.encode_batch(
            [state],
            goals=[[goals[0]]],
            subgoal_layers=[[[goals[0]], [goals[1]]]],
        )


def test_transition_batch_requires_successors(small_blocks):
    _space, domain, problem = small_blocks
    state = problem.get_initial_state()
    encoder = TransitionHGraphEncoder(domain)

    with pytest.raises(
        ValueError,
        match="successors must be provided for transition batch encoding",
    ):
        encoder.encode_batch([state], goals=_problem_goals(problem))


def test_transition_encode_rejects_unsupported_payloads(small_blocks):
    space, domain, problem = small_blocks
    state = problem.get_initial_state()
    goals = _problem_goals(problem)
    (action0, successor), _ = _first_transitions(space, state, count=2)
    encoder = TransitionHGraphEncoder(domain)

    with pytest.raises(
        ValueError,
        match="Transition encoders do not support explicit action payloads",
    ):
        encoder.encode(
            state,
            successor=successor,
            actions=[action0],
        )

    if goals:
        with pytest.raises(
            ValueError,
            match="Transition encoders do not support history_subgoals payloads",
        ):
            encoder.encode(
                state,
                successor=successor,
                history_subgoals=[(-1, [goals[0]])],
                history_max_steps=7,
            )


def test_transition_batch_rejects_unsupported_payloads(small_blocks):
    space, domain, problem = small_blocks
    state = problem.get_initial_state()
    goals = _problem_goals(problem)
    (action0, successor), _ = _first_transitions(space, state, count=2)
    encoder = TransitionHGraphEncoder(domain)

    with pytest.raises(
        ValueError,
        match="Transition batch encoding does not support explicit action payloads",
    ):
        encoder.encode_batch(
            [state],
            successors=[successor],
            actions=[action0],
        )

    if goals:
        with pytest.raises(
            ValueError,
            match="Transition batch encoding does not support history_subgoals payloads",
        ):
            encoder.encode_batch(
                [state],
                successors=[successor],
                history_subgoals=[(-1, [goals[0]])],
                history_max_steps=7,
            )


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


def test_horizon_encode_rejects_unsupported_payloads(small_blocks):
    space, domain, problem = small_blocks
    root = problem.get_initial_state()
    goals = _problem_goals(problem)
    (action0, succ0), _ = _first_transitions(space, root, count=2)
    dag = _single_transition_dag(root, action0, succ0)
    encoder = HorizonEncoder(domain)

    with pytest.raises(
        ValueError,
        match="HorizonEncoder does not support explicit action payloads",
    ):
        encoder.encode(root, dag=dag, actions=[action0])

    if goals:
        with pytest.raises(
            ValueError,
            match="HorizonEncoder does not support history_subgoals payloads",
        ):
            encoder.encode(
                root,
                dag=dag,
                history_subgoals=[(-1, [goals[0]])],
                history_max_steps=1,
            )


def test_horizon_batch_rejects_unsupported_payloads(small_blocks):
    space, domain, problem = small_blocks
    root = problem.get_initial_state()
    goals = _problem_goals(problem)
    (action0, succ0), _ = _first_transitions(space, root, count=2)
    dag = _single_transition_dag(root, action0, succ0)
    encoder = HorizonEncoder(domain)

    with pytest.raises(
        ValueError,
        match="Horizon batch encoding does not support explicit action payloads",
    ):
        encoder.encode_batch(
            [root],
            dags=[dag],
            actions=[action0],
        )

    if goals:
        with pytest.raises(
            ValueError,
            match="Horizon batch encoding does not support history_subgoals payloads",
        ):
            encoder.encode_batch(
                [root],
                dags=[dag],
                history_subgoals=[(-1, [goals[0]])],
                history_max_steps=1,
            )


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


def test_ilg_grounded_batch_accepts_goals_and_actions():
    domain, problem = _load_grounded_problem("blocks", "smedium")
    state = problem.get_initial_state()
    goals = _problem_goals(problem)
    actions = list(state.generate_applicable_actions())
    if not actions:
        pytest.skip("Grounded fixture does not provide applicable actions.")

    encoder = ILGEncoder(domain)
    encoding = encoder.encode_batch(
        [state, state],
        goals=[goals, goals],
        actions=[[actions[0]], []],
    )
    data = encoding.as_pyg(as_batch=True)

    assert encoding.num_graphs == 2
    assert "action" in data.node_types
    assert data["action"].num_nodes == 1


def test_ilg_batch_accepts_mixed_wrapper_and_advanced_payloads(small_blocks):
    space, domain, problem = small_blocks
    state = problem.get_initial_state()
    goals = _problem_goals(problem)
    if not goals:
        pytest.skip("Fixture has no goals.")
    (action0, _succ0), _ = _first_transitions(space, state, count=2)

    encoder = ILGEncoder(domain)
    encoding = encoder.encode_batch(
        [state, state],
        goals=[
            goals,
            [getattr(goal, "_advanced_ground_literal", goal) for goal in goals],
        ],
        actions=[[action0], [adv_action(action0)]],
        subgoal_layers=[None, None],
    )

    assert encoding.num_graphs == 2


def test_ilg_batch_supports_heterogeneous_action_arities(small_blocks):
    space, domain, problem = small_blocks
    root = problem.get_initial_state()
    action_states = _find_distinct_action_arities(space, root)
    states = [entry[0] for entry in action_states]
    actions = [[entry[1]] for entry in action_states]
    max_arity = max(_action_arity(entry[1]) for entry in action_states)

    encoder = ILGEncoder(domain)
    data = encoder.encode_batch(states, actions=actions).as_pyg(as_batch=True)

    assert "action" in data.node_types
    assert data["action"].num_nodes == len(action_states)
    assert data["action"].x.shape[1] == max_arity + 1


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


def test_core_batch_input_boundary_adapts_all_standard_lanes(small_blocks):
    space, _domain, problem = small_blocks
    state = problem.get_initial_state()
    goals = _problem_goals(problem)
    if not goals:
        pytest.skip("Fixture has no goals.")
    (action, successor), _ = _first_transitions(space, state, count=2)

    from mifrost.encoders._batch_contract import prepare_core_batch_inputs
    from mifrost.encoders.types import BatchParam

    inputs = prepare_core_batch_inputs(
        BatchParam.separate([state]),
        goals=BatchParam.shared([goals[0]]),
        actions=BatchParam.separate([[action]]),
        subgoal_layers=[[[goals[0]]]],
        history_subgoals=[[(0, [goals[0]])]],
        successors=BatchParam.shared(successor),
    )

    advanced_goal = getattr(goals[0], "_advanced_ground_literal", goals[0])
    assert inputs.states == BatchParam.separate([adv_state(state)])
    assert inputs.goals == BatchParam.shared([advanced_goal])
    assert inputs.actions == BatchParam.separate([[adv_action(action)]])
    assert inputs.subgoal_layers == [[[advanced_goal]]]
    assert inputs.history_subgoals == [[(0, [advanced_goal])]]
    assert inputs.successors == BatchParam.shared(adv_state(successor))


def test_lane_optional_payloads_prepare_actions_and_history(small_blocks):
    space, _domain, problem = small_blocks
    state = problem.get_initial_state()
    goals = _problem_goals(problem)
    if not goals:
        pytest.skip("Fixture has no goals.")
    transitions = _first_transitions(space, state, count=1)
    if not transitions:
        pytest.skip("Fixture does not provide forward transitions.")
    action0, _ = transitions[0]

    from mifrost.encoders._lane_specs import prepare_optional_payloads

    payloads = prepare_optional_payloads(
        actions=[action0],
        history_subgoals=[(0, [goals[0]])],
    )

    assert payloads.actions == [adv_action(action0)]
    assert payloads.history_subgoals == [
        (0, [getattr(goals[0], "_advanced_ground_literal", goals[0])])
    ]


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
