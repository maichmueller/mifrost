from __future__ import annotations

import pytest

import mifrost

from mifrost.encoders import (
    BatchParam,
    HGraphEncoder,
    HorizonEncoder,
    TransitionHGraphEncoder,
)

from .test_utils import adv_action, adv_state, hetero_data_equal


def _problem_goals(problem):
    return list(problem.get_goal_condition().get_literals())


def _first_action(space, state):
    transitions = list(space.get_forward_transitions(state))
    if not transitions:
        pytest.skip("Fixture does not provide forward transitions.")
    return transitions[0][0]


def _first_transitions(space, state, count: int = 2):
    transitions = [
        (action, target)
        for action, target in space.get_forward_transitions(state)
        if action is not None and target is not None
    ]
    if len(transitions) < count:
        pytest.skip("Fixture does not provide enough transitions.")
    return transitions[:count]


def _distinct_changed_transitions(space, state, count: int = 2):
    root_repr = str(adv_state(state))
    out = []
    seen_targets = set()
    for action, target in space.get_forward_transitions(state):
        if action is None or target is None:
            continue
        target_repr = str(adv_state(target))
        if target_repr == root_repr or target_repr in seen_targets:
            continue
        seen_targets.add(target_repr)
        out.append((action, target))
        if len(out) >= count:
            return out
    pytest.skip("Fixture does not provide enough distinct changed transitions.")


def _single_transition_dag(root, action, successor):
    dag = mifrost.TransitionDAG(adv_state(root))
    dag.register_transition(
        adv_state(root),
        adv_state(successor),
        adv_action(action),
    )
    return dag


def _single_transition_pygraph(root, action, successor):
    rx = pytest.importorskip("rustworkx")

    graph = rx.PyDiGraph()
    root_idx = graph.add_node(root)
    succ_idx = graph.add_node(successor)
    graph.add_edge(root_idx, succ_idx, action)
    return graph


def _successor_node_names(encoding) -> list[str]:
    data = encoding.as_pyg(as_batch=True)
    names: list[str] = []
    for node_type in data.node_types:
        if "[suc]" not in node_type:
            continue
        node_names = getattr(data[node_type], "node_names", None)
        if not node_names:
            continue
        for entry in node_names:
            if isinstance(entry, list):
                names.extend(str(item) for item in entry)
            else:
                names.append(str(entry))
    return sorted(names)


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


def test_transition_batch_param_supports_shared_and_separate_successors(small_blocks):
    space, domain, problem = small_blocks
    state = problem.get_initial_state()
    goals = _problem_goals(problem)
    (_action0, succ0), (_action1, succ1) = _distinct_changed_transitions(
        space, state, count=2
    )

    encoder = TransitionHGraphEncoder(domain)

    shared = encoder.encode_batch(
        [state, state],
        successors=BatchParam.shared(succ0),
        goals=goals,
    )
    repeated_shared = encoder.encode_batch(
        [state, state],
        successors=[succ0, succ0],
        goals=goals,
    )
    self_successor = encoder.encode_batch(
        [state, state],
        successors=BatchParam.shared(state),
        goals=goals,
    )
    assert shared.num_graphs == 2
    assert hetero_data_equal(shared, repeated_shared)
    assert _successor_node_names(shared) == _successor_node_names(repeated_shared)
    assert _successor_node_names(shared) != _successor_node_names(self_successor)

    direct_shared = encoder.encode_batch(
        [state, state],
        successors=succ0,
        goals=goals,
    )
    assert direct_shared.num_graphs == 2
    assert hetero_data_equal(direct_shared, repeated_shared)
    assert _successor_node_names(direct_shared) == _successor_node_names(
        repeated_shared
    )

    separate = encoder.encode_batch(
        [state, state],
        successors=BatchParam.separate([succ0, succ1]),
        goals=BatchParam.separate([goals, goals]),
        subgoal_layers=[None, None],
    )
    explicit_separate = encoder.encode_batch(
        [state, state],
        successors=[succ0, succ1],
        goals=[goals, goals],
        subgoal_layers=[None, None],
    )
    assert separate.num_graphs == 2
    assert hetero_data_equal(separate, explicit_separate)
    assert _successor_node_names(separate) == _successor_node_names(explicit_separate)
    assert _successor_node_names(separate) != _successor_node_names(repeated_shared)


def test_transition_batch_param_rejects_missing_or_invalid_successors(small_blocks):
    space, domain, problem = small_blocks
    state = problem.get_initial_state()
    (_action0, succ0), _ = _first_transitions(space, state, count=2)

    encoder = TransitionHGraphEncoder(domain)

    with pytest.raises(
        ValueError,
        match="successors must be provided for transition batch encoding",
    ):
        encoder.encode_batch([state], successors=BatchParam.none())

    with pytest.raises(
        TypeError,
        match="successors entry at index 1 has invalid type",
    ):
        encoder.encode_batch(
            [state, state],
            successors=BatchParam.separate([succ0, None]),
        )


def test_horizon_batch_param_supports_shared_and_separate_dags(small_blocks):
    space, domain, problem = small_blocks
    root = problem.get_initial_state()
    goals = _problem_goals(problem)
    (action0, succ0), _ = _distinct_changed_transitions(space, root, count=2)
    dag0 = _single_transition_dag(root, action0, succ0)

    encoder = HorizonEncoder(domain, ignore_actions=False)

    shared = encoder.encode_batch(
        [root, root],
        dags=BatchParam.shared(dag0),
        goals=[goals, goals],
    )
    repeated_shared = encoder.encode_batch(
        [root, root],
        dags=[dag0, dag0],
        goals=[goals, goals],
    )
    assert shared.num_graphs == 2
    assert hetero_data_equal(shared, repeated_shared)
    assert shared.get_field("target_indices").numel() == 2
    no_dags = encoder.encode_batch(
        [root, root],
        dags=BatchParam.none(),
        goals=[goals, goals],
    )
    assert no_dags.get_field("target_indices").numel() == 0
    assert shared.num_nodes > no_dags.num_nodes
    assert shared.num_edges > no_dags.num_edges

    separate = encoder.encode_batch(
        [root, root],
        dags=BatchParam.separate([dag0, None]),
        goals=[goals, goals],
    )
    assert separate.num_graphs == 2
    assert separate.get_field("target_indices").numel() == 1
    assert separate.get_field("target_indices_ptr").tolist() == [0, 1, 1]

    explicit_none = encoder.encode_batch(
        [root],
        dags=BatchParam.none(),
        goals=goals,
    )
    implicit_none = encoder.encode_batch([root], goals=goals)
    assert explicit_none.num_nodes == implicit_none.num_nodes
    assert explicit_none.num_edges == implicit_none.num_edges
    assert explicit_none.get_field("target_indices").numel() == 0


def test_horizon_batch_accepts_rustworkx_dags(small_blocks):
    pytest.importorskip("rustworkx")

    space, domain, problem = small_blocks
    root = problem.get_initial_state()
    goals = _problem_goals(problem)
    (action0, succ0), _ = _distinct_changed_transitions(space, root, count=2)
    dag0 = _single_transition_dag(root, action0, succ0)
    graph0 = _single_transition_pygraph(root, action0, succ0)

    encoder = HorizonEncoder(domain, ignore_actions=False)

    direct_shared = encoder.encode_batch(
        [root, root],
        dags=graph0,
        goals=[goals, goals],
    )
    repeated_shared = encoder.encode_batch(
        [root, root],
        dags=[dag0, dag0],
        goals=[goals, goals],
    )
    assert hetero_data_equal(direct_shared, repeated_shared)

    wrapped_shared = encoder.encode_batch(
        [root, root],
        dags=BatchParam.shared(graph0),
        goals=[goals, goals],
    )
    assert hetero_data_equal(wrapped_shared, repeated_shared)

    per_entry_graphs = encoder.encode_batch(
        [root, root],
        dags=[graph0, None],
        goals=[goals, goals],
    )
    per_entry_dags = encoder.encode_batch(
        [root, root],
        dags=[dag0, None],
        goals=[goals, goals],
    )
    assert hetero_data_equal(per_entry_graphs, per_entry_dags)

    wrapped_per_entry = encoder.encode_batch(
        [root, root],
        dags=BatchParam.separate([graph0, None]),
        goals=[goals, goals],
    )
    assert hetero_data_equal(wrapped_per_entry, per_entry_dags)

    generator_per_entry = encoder.encode_batch(
        [root, root],
        dags=(entry for entry in [graph0, None]),
        goals=[goals, goals],
    )
    assert hetero_data_equal(generator_per_entry, per_entry_dags)


def test_horizon_batch_rustworkx_dags_enforce_root_match(small_blocks):
    pytest.importorskip("rustworkx")

    space, domain, problem = small_blocks
    root = problem.get_initial_state()
    goals = _problem_goals(problem)
    (_action0, succ0), _ = _distinct_changed_transitions(space, root, count=2)

    encoder = HorizonEncoder(domain, ignore_actions=False)
    mismatched_graph = _single_transition_pygraph(succ0, _action0, root)

    with pytest.raises(ValueError, match="dag root must match root state"):
        encoder.encode_batch(
            [root, root],
            dags=mismatched_graph,
            goals=[goals, goals],
        )

    matching_graph = _single_transition_pygraph(root, _action0, succ0)
    with pytest.raises(ValueError, match="dag root must match root state"):
        encoder.encode_batch(
            [root, root],
            dags=[matching_graph, mismatched_graph],
            goals=[goals, goals],
        )
