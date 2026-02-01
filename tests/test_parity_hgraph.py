from __future__ import annotations

from pathlib import Path

import pymimir
import pytest
import torch
from torch_geometric.data import Batch

import mifrost
from tests.conftest import load_problem, problem_setup

from tests.ground_truth.hgraph_encoder import HGraphEncoder

SMALL_PARITY_CASES = [
    ("blocks", "probBLOCKS-4-0"),
    ("gripper", "gripper_b-5"),
    ("delivery", "instance_2x2_p-2_0"),
]


def _maybe_subgoal_layers(
    goals: list[pymimir.GroundLiteral],
    include_subgoals: bool,
) -> list[list[pymimir.GroundLiteral]] | None:
    if not include_subgoals or not goals:
        return None
    return [goals[:1]]


def _compare_hetero(py_data, cpp_data) -> None:
    assert set(py_data.node_types) == set(cpp_data.node_types)
    for node_type in py_data.node_types:
        assert torch.equal(py_data[node_type].x, cpp_data[node_type].x)
        assert list(py_data[node_type].node_names) == list(
            cpp_data[node_type].node_names
        )

    assert list(py_data.object_names) == list(cpp_data.object_names)

    assert set(py_data.edge_types) == set(cpp_data.edge_types)
    for edge_type in py_data.edge_types:
        assert torch.equal(
            py_data[edge_type].edge_index, cpp_data[edge_type].edge_index
        )


@pytest.mark.parametrize("include_goals", [False, True])
@pytest.mark.parametrize("include_actions", [False, True])
@pytest.mark.parametrize("include_subgoals", [False, True])
@pytest.mark.parametrize(
    ("domain", "problem"),
    [
        ["blocks", "probBLOCKS-4-0"],
        ["blocks_eq", "medium"],
        ["delivery", "instance_4x4_p-2_0"],
        ["gripper", "gripper_b-5"],
        ["reward", "instance_5x5_0"],
        ["spanner", "medium"],
    ],
)
def test_hgraph_parity_blocks_inputs(
    include_goals: bool,
    include_actions: bool,
    include_subgoals: bool,
    domain: str,
    problem: str,
):
    domain, problem, state, _domain_path, _problem_path = load_problem(domain, problem)

    goals = list(problem.get_goal_condition().get_literals())
    actions = state.generate_applicable_actions()
    subgoal_layers = _maybe_subgoal_layers(goals, include_subgoals)

    py_encoder = HGraphEncoder(
        domain,
        ignore_actions=not include_actions,
        max_goal_level=1,
    )
    py_data = py_encoder.encode_state(
        state,
        goals=goals if include_goals else None,
        actions=actions if include_actions else None,
        subgoal_layers=subgoal_layers,
    )

    cpp_encoder = mifrost.HGraphEncoder(
        domain,
        ignore_actions=not include_actions,
        max_goal_level=1,
    )
    cpp_data = cpp_encoder.encode(
        state,
        goals=goals if include_goals else None,
        actions=actions if include_actions else None,
        subgoal_layers=subgoal_layers,
    )

    _compare_hetero(py_data, cpp_data)


@pytest.mark.parametrize("include_static", [False, True])
@pytest.mark.parametrize("include_lgan_edges", [False, True])
@pytest.mark.parametrize("add_nullary_predicates", [False, True])
@pytest.mark.parametrize("support_literals", [False, True])
@pytest.mark.parametrize(
    ("domain", "problem"),
    SMALL_PARITY_CASES,
    ids=[f"{domain}:{problem}" for domain, problem in SMALL_PARITY_CASES],
)
def test_hgraph_parity_flag_variants(
    include_static: bool,
    include_lgan_edges: bool,
    add_nullary_predicates: bool,
    support_literals: bool,
    domain: str,
    problem: str,
):
    domain, problem, state, _domain_path, _problem_path = load_problem(domain, problem)

    goals = list(problem.get_goal_condition().get_literals())
    subgoal_layers = _maybe_subgoal_layers(goals, include_subgoals=True)

    py_encoder = HGraphEncoder(
        domain,
        ignore_actions=True,
        include_static=include_static,
        include_lgan_edges=include_lgan_edges,
        add_nullary_predicates=add_nullary_predicates,
        support_literals=support_literals,
        max_goal_level=1,
    )
    py_data = py_encoder.encode_state(
        state,
        goals=goals,
        actions=None,
        subgoal_layers=subgoal_layers,
    )

    cpp_encoder = mifrost.HGraphEncoder(
        domain,
        ignore_actions=True,
        include_static=include_static,
        include_lgan_edges=include_lgan_edges,
        add_nullary_predicates=add_nullary_predicates,
        support_literals=support_literals,
        max_goal_level=1,
    )
    cpp_data = cpp_encoder.encode(
        state,
        goals=goals,
        actions=None,
        subgoal_layers=subgoal_layers,
    )

    _compare_hetero(py_data, cpp_data)


@pytest.mark.parametrize("include_goals", [False, True])
@pytest.mark.parametrize("include_actions", [False, True])
@pytest.mark.parametrize("include_subgoals", [False, True])
@pytest.mark.parametrize(
    ("domain", "problem"),
    SMALL_PARITY_CASES,
    ids=[f"{domain}:{problem}" for domain, problem in SMALL_PARITY_CASES],
)
def test_hgraph_streaming_parity(
    include_goals: bool,
    include_actions: bool,
    include_subgoals: bool,
    domain: str,
    problem: str,
):
    space, domain, problem = problem_setup(domain, problem)

    root = problem.get_initial_state()
    transitions = list(space.get_forward_transitions(root))
    if not transitions:
        pytest.skip("Fixture does not provide a successor state for streaming parity.")
    _, successor = transitions[0]
    states = [root, successor]
    goals = list(problem.get_goal_condition().get_literals())
    subgoal_layers = _maybe_subgoal_layers(goals, include_subgoals)
    actions_per_state = [
        state.generate_applicable_actions() if include_actions else None
        for state in states
    ]

    py_encoder = HGraphEncoder(
        domain,
        ignore_actions=not include_actions,
        max_goal_level=1,
    )
    py_data_list = [
        py_encoder.encode_state(
            state,
            goals=goals if include_goals else None,
            actions=actions if include_actions else None,
            subgoal_layers=subgoal_layers,
        )
        for state, actions in zip(states, actions_per_state)
    ]
    expected_batch = Batch.from_data_list(py_data_list)

    cpp_encoder = mifrost.HGraphEncoder(
        domain,
        ignore_actions=not include_actions,
        max_goal_level=1,
    )
    stream = cpp_encoder.stream()
    for state, actions in zip(states, actions_per_state):
        stream.append(
            state,
            goals=goals if include_goals else None,
            actions=actions if include_actions else None,
            subgoal_layers=subgoal_layers,
        )
    actual_batch = stream.flush(as_batch=True)

    _compare_hetero(expected_batch, actual_batch)
