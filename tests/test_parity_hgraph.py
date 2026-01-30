from __future__ import annotations

from pathlib import Path

import pymimir
import pytest
import torch

import mifrost

from tests.ground_truth.hgraph_encoder import HGraphEncoder


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
    domain, problem, state, _domain_path, _problem_path = _load_problem()

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
