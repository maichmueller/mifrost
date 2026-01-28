from __future__ import annotations

from pathlib import Path

import pymimir
import pytest
import torch

import mifrost

from tests.ground_truth.hgraph_encoder import HGraphEncoder


def _load_blocks_problem() -> tuple[
    pymimir.Domain, pymimir.Problem, pymimir.State, Path, Path
]:
    root = Path(__file__).resolve().parents[1]
    domain_path = root / "data" / "pddl" / "blocks" / "domain.pddl"
    problem_path = root / "data" / "pddl" / "blocks" / "probBLOCKS-4-0.pddl"

    domain = pymimir.Domain(domain_path)
    problem = pymimir.Problem(domain, problem_path, mode="grounded")
    state = problem.get_initial_state()
    return domain, problem, state, domain_path, problem_path


def _make_test_actions(problem: pymimir.Problem) -> list[pymimir.GroundAction]:
    domain = problem.get_domain()
    objects = list(problem.get_objects())
    objects = sorted(objects, key=lambda obj: obj.get_index())
    if not objects:
        return []

    actions = []
    for action in domain.get_actions():
        arity = action.get_arity()
        if arity == 0:
            actions.append(problem.new_ground_action(action, []))
            continue
        chosen = [objects[i % len(objects)] for i in range(arity)]
        actions.append(problem.new_ground_action(action, chosen))
    return actions


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
def test_hgraph_parity_blocks_inputs(
    include_goals: bool, include_actions: bool, include_subgoals: bool
):
    domain, problem, state, _domain_path, _problem_path = _load_blocks_problem()

    goals = list(problem.get_goal_condition().get_literals())
    actions = _make_test_actions(problem) if include_actions else []
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
