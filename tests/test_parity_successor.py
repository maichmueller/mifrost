from __future__ import annotations

import pytest

import mifrost
from mifrost.encoders import _parts_to_pyg, _split_goals
from tests.conftest import problem_setup
from tests.encoding.test_utils import adv_domain, adv_state
from tests.ground_truth.pyencoding_ref.transition_hetero_encoder import (
    TransitionEffectsHGraphEncoder as RefDeltaEncoder,
    TransitionHGraphEncoder as RefFullEncoder,
)
from tests.parity_utils import canonical_graph


SMALL_CASES = [
    ("blocks", "probBLOCKS-4-0"),
    ("gripper", "gripper_b-5"),
    ("spanner", "medium"),
    ("delivery", "instance_2x2_p-2_0"),
]


def _first_successor(space, root):
    for action, successor in space.get_forward_transitions(root):
        return action, successor
    pytest.skip("Fixture does not provide forward transitions.")


def _goal_inputs(problem):
    goals = list(problem.get_goal_condition().get_literals())
    inputs, _ = _split_goals(goals, subgoal_layers=None)
    return goals, inputs


@pytest.mark.parametrize(
    ("domain", "problem"),
    SMALL_CASES,
    ids=[f"{domain}:{problem}" for domain, problem in SMALL_CASES],
)
def test_successor_encoder_parity_full(domain: str, problem: str):
    space, domain_obj, problem_obj = problem_setup(domain, problem)
    root = problem_obj.get_initial_state()
    action, successor = _first_successor(space, root)

    goals, goal_inputs = _goal_inputs(problem_obj)

    ref_encoder = RefFullEncoder(domain_obj, include_lgan_edges=False)
    ref_data = ref_encoder.encode(root, goals=goals, successor=successor)
    ref_graph = ref_encoder.to_networkx(ref_data)

    config = mifrost.SuccessorEncoderConfig()
    config.successor_mode = mifrost.SuccessorEncoderMode.Full
    config.successor_suffix = "[suc]"
    config.include_lgan_edges = False
    encoder = mifrost.SuccessorHGraphEncoderEngine(adv_domain(domain_obj), config)
    parts = encoder.encode(adv_state(root), adv_state(successor), goal_inputs)
    cpp_data = _parts_to_pyg(parts, as_batch=False)
    cpp_graph = ref_encoder.to_networkx(cpp_data)

    assert canonical_graph(ref_graph) == canonical_graph(cpp_graph)


@pytest.mark.parametrize(
    ("domain", "problem"),
    SMALL_CASES,
    ids=[f"{domain}:{problem}" for domain, problem in SMALL_CASES],
)
def test_successor_encoder_parity_delta(domain: str, problem: str):
    space, domain_obj, problem_obj = problem_setup(domain, problem)
    root = problem_obj.get_initial_state()
    action, successor = _first_successor(space, root)

    goals, goal_inputs = _goal_inputs(problem_obj)

    ref_encoder = RefDeltaEncoder(domain_obj, include_lgan_edges=False)
    ref_data = ref_encoder.encode(root, goals=goals, successor=successor)
    ref_graph = ref_encoder.to_networkx(ref_data)

    config = mifrost.SuccessorEncoderConfig()
    config.successor_mode = mifrost.SuccessorEncoderMode.Delta
    config.successor_suffix = ""
    config.include_lgan_edges = False
    encoder = mifrost.SuccessorHGraphEncoderEngine(adv_domain(domain_obj), config)
    parts = encoder.encode(adv_state(root), adv_state(successor), goal_inputs)
    cpp_data = _parts_to_pyg(parts, as_batch=False)
    cpp_graph = ref_encoder.to_networkx(cpp_data)

    assert canonical_graph(ref_graph) == canonical_graph(cpp_graph)
