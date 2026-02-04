from __future__ import annotations

from dataclasses import dataclass

import pytest

import mifrost
from mifrost.encoders import _parts_to_pyg, _split_goals
from tests.conftest import problem_setup
from tests.encoding.test_utils import adv_domain, adv_state
from tests.ground_truth.pyencoding_ref.horizon_hetero_encoder import (
    HorizonHGraphEncoder as RefHorizonEncoder,
)
from tests.parity_utils import canonical_graph


SMALL_CASES = [
    ("blocks", "probBLOCKS-4-0"),
    ("gripper", "gripper_b-5"),
    ("spanner", "medium"),
    ("delivery", "instance_2x2_p-2_0"),
]


def _collect_transitions(space, root, limit: int = 3):
    transitions = []
    for action, successor in space.get_forward_transitions(root):
        transitions.append((action, successor))
        if len(transitions) >= limit:
            break
    if not transitions:
        pytest.skip("Fixture does not provide forward transitions.")
    return transitions


def _goal_inputs(problem):
    goals = list(problem.get_goal_condition().get_literals())
    inputs, _ = _split_goals(goals, subgoal_layers=None)
    return goals, inputs


@pytest.mark.parametrize(
    ("domain", "problem"),
    SMALL_CASES,
    ids=[f"{domain}:{problem}" for domain, problem in SMALL_CASES],
)
@pytest.mark.parametrize("mode", ["full", "delta"])
def test_horizon_encoder_parity(domain: str, problem: str, mode: str):
    space, domain_obj, problem_obj = problem_setup(domain, problem)
    root = problem_obj.get_initial_state()
    transitions = _collect_transitions(space, root, limit=3)

    # Reference encoder expects a list of paths (each path is a sequence of states).
    ref_transitions = [[succ] for _action, succ in transitions]
    ref_encoder = RefHorizonEncoder(
        domain_obj,
        successor_mode=mode,
        ignore_actions=True,
        include_lgan_edges=False,
    )
    goals, goal_inputs = _goal_inputs(problem_obj)
    ref_data = ref_encoder.encode(root, goals=goals, transitions=ref_transitions)
    ref_graph = ref_encoder.to_networkx(ref_data)

    # C++ encoder
    config = mifrost.HorizonEncoderConfig()
    config.transition_mode = (
        mifrost.HorizonEncoderMode.Full
        if mode == "full"
        else mifrost.HorizonEncoderMode.Delta
    )
    config.ignore_actions = True
    config.include_lgan_edges = False
    encoder = mifrost.HorizonHGraphEncoderEngine(adv_domain(domain_obj), config)

    dag = mifrost.TransitionDAG(adv_state(root))
    for action, succ in transitions:
        dag.register_transition(adv_state(root), adv_state(succ), None)

    parts = encoder.encode(adv_state(root), dag, goal_inputs)
    cpp_data = _parts_to_pyg(parts, as_batch=False)
    cpp_graph = ref_encoder.to_networkx(cpp_data)

    assert canonical_graph(ref_graph) == canonical_graph(cpp_graph)
