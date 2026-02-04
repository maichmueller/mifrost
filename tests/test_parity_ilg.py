from __future__ import annotations

import pytest

import mifrost
from tests.conftest import load_problem
from tests.ground_truth.pyencoding_ref.ilg_hetero_encoder import (
    ILGHGraphEncoder as RefILGEncoder,
)
from tests.parity_utils import canonical_graph


SMALL_CASES = [
    ("blocks", "probBLOCKS-4-0"),
    ("gripper", "gripper_b-5"),
    ("spanner", "medium"),
    ("delivery", "instance_2x2_p-2_0"),
]


@pytest.mark.parametrize(
    ("domain", "problem"),
    SMALL_CASES,
    ids=[f"{domain}:{problem}" for domain, problem in SMALL_CASES],
)
def test_ilg_encoder_parity(domain: str, problem: str) -> None:
    domain_obj, problem_obj, state, _domain_path, _problem_path = load_problem(
        domain, problem
    )
    goals = list(problem_obj.get_goal_condition().get_literals())

    ref_encoder = RefILGEncoder(domain_obj, include_lgan_edges=False)
    ref_data = ref_encoder.encode(state, goals=goals)
    ref_graph = ref_encoder.to_networkx(ref_data)

    cpp_encoder = mifrost.ILGEncoder(domain_obj, include_lgan_edges=False)
    cpp_data = cpp_encoder.encode(state, goals=goals)
    cpp_graph = ref_encoder.to_networkx(cpp_data)

    assert canonical_graph(ref_graph) == canonical_graph(cpp_graph)
