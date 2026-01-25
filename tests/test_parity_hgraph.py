from __future__ import annotations

from pathlib import Path

import pymimir
import torch

import mifrost

from tests.ground_truth.hgraph_encoder import HGraphEncoder
from tests.ground_truth.mifrost_adapter import mifrost_dict_to_heterodata


def _load_blocks_problem() -> tuple[pymimir.Domain, pymimir.Problem, pymimir.State]:
    root = Path(__file__).resolve().parents[1]
    domain_path = root / "data" / "pddl" / "blocks" / "domain.pddl"
    problem_path = root / "data" / "pddl" / "blocks" / "probBLOCKS-4-0.pddl"

    domain = pymimir.Domain(domain_path)
    problem = pymimir.Problem(domain, problem_path, mode="grounded")
    state = problem.get_initial_state()
    return domain, problem, state


def _get_advanced(obj, attr: str):
    try:
        return getattr(obj, attr)
    except AttributeError as exc:
        raise AttributeError(f"{obj} missing {attr} for pymimir interop") from exc


def test_hgraph_parity_blocks_initial_state():
    domain, _problem, state = _load_blocks_problem()

    py_encoder = HGraphEncoder(domain)
    py_data = py_encoder.encode_state(state)

    adv_domain = _get_advanced(domain, "_advanced_domain")
    adv_state = _get_advanced(state, "_advanced_state")
    cpp_encoder = mifrost.HGraphStreamEncoder(adv_domain)
    cpp_payload = cpp_encoder.encode_state(adv_state)
    cpp_data = mifrost_dict_to_heterodata(cpp_payload)

    assert set(py_data.node_types) == set(cpp_data.node_types)
    for node_type in py_data.node_types:
        assert torch.equal(py_data[node_type].x, cpp_data[node_type].x)

    assert set(py_data.edge_types) == set(cpp_data.edge_types)
    for edge_type in py_data.edge_types:
        assert torch.equal(
            py_data[edge_type].edge_index, cpp_data[edge_type].edge_index
        )
