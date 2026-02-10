from __future__ import annotations

import pytest
import torch

import mifrost
from mifrost.encoders.accessors import (
    atom_objects,
    literal_atom,
    literal_polarity,
    object_name,
    predicate,
    predicate_name,
)
from tests.conftest import load_problem


SMALL_CASES = [
    ("blocks", "smedium"),
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
    cpp_encoder = mifrost.ILGEncoder(domain_obj, include_lgan_edges=False)
    cpp_data = cpp_encoder.encode(state, goals=goals).as_pyg()

    symbol_type_id = cpp_encoder.symbol_type_id
    assert symbol_type_id in cpp_data.node_types

    symbol_names = list(cpp_data[symbol_type_id].node_names)
    assert list(cpp_data.object_names) == symbol_names

    expected_objects = [object_name(obj) for obj in problem_obj.get_objects()] + [
        object_name(obj) for obj in problem_obj.get_domain().get_constants()
    ]
    for obj_name in expected_objects:
        assert obj_name in symbol_names

    facts = list(state.get_atoms())
    satisfied: set = set()
    missing_goal_facts: list = []
    for goal in goals:
        atom = literal_atom(goal)
        if any(atom == f for f in facts) == literal_polarity(goal):
            satisfied.add(atom)
        else:
            missing_goal_facts.append(atom)

    pred_atoms: dict[str, list] = {}
    for atom in facts + missing_goal_facts:
        pred_atoms.setdefault(predicate_name(predicate(atom)), []).append(atom)

    symbol_index = {name: idx for idx, name in enumerate(symbol_names)}

    for pred_name, atoms in pred_atoms.items():
        assert pred_name in cpp_data.node_types
        node_names = list(cpp_data[pred_name].node_names)
        node_index = {name: idx for idx, name in enumerate(node_names)}
        for atom in atoms:
            atom_key = str(atom)
            assert atom_key in node_index
            atom_idx = node_index[atom_key]
            terms = list(atom_objects(atom))
            if not terms:
                continue
            for pos, obj in enumerate(terms):
                obj_idx = symbol_index[object_name(obj)]
                edge_type = (symbol_type_id, str(pos), pred_name)
                rev_edge_type = (pred_name, str(pos), symbol_type_id)
                assert edge_type in cpp_data.edge_types
                assert rev_edge_type in cpp_data.edge_types
                src, dst = cpp_data[edge_type].edge_index
                assert ((src == obj_idx) & (dst == atom_idx)).any()
                rsrc, rdst = cpp_data[rev_edge_type].edge_index
                assert ((rsrc == atom_idx) & (rdst == obj_idx)).any()

    # Ensure encoded tensors exist and are consistent sizes.
    assert isinstance(cpp_data[symbol_type_id].x, torch.Tensor)
