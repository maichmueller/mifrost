from __future__ import annotations

from collections import defaultdict

from torch_geometric.data import HeteroData

import mifrost
from mifrost.encoders import HGraphEncoder

from .test_utils import predicate_arity, state_atoms, to_named_networkx


def validate_hetero_data(
    data: HeteroData, relation_dict: dict[str, int], symbol_type_id: str
):
    assert symbol_type_id in data.node_types
    x_dict = data.x_dict
    edge_index_dict = data.edge_index_dict

    for node_type in data.node_types:
        if node_type == symbol_type_id:
            continue
        if node_type not in relation_dict:
            continue
        arity = relation_dict[node_type]

        allowed_atom_indices = set(range(x_dict[node_type].shape[0]))
        incoming_edges_by_atom = defaultdict(int)
        outgoing_edges_by_atom = defaultdict(int)
        for pos in range(arity):
            fwd_key = (symbol_type_id, str(pos), node_type)
            rev_key = (node_type, str(pos), symbol_type_id)
            if fwd_key not in edge_index_dict or rev_key not in edge_index_dict:
                continue
            dest_indices = edge_index_dict[fwd_key][1]
            for dst_index in dest_indices:
                incoming_edges_by_atom[dst_index.item()] += 1
                assert dst_index.item() in allowed_atom_indices

            source_indices = edge_index_dict[rev_key][0]
            for src_index in source_indices:
                outgoing_edges_by_atom[src_index.item()] += 1
                assert src_index.item() in allowed_atom_indices

        incoming_edges_as_expected = all(
            incoming_edges_by_atom[i] == arity for i in allowed_atom_indices
        )
        assert incoming_edges_as_expected, (
            f"{node_type=}, {incoming_edges_by_atom=}, {allowed_atom_indices=}, {arity=}"
        )
        assert all(outgoing_edges_by_atom[i] == arity for i in allowed_atom_indices)


def test_hetero_data(small_blocks):
    space, domain, problem = small_blocks
    encoder = HGraphEncoder(domain)
    symbol_type_id = mifrost.DEFAULT_SYMBOL_TYPE_ID
    # This test validates positional arity constraints for base predicate node
    # types only. Additional relation types (goal/sat/action/horizon links) are
    # covered by dedicated encoder tests.
    base_predicate_arities = {
        pred.get_name(): pred.get_arity() for pred in domain.get_predicates()
    }

    for state in space.get_states()[:5]:
        data = encoder.encode(state)
        validate_hetero_data(data, base_predicate_arities, symbol_type_id)

        goals = list(problem.get_goal_condition().get_literals())
        direct = encoder.encode(state, goals=goals)
        validate_hetero_data(direct, base_predicate_arities, symbol_type_id)


def test_nullary_predicates_connect_to_placeholder(small_blocks):
    space, domain, problem = small_blocks
    symbol_type_id = mifrost.DEFAULT_SYMBOL_TYPE_ID
    nullary_object_name = "![nullary_symbol]!"
    encoder = HGraphEncoder(
        domain,
        add_nullary_predicates=True,
        nullary_object_name=nullary_object_name,
        symbol_type_id=symbol_type_id,
    )
    state = problem.get_initial_state()

    pyg_data = encoder.encode(state)
    graph = to_named_networkx(pyg_data)
    placeholder = nullary_object_name
    assert graph.has_node(placeholder)

    nullary_atoms = [
        atom
        for atom in state_atoms(state, with_statics=False)
        if predicate_arity(atom) == 0
    ]
    if not nullary_atoms:
        import pytest

        pytest.skip("Fixture does not include nullary predicates.")

    for atom in nullary_atoms:
        atom_node = str(atom)
        assert graph.has_node(atom_node)
        edge_data = graph.get_edge_data(placeholder, atom_node)
        if edge_data is None:
            edge_data = graph.get_edge_data(atom_node, placeholder)
        assert edge_data is not None, f"No edge found for nullary atom {atom}"
        entries = edge_data.values() if isinstance(edge_data, dict) else [edge_data]
        assert all(entry.get("position") == 0 for entry in entries)

    assert placeholder in pyg_data.object_names
    placeholder_idx = pyg_data.object_names.index(placeholder)

    for atom in nullary_atoms:
        pred = atom.get_predicate()
        predicate_type = mifrost.RelationFormatter.format_predicate(
            getattr(pred, "_advanced_predicate", pred)
        )
        edge_type = (symbol_type_id, "0", predicate_type)
        assert edge_type in pyg_data.edge_types, (
            f"Edge type {edge_type} missing. Available: {pyg_data.edge_types}"
        )
        edge_index = pyg_data[edge_type].edge_index
        found = (edge_index[0] == placeholder_idx).any()
        assert found, (
            f"Edge from placeholder {placeholder} (idx {placeholder_idx}) to nullary atom {atom} missing."
        )


def test_consistent_object_node_to_names(small_blocks, medium_blocks):
    space, domain, problem = small_blocks
    space2, domain2, problem2 = medium_blocks
    encoder = HGraphEncoder(domain)
    initial = problem.get_initial_state()
    initial_pyg = encoder.encode(initial)
    successors = [
        encoder.encode(target)
        for action, target in space.get_forward_transitions(initial)
    ]
    initial2 = problem2.get_initial_state()
    initial_pyg2 = encoder.encode(initial2)
    successors2 = [
        encoder.encode(target)
        for action, target in space2.get_forward_transitions(initial2)
    ]
    if not successors or not successors2:
        import pytest

        pytest.skip("Fixture does not provide successors for object-name checks.")
    assert all(
        initial_pyg.object_names == successor.object_names for successor in successors
    )
    assert all(
        initial_pyg2.object_names == successor2.object_names
        for successor2 in successors2
    )
    assert initial_pyg.object_names != initial_pyg2.object_names and any(
        successor.object_names != successor2.object_names
        for successor in successors
        for successor2 in successors2
    )
