from __future__ import annotations

import networkx as nx
import networkx.algorithms.isomorphism as iso

import mifrost

from .test_utils import (
    adv_domain,
    adv_state,
    goal_inputs_from_problem,
    object_names,
    parts_to_pyg,
    predicate_arity,
    predicate_name,
    state_atoms,
    to_named_networkx,
)


def _encode_delta_graph(domain, current, successor, goals, suffix="[suc]"):
    config = mifrost.SuccessorEncoderConfig()
    config.successor_mode = mifrost.SuccessorEncoderMode.Delta
    config.successor_suffix = suffix
    config.add_nullary_predicates = False
    encoder = mifrost.SuccessorHGraphEncoderEngine(adv_domain(domain), config)
    parts = encoder.encode(adv_state(current), adv_state(successor), goals)
    data = parts_to_pyg(parts)
    return to_named_networkx(data), data


def _diff_atoms(state, successor):
    base_atoms = set(state_atoms(state, with_statics=False))
    successor_atoms = set(state_atoms(successor, with_statics=False))
    added = successor_atoms - base_atoms
    removed = base_atoms - successor_atoms
    return added, removed


def _assert_edges_for_atom(graph, atom_node, atom):
    for pos, obj_name in enumerate(object_names(atom)):
        edge_data = graph.get_edge_data(obj_name, atom_node)
        if edge_data is None:
            edge_data = graph.get_edge_data(atom_node, obj_name)
        assert edge_data is not None, f"Missing edge between {atom_node} and {obj_name}"
        entries = edge_data.values() if isinstance(edge_data, dict) else [edge_data]
        assert any(entry.get("position") == pos for entry in entries)


def test_transition_change_encoder_marks_added_and_removed_atoms(small_blocks):
    space, domain, problem = small_blocks

    state = problem.get_initial_state()
    _, successor = next(iter(space.get_forward_transitions(state)))

    added, removed = _diff_atoms(state, successor)
    assert added or removed, "Fixture should produce at least one changed atom"

    goals = goal_inputs_from_problem(problem)
    graph, _ = _encode_delta_graph(domain, state, successor, goals)

    formatter = mifrost.RelationFormatter
    addition_nodes = set()
    deletion_nodes = set()

    for atom in added:
        if predicate_arity(atom) == 0:
            continue
        node_name = f"{atom}[suc]"
        node_type = formatter.format_predicate(
            predicate_name(atom), polarity=True, suffix="[suc]"
        )
        assert node_name in graph.nodes, f"Added atom node {node_name} missing"
        assert graph.nodes[node_name]["type"] == node_type, (
            f"Added atom node type {graph.nodes[node_name]['type']} != {node_type}"
        )
        _assert_edges_for_atom(graph, node_name, atom)
        addition_nodes.add(node_name)

    for atom in removed:
        if predicate_arity(atom) == 0:
            continue
        node_name = f"{atom}[suc]"
        node_type = formatter.format_predicate(
            predicate_name(atom), polarity=False, suffix="[suc]"
        )
        assert node_name in graph.nodes, f"Removed atom node {node_name} missing"
        assert graph.nodes[node_name]["type"] == node_type, (
            f"Removed atom node type {graph.nodes[node_name]['type']} != {node_type}"
        )
        _assert_edges_for_atom(graph, node_name, atom)
        deletion_nodes.add(node_name)

    literal_nodes = {
        node
        for node, data in graph.nodes(data=True)
        if str(data.get("type", "")).endswith("[suc]")
        and (data["type"].startswith("[+]") or data["type"].startswith("[-]"))
    }
    expected_literal_nodes = addition_nodes | deletion_nodes
    assert literal_nodes == expected_literal_nodes, (
        f"Unexpected literal nodes encoded: {literal_nodes ^ expected_literal_nodes}"
    )


def test_transition_change_encoder_no_diff_for_identical_states(small_blocks):
    space, domain, problem = small_blocks
    state = problem.get_initial_state()
    goals = goal_inputs_from_problem(problem)

    graph, _ = _encode_delta_graph(domain, state, state, goals)

    for node, data in graph.nodes(data=True):
        ntype = data["type"]
        assert not (ntype.startswith("[+]") or ntype.startswith("[-]")), (
            f"Encountered unexpected literal node {node} while encoding identical states"
        )


def test_transition_change_encoder_nullary_placeholder(small_blocks):
    space, domain, problem = small_blocks
    state = problem.get_initial_state()
    _, successor = next(iter(space.get_forward_transitions(state)))

    added, removed = _diff_atoms(state, successor)
    nullary_added = [atom for atom in added if predicate_arity(atom) == 0]
    nullary_removed = [atom for atom in removed if predicate_arity(atom) == 0]
    assert nullary_added or nullary_removed, (
        "Expected a nullary predicate change for this transition"
    )

    goals = goal_inputs_from_problem(problem)
    config = mifrost.SuccessorEncoderConfig()
    config.successor_mode = mifrost.SuccessorEncoderMode.Delta
    config.successor_suffix = "[suc]"
    config.add_nullary_predicates = True
    encoder = mifrost.SuccessorHGraphEncoderEngine(adv_domain(domain), config)
    parts = encoder.encode(adv_state(state), adv_state(successor), goals)
    data = parts_to_pyg(parts)
    graph = to_named_networkx(data)
    placeholder = config.nullary_object_name
    assert graph.has_node(placeholder), "Placeholder object node missing"

    for atom in nullary_added + nullary_removed:
        node_name = f"{atom}[suc]"
        edge_data = graph.get_edge_data(placeholder, node_name)
        if edge_data is None:
            edge_data = graph.get_edge_data(node_name, placeholder)
        assert edge_data is not None, f"No placeholder edge for nullary atom {atom}"
        entries = edge_data.values() if isinstance(edge_data, dict) else [edge_data]
        assert all(entry.get("position") == 0 for entry in entries)


def test_transition_change_encoder_roundtrip_pyg_networkx(small_blocks):
    space, domain, problem = small_blocks
    state = problem.get_initial_state()
    _, successor = next(iter(space.get_forward_transitions(state)))
    goals = goal_inputs_from_problem(problem)

    graph, _ = _encode_delta_graph(domain, state, successor, goals)
    reconstructed = graph

    node_match = iso.categorical_node_match(["type"], [None])
    edge_match = iso.numerical_multiedge_match(["position"], [None])
    assert nx.is_isomorphic(
        graph, reconstructed, node_match=node_match, edge_match=edge_match
    )
