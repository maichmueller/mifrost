from __future__ import annotations

from itertools import islice

import networkx as nx
import networkx.algorithms.isomorphism as iso
import pytest

import mifrost
from mifrost.encoders import HGraphEncoder, TransitionHGraphEncoder

from .test_utils import (
    adv_domain,
    adv_state,
    format_atom_with_suffix,
    goal_inputs_from_problem,
    object_names,
    encoding_dict_to_pyg,
    predicate,
    predicate_arity,
    state_atoms,
    to_named_networkx,
)


def _strip_successor_nodes(graph: nx.Graph, successor_suffix: str) -> nx.Graph:
    nodes = [
        node
        for node, data in graph.nodes(data=True)
        if not str(data.get("type", "")).endswith(successor_suffix)
    ]
    return graph.subgraph(nodes).copy()


def _assert_base_isomorphism(transition_graph, base_graph, successor_suffix):
    filtered = _strip_successor_nodes(transition_graph, successor_suffix)
    node_match = iso.categorical_node_match(["type"], [None])
    edge_match = iso.numerical_multiedge_match(["position"], [None])
    assert nx.is_isomorphic(
        filtered, base_graph, node_match=node_match, edge_match=edge_match
    )


def _encode_successor_graph(
    domain,
    current,
    successor,
    goals,
    *,
    mode,
    successor_suffix: str = "[suc]",
):
    config = mifrost.SuccessorEncoderConfig()
    config.successor_mode = mode
    config.successor_suffix = successor_suffix
    encoder = mifrost.SuccessorHGraphEncoderEngine(adv_domain(domain), config)
    encoding_dict = encoder.encode(adv_state(current), adv_state(successor), goals)
    return to_named_networkx(encoding_dict_to_pyg(encoding_dict))


def test_transition_encoder_preserves_state_structure(small_blocks):
    space, domain, problem = small_blocks
    base_encoder = HGraphEncoder(domain)

    state = problem.get_initial_state()
    transitions = list(space.get_forward_transitions(state))
    if not transitions:
        pytest.skip("Fixture does not provide forward transitions.")
    successor_state = transitions[0][1]
    goals = goal_inputs_from_problem(problem)

    transition_graph = _encode_successor_graph(
        domain,
        state,
        successor_state,
        goals,
        mode=mifrost.SuccessorEncoderMode.Full,
    )
    base_graph = to_named_networkx(base_encoder.encode(state))

    _assert_base_isomorphism(transition_graph, base_graph, "[suc]")


def test_transition_encoder_successor_predicates_single_successor(small_blocks):
    space, domain, problem = small_blocks
    state = problem.get_initial_state()
    transitions = list(space.get_forward_transitions(state))
    if not transitions:
        pytest.skip("Fixture does not provide forward transitions.")

    _, successor_state = transitions[0]
    goals = goal_inputs_from_problem(problem)
    successor_suffix = "[suc]"
    transition_graph = _encode_successor_graph(
        domain,
        state,
        successor_state,
        goals,
        mode=mifrost.SuccessorEncoderMode.Full,
        successor_suffix=successor_suffix,
    )

    successor_nodes = {
        node: data
        for node, data in transition_graph.nodes(data=True)
        if str(data.get("type", "")).endswith(successor_suffix)
    }
    assert successor_nodes, "Expected successor predicate nodes to be present"

    formatter = mifrost.RelationFormatter

    for atom in state_atoms(successor_state, with_statics=False):
        if predicate_arity(atom) == 0:
            continue
        node_name = format_atom_with_suffix(atom, successor_suffix)
        node_type = formatter.format_predicate(predicate(atom), suffix=successor_suffix)
        assert node_name in successor_nodes
        assert successor_nodes[node_name]["type"] == node_type, (
            f"Node {node_name} missing successor type"
        )

        for pos, obj_name in enumerate(object_names(atom)):
            obj_node = obj_name
            assert transition_graph.has_node(obj_node)
            edge_data = transition_graph.get_edge_data(obj_node, node_name)
            if edge_data is None:
                edge_data = transition_graph.get_edge_data(node_name, obj_node)
            assert edge_data is not None, (
                f"Missing edge between {obj_node} and {node_name}"
            )
            entries = edge_data.values() if isinstance(edge_data, dict) else [edge_data]
            assert any(entry.get("position") == pos for entry in entries)


def test_transition_encoder_multiple_states_and_successors(medium_blocks):
    space, domain, problem = medium_blocks
    base_encoder = HGraphEncoder(domain)
    goals = goal_inputs_from_problem(problem)

    states = space.get_states()[:5]
    successor_suffix = "[suc]"

    for state in states:
        successors = list(space.get_forward_transitions(state))
        if not successors:
            continue
        base_graph = to_named_networkx(base_encoder.encode(state))

        for _, target in successors:
            successor_state = target
            transition_graph = _encode_successor_graph(
                domain,
                state,
                successor_state,
                goals,
                mode=mifrost.SuccessorEncoderMode.Full,
                successor_suffix=successor_suffix,
            )
            _assert_base_isomorphism(transition_graph, base_graph, successor_suffix)

            successor_facts = list(state_atoms(successor_state, with_statics=False))
            encoded_successors = [
                node
                for node, data in transition_graph.nodes(data=True)
                if str(data.get("type", "")).endswith(successor_suffix)
            ]
            nullary_filter = lambda atom: predicate_arity(atom) > 0
            assert len(sorted(encoded_successors)) == len(
                sorted(
                    set(str(atom) for atom in filter(nullary_filter, successor_facts))
                )
            )

            for atom in successor_facts:
                node_name = format_atom_with_suffix(atom, successor_suffix)
                if predicate_arity(atom) == 0:
                    continue
                assert node_name in encoded_successors


def test_transition_encoder_roundtrip_pyg_networkx(small_blocks):
    space, domain, problem = small_blocks
    state = problem.get_initial_state()
    transitions = list(space.get_forward_transitions(state))
    if not transitions:
        pytest.skip("Fixture does not provide forward transitions.")
    _, successor = transitions[0]
    goals = goal_inputs_from_problem(problem)
    graph = _encode_successor_graph(
        domain,
        state,
        successor,
        goals,
        mode=mifrost.SuccessorEncoderMode.Full,
    )
    reconstructed = graph

    node_match = iso.categorical_node_match(["type"], [None])
    edge_match = iso.numerical_multiedge_match(["position"], [None])
    assert nx.is_isomorphic(
        graph,
        reconstructed,
        node_match=node_match,
        edge_match=edge_match,
    )


def test_transition_encoder_multiple_roundtrips(medium_blocks):
    space, domain, problem = medium_blocks
    goals = goal_inputs_from_problem(problem)

    states = space.get_states()[:3]

    for state in states:
        for _, target in islice(space.get_forward_transitions(state), 0, 2):
            reconstructed = _encode_successor_graph(
                domain,
                state,
                target,
                goals,
                mode=mifrost.SuccessorEncoderMode.Full,
            )

            node_match = iso.categorical_node_match(["type"], [None])
            edge_match = iso.numerical_multiedge_match(["position"], [None])
            assert nx.is_isomorphic(
                reconstructed,
                reconstructed,
                node_match=node_match,
                edge_match=edge_match,
            )


def test_transition_encoder_nullary_placeholder(small_blocks):
    space, domain, problem = small_blocks
    state = problem.get_initial_state()
    successor_state = state
    goals = goal_inputs_from_problem(problem)

    config = mifrost.SuccessorEncoderConfig()
    config.successor_mode = mifrost.SuccessorEncoderMode.Full
    config.add_nullary_predicates = True
    encoder = mifrost.SuccessorHGraphEncoderEngine(adv_domain(domain), config)

    encoding_dict = encoder.encode(adv_state(state), adv_state(successor_state), goals)
    data = encoding_dict_to_pyg(encoding_dict)
    graph = to_named_networkx(data)
    placeholder = config.nullary_object_name
    assert graph.has_node(placeholder)

    successor_nullary_atoms = [
        atom
        for atom in state_atoms(successor_state, with_statics=False)
        if predicate_arity(atom) == 0
    ]
    if not successor_nullary_atoms:
        pytest.skip("Fixture does not include nullary predicates.")

    for atom in successor_nullary_atoms:
        atom_node = format_atom_with_suffix(atom, "[suc]")
        assert graph.has_node(atom_node)
        edge_data = graph.get_edge_data(placeholder, atom_node)
        if edge_data is None:
            edge_data = graph.get_edge_data(atom_node, placeholder)
        assert edge_data is not None, f"No edge found for successor nullary atom {atom}"
        entries = edge_data.values() if isinstance(edge_data, dict) else [edge_data]
        assert all(entry.get("position") == 0 for entry in entries)

    assert placeholder in data.object_names
    placeholder_idx = data.object_names.index(placeholder)

    symbol_type_id = config.symbol_type_id
    for atom in successor_nullary_atoms:
        predicate_type = mifrost.RelationFormatter.format_predicate(
            predicate(atom), suffix="[suc]"
        )
        edge_type = (symbol_type_id, "0", predicate_type)
        assert edge_type in data.edge_types
        edge_index = data[edge_type].edge_index
        assert (edge_index[0] == placeholder_idx).all()


def test_transition_encode_batch_rejects_actions_and_history(small_blocks):
    space, domain, problem = small_blocks
    state = problem.get_initial_state()
    transitions = list(space.get_forward_transitions(state))
    if not transitions:
        pytest.skip("Fixture does not provide forward transitions.")

    action, successor = transitions[0]
    goals = list(problem.get_goal_condition().get_literals())
    encoder = TransitionHGraphEncoder(domain)
    with pytest.raises(
        ValueError,
        match="Transition batch encoding does not support explicit action payloads",
    ):
        encoder.encode_batch(
            [state],
            successors=[successor],
            actions=[action],
        )

    if goals:
        with pytest.raises(
            ValueError,
            match="Transition batch encoding does not support history_subgoals payloads",
        ):
            encoder.encode_batch(
                [state],
                successors=[successor],
                history_subgoals=[(-1, [goals[0]])],
                history_max_steps=2,
            )


def test_transition_encode_batch_accepts_empty_unsupported_payloads(small_blocks):
    space, domain, problem = small_blocks
    state = problem.get_initial_state()
    transitions = list(space.get_forward_transitions(state))
    if not transitions:
        pytest.skip("Fixture does not provide forward transitions.")

    _action, successor = transitions[0]
    encoder = TransitionHGraphEncoder(domain)
    encoding = encoder.encode_batch(
        [state],
        successors=[successor],
        actions=[],
        history_subgoals=[],
    )

    assert encoding.num_graphs == 1
