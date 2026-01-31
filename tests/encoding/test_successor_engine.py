from __future__ import annotations

import networkx as nx
import networkx.algorithms.isomorphism as iso

import mifrost
from mifrost.encoders import HGraphEncoder

from .test_utils import (
    goal_inputs_from_problem,
    parts_to_pyg,
    state_atoms,
    to_named_networkx,
    object_names,
)


def _adv(obj):
    return getattr(obj, "_advanced_state", obj)


def _adv_domain(obj):
    return getattr(obj, "_advanced_domain", obj)


def _predicate(atom):
    return atom.get_predicate() if hasattr(atom, "get_predicate") else atom.predicate


def _predicate_name(atom) -> str:
    pred = _predicate(atom)
    return pred.get_name() if hasattr(pred, "get_name") else pred.name


def _arity(atom) -> int:
    pred = _predicate(atom)
    return pred.get_arity()


def _strip_successor_nodes(graph: nx.Graph, successor_suffix: str) -> nx.Graph:
    nodes = [
        node
        for node, data in graph.nodes(data=True)
        if not str(data.get("type", "")).endswith(successor_suffix)
    ]
    return graph.subgraph(nodes).copy()


def _encode_successor_graph(
    domain,
    current,
    successor,
    goals,
    mode,
    *,
    successor_suffix: str = "[suc]",
):
    config = mifrost.SuccessorEncoderConfig()
    config.successor_mode = mode
    config.successor_suffix = successor_suffix
    encoder = mifrost.SuccessorHGraphEncoderEngine(_adv_domain(domain), config)
    parts = encoder.encode(_adv(current), _adv(successor), goals)
    data = parts_to_pyg(parts)
    return to_named_networkx(data)


def test_successor_full_preserves_state_structure(small_blocks):
    space, domain, problem = small_blocks
    base_encoder = HGraphEncoder(domain)

    state = problem.get_initial_state()
    _, successor_state = next(space.get_forward_transitions(state))

    goals = goal_inputs_from_problem(problem)
    successor_suffix = "[suc]"
    transition_graph = _encode_successor_graph(
        domain,
        state,
        successor_state,
        goals,
        mifrost.SuccessorEncoderMode.Full,
        successor_suffix=successor_suffix,
    )
    base_graph = to_named_networkx(base_encoder.encode(state))

    filtered = _strip_successor_nodes(transition_graph, successor_suffix)
    node_match = iso.categorical_node_match(["type"], [None])
    edge_match = iso.numerical_multiedge_match(["position"], [None])
    assert nx.is_isomorphic(
        filtered, base_graph, node_match=node_match, edge_match=edge_match
    )


def test_successor_delta_marks_added_and_removed_atoms(small_blocks):
    space, domain, problem = small_blocks
    state = problem.get_initial_state()
    _, successor_state = next(space.get_forward_transitions(state))

    goals = goal_inputs_from_problem(problem)
    successor_suffix = "[suc]"
    graph = _encode_successor_graph(
        domain,
        state,
        successor_state,
        goals,
        mifrost.SuccessorEncoderMode.Delta,
        successor_suffix=successor_suffix,
    )

    base_atoms = set(state_atoms(state, with_statics=False))
    successor_atoms = set(state_atoms(successor_state, with_statics=False))
    added = successor_atoms - base_atoms
    removed = base_atoms - successor_atoms

    formatter = mifrost.RelationFormatter

    for atom in added:
        if _arity(atom) == 0:
            continue
        node_name = f"{atom}{successor_suffix}"
        node_type = formatter.format_predicate(
            _predicate_name(atom), polarity=True, suffix=successor_suffix
        )
        assert node_name in graph.nodes
        assert graph.nodes[node_name]["type"] == node_type
        for pos, obj_name in enumerate(object_names(atom)):
            edge_data = graph.get_edge_data(obj_name, node_name)
            if edge_data is None:
                edge_data = graph.get_edge_data(node_name, obj_name)
            assert edge_data is not None
            entries = edge_data.values() if isinstance(edge_data, dict) else [edge_data]
            assert any(entry.get("position") == pos for entry in entries)

    for atom in removed:
        if _arity(atom) == 0:
            continue
        node_name = f"{atom}{successor_suffix}"
        node_type = formatter.format_predicate(
            _predicate_name(atom), polarity=False, suffix=successor_suffix
        )
        assert node_name in graph.nodes
        assert graph.nodes[node_name]["type"] == node_type
        for pos, obj_name in enumerate(object_names(atom)):
            edge_data = graph.get_edge_data(obj_name, node_name)
            if edge_data is None:
                edge_data = graph.get_edge_data(node_name, obj_name)
            assert edge_data is not None
            entries = edge_data.values() if isinstance(edge_data, dict) else [edge_data]
            assert any(entry.get("position") == pos for entry in entries)
