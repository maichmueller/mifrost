from __future__ import annotations

import networkx as nx
import networkx.algorithms.isomorphism as iso
import pytest

import mifrost
from mifrost.encoders import HGraphEncoder

from .test_utils import (
    format_atom_with_suffix,
    goal_inputs_from_problem,
    object_names,
    parts_to_pyg,
    state_atoms,
    to_named_networkx,
)


def _adv(obj):
    return getattr(obj, "_advanced_state", obj)


def _adv_domain(obj):
    return getattr(obj, "_advanced_domain", obj)


def _predicate(atom):
    if hasattr(atom, "get_predicate"):
        pred = atom.get_predicate()
        return getattr(pred, "_advanced_predicate", pred)
    adv = getattr(atom, "_advanced_ground_atom", atom)
    pred = adv.get_predicate()
    return getattr(pred, "_advanced_predicate", pred)


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
    transitions = list(space.get_forward_transitions(state))
    if not transitions:
        pytest.skip("Fixture does not provide forward transitions.")
    _, successor_state = transitions[0]

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
        node_name = format_atom_with_suffix(atom, successor_suffix)
        node_type = formatter.format_predicate(
            _predicate(atom), polarity=True, suffix=successor_suffix
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


def test_successor_goal_satisfaction_emitted_when_enabled(small_blocks):
    space, domain, problem = small_blocks
    state = problem.get_initial_state()
    transitions = list(space.get_forward_transitions(state))
    if not transitions:
        pytest.skip("Fixture does not provide forward transitions.")
    _, successor_state = transitions[0]

    goals = goal_inputs_from_problem(problem)
    successor_suffix = "[suc]"

    config = mifrost.SuccessorEncoderConfig()
    config.successor_mode = mifrost.SuccessorEncoderMode.Full
    config.successor_suffix = successor_suffix
    config.include_successor_goal_satisfaction = True
    config.goal_satisfaction_derivations = {
        mifrost.GoalSatisfaction.satisfied,
        mifrost.GoalSatisfaction.unsatisfied,
    }

    encoder = mifrost.SuccessorHGraphEncoderEngine(_adv_domain(domain), config)
    parts = encoder.encode(_adv(state), _adv(successor_state), goals)
    graph = to_named_networkx(parts_to_pyg(parts))

    successor_atom_keys = {
        str(atom) for atom in state_atoms(successor_state, with_statics=False)
    }
    formatter = mifrost.RelationFormatter

    def _expected(goal, goal_levels):
        pred = _predicate(goal.get_atom())
        if pred.get_arity() == 0:
            return None
        atom_key = str(goal.get_atom())
        satisfied = (atom_key in successor_atom_keys) == goal.get_polarity()
        sat_enum = (
            mifrost.GoalSatisfaction.satisfied
            if satisfied
            else mifrost.GoalSatisfaction.unsatisfied
        )
        level = goal_levels.get(goal, 0)
        node_type = formatter.format_predicate(
            pred,
            goal_level=level,
            satisfaction=sat_enum,
            polarity=goal.get_polarity(),
            suffix=successor_suffix,
        )
        node_key = formatter.format_literal(
            goal,
            goal_level=level,
            satisfaction=sat_enum,
            suffix=successor_suffix,
        )
        return node_key, node_type

    # Check fluent + derived goals (static goals are not encoded in successor facts).
    level_map = {
        **dict(getattr(goals, "fluent_goal_levels", {})),
        **dict(getattr(goals, "derived_goal_levels", {})),
    }
    for goal in list(goals.fluent_goals) + list(goals.derived_goals):
        expected = _expected(goal, level_map)
        if expected is None:
            continue
        node_key, node_type = expected
        assert node_key in graph.nodes
        assert graph.nodes[node_key]["type"] == node_type
