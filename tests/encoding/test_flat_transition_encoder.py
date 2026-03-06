from __future__ import annotations

import pytest
from torch_geometric.data import Batch

from mifrost.encoders import (
    FlatTransitionEffectsEncoder,
    FlatTransitionEncoder,
)
from mifrost.encoders.flat_data import flat_relation_data_from_pyg

from .test_flat_horizon_encoder import _first_distinct_changed_transitions
from .test_flat_relation_encoder import _assert_flat_batch_equal
from .test_utils import predicate, predicate_arity, state_atoms

import mifrost


def test_flat_transition_encode_requires_successor(small_blocks):
    _space, domain, problem = small_blocks
    state = problem.get_initial_state()
    encoder = FlatTransitionEncoder(domain)

    with pytest.raises(
        ValueError, match="successor must be provided for transition encoding"
    ):
        encoder.encode(state)


def test_flat_transition_batch_requires_successors(small_blocks):
    _space, domain, problem = small_blocks
    state = problem.get_initial_state()
    encoder = FlatTransitionEncoder(domain)

    with pytest.raises(
        ValueError,
        match="successors must be provided for transition batch encoding",
    ):
        encoder.encode_batch([state])


def test_flat_transition_rejects_unsupported_payloads(small_blocks):
    space, domain, problem = small_blocks
    state = problem.get_initial_state()
    goals = list(problem.get_goal_condition().get_literals())
    transitions = _first_distinct_changed_transitions(space, state, count=1)
    action0, successor = transitions[0]
    encoder = FlatTransitionEncoder(domain)

    with pytest.raises(
        ValueError,
        match="Transition encoders do not support explicit action payloads",
    ):
        encoder.encode(
            state,
            successor=successor,
            actions=[action0],
        )

    if goals:
        with pytest.raises(
            ValueError,
            match="Transition encoders do not support history_subgoals payloads",
        ):
            encoder.encode(
                state,
                successor=successor,
                history_subgoals=[(-1, [goals[0]])],
                history_max_steps=2,
            )


def test_flat_transition_batch_rejects_unsupported_payloads(small_blocks):
    space, domain, problem = small_blocks
    state = problem.get_initial_state()
    goals = list(problem.get_goal_condition().get_literals())
    transitions = _first_distinct_changed_transitions(space, state, count=1)
    action0, successor = transitions[0]
    encoder = FlatTransitionEncoder(domain)

    with pytest.raises(
        ValueError,
        match="Transition batch encoding does not support explicit action payloads",
    ):
        encoder.encode_batch(
            [state],
            successors=[successor],
            actions=[action0],
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


def test_flat_transition_encoder_emits_state_target_metadata(small_blocks):
    space, domain, problem = small_blocks
    state = problem.get_initial_state()
    transitions = _first_distinct_changed_transitions(space, state, count=1)
    successor = transitions[0][1]
    encoder = FlatTransitionEncoder(domain)

    data = encoder.encode_pyg(
        state,
        successor=successor,
        goals=list(problem.get_goal_condition().get_literals()),
    )

    assert data.target_entity_groups == ["state"]
    assert data.target_groups == ["state"]
    assert data.target_entity_sizes.tolist() == [2]
    assert data.target_sizes.tolist() == [1]
    assert data.graph_target_entity_names(0) == ["target:0", "target:1"]
    assert data.graph_target_indices(0).tolist() == [1]
    assert data.graph_target_positions(0).tolist() == [
        data.graph_target_entity_indices(0)[1].item()
    ]


def test_flat_transition_batch_matches_from_data_list(small_blocks):
    space, domain, problem = small_blocks
    state = problem.get_initial_state()
    transitions = _first_distinct_changed_transitions(space, state, count=2)
    if len(transitions) < 2:
        pytest.skip("Fixture should yield at least 2 distinct changed transitions")

    successor0 = transitions[0][1]
    successor1 = transitions[1][1]
    encoder = FlatTransitionEncoder(domain)

    actual = encoder.encode_batch(
        [state, state],
        successors=[successor0, successor1],
    ).as_pyg(as_batch=True)
    expected = flat_relation_data_from_pyg(
        Batch.from_data_list(
            [
                encoder.encode_pyg(state, successor=successor0),
                encoder.encode_pyg(state, successor=successor1),
            ]
        )
    )

    _assert_flat_batch_equal(actual, expected)


def test_flat_transition_effects_encoder_emits_only_changed_successor_literals(
    small_blocks,
):
    space, domain, problem = small_blocks
    state = problem.get_initial_state()
    transitions = _first_distinct_changed_transitions(space, state, count=1)
    successor = transitions[0][1]
    encoder = FlatTransitionEffectsEncoder(domain)

    data = encoder.encode_pyg(
        state,
        successor=successor,
        goals=[],
    )

    current_atoms = {
        atom
        for atom in state_atoms(state, with_statics=False)
        if predicate_arity(atom) > 0
    }
    successor_atoms = {
        atom
        for atom in state_atoms(successor, with_statics=False)
        if predicate_arity(atom) > 0
    }
    formatter = mifrost.RelationFormatter
    expected_successor_relations = {
        formatter.format_predicate(predicate(atom), polarity=True)
        for atom in successor_atoms - current_atoms
    } | {
        formatter.format_predicate(predicate(atom), polarity=False)
        for atom in current_atoms - successor_atoms
    }

    successor_position = int(data.graph_target_positions(0)[0].item())
    actual_successor_relations = {
        relation_name
        for relation_name, instances in data.flattened_relations.items()
        if instances.numel() and (instances[:, 0] == successor_position).any().item()
    }

    assert actual_successor_relations == expected_successor_relations


def test_flat_transition_to_networkx_exposes_successor_target_metadata(small_blocks):
    space, domain, problem = small_blocks
    state = problem.get_initial_state()
    transitions = _first_distinct_changed_transitions(space, state, count=1)
    successor = transitions[0][1]
    encoder = FlatTransitionEncoder(domain)

    data = encoder.encode_pyg(
        state,
        successor=successor,
        goals=list(problem.get_goal_condition().get_literals()),
    )
    graph = encoder.to_networkx(data)
    attrs = graph.nodes["target:1"]

    assert attrs["target_group"] == "state"
    assert attrs["target_index"] == 1
    assert attrs["target_depth"] == 1
    assert attrs["target_candidate_id"] == 1
