from __future__ import annotations

import networkx as nx
import pytest
import torch
from torch_geometric.data import Batch

import mifrost
from mifrost.encoders import (
    FlatRelationData,
    FlatRelationEncoder,
    _encoding_dict_to_pyg,
)


def _assert_flat_batch_equal(
    actual: FlatRelationData, expected: FlatRelationData
) -> None:
    assert type(actual).__name__ == type(expected).__name__
    assert actual.schema == expected.schema
    assert torch.equal(actual.x, expected.x)
    assert torch.equal(actual.node_sizes, expected.node_sizes)
    assert torch.equal(actual.object_sizes, expected.object_sizes)
    assert torch.equal(actual.object_indices, expected.object_indices)
    assert torch.equal(actual.relation_counts, expected.relation_counts)
    assert torch.equal(actual.relation_args, expected.relation_args)
    if hasattr(expected, "batch"):
        assert torch.equal(actual.batch, expected.batch)
    if hasattr(expected, "ptr"):
        assert torch.equal(actual.ptr, expected.ptr)
    assert getattr(actual, "node_names", None) == getattr(expected, "node_names", None)
    assert getattr(actual, "object_names", None) == getattr(
        expected, "object_names", None
    )


def _first_action(space, state):
    transitions = list(space.get_forward_transitions(state))
    actions = [action for action, _ in transitions if action is not None]
    if not actions:
        pytest.skip("Fixture does not provide applicable actions.")
    return actions[0]


def test_flat_relation_encoder_returns_flat_relation_data(small_blocks):
    _space, domain, problem = small_blocks
    encoder = FlatRelationEncoder(domain)
    data = encoder.encode_pyg(problem.get_initial_state())

    assert isinstance(data, FlatRelationData)
    assert data.schema.names == tuple(encoder.engine.relation_names)
    assert data.schema.arities == tuple(encoder.engine.relation_arities)
    assert data.relation_counts.shape == (1, len(data.schema.names))
    assert data.node_sizes.shape == (1,)
    assert data.object_sizes.shape == (1,)
    assert torch.equal(
        data.object_indices,
        torch.arange(data.object_sizes[0].item(), dtype=torch.long),
    )
    flattened = data.flattened_relations
    assert set(flattened.keys()) == set(data.schema.names)
    for relation_name, tensor in flattened.items():
        relation_id = data.schema.name_to_id[relation_name]
        assert tensor.shape[1] == data.schema.arities[relation_id]


def test_flat_relation_batch_matches_from_data_list(small_blocks):
    space, domain, problem = small_blocks
    encoder = FlatRelationEncoder(domain)
    states = [
        problem.get_initial_state(),
        space._advanced_state_space_sampler.sample_state_n_steps_from_goal(0),
    ]

    actual = encoder.encode_batch(states).as_pyg(as_batch=True)
    expected = Batch.from_data_list([encoder.encode_pyg(state) for state in states])

    _assert_flat_batch_equal(actual, expected)


def test_flat_relation_python_conversion_matches_native(small_blocks):
    space, domain, problem = small_blocks
    encoder = FlatRelationEncoder(domain)
    states = [
        problem.get_initial_state(),
        space._advanced_state_space_sampler.sample_state_n_steps_from_goal(0),
    ]

    encoding = encoder.encode_batch(states)
    actual = _encoding_dict_to_pyg(encoding.as_dict(), as_batch=True)
    expected = encoding.as_pyg(as_batch=True)

    _assert_flat_batch_equal(actual, expected)


def test_flat_relation_rejects_actions(small_blocks):
    space, domain, problem = small_blocks
    encoder = FlatRelationEncoder(domain)
    state = problem.get_initial_state()
    action = _first_action(space, state)

    with pytest.raises(ValueError, match="does not support explicit action payloads"):
        encoder.encode(state, actions=[action])

    with pytest.raises(ValueError, match="does not support explicit action payloads"):
        encoder.encode_batch([state], actions=[[action]])


def test_flat_relation_visualization_is_reconstructable(small_blocks):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")

    _space, domain, problem = small_blocks
    encoder = FlatRelationEncoder(domain)
    data = encoder.encode_pyg(problem.get_initial_state())
    graph = encoder.to_networkx(data)

    assert isinstance(graph, nx.MultiDiGraph)
    relation_nodes = [
        node
        for node, attrs in graph.nodes(data=True)
        if attrs.get("kind") == "relation"
    ]
    assert len(relation_nodes) == int(data.relation_counts.sum().item())

    expected_edges = sum(
        tensor.shape[0] * tensor.shape[1]
        for tensor in data.flattened_relations_view(graph_index=0).values()
    )
    assert graph.number_of_edges() == expected_edges

    ax = encoder.draw(data)
    assert ax is not None
