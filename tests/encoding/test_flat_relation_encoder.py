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
from mifrost.encoders.flat_data import flat_relation_data_from_pyg

from .test_utils import adv_action


def _assert_flat_batch_equal(
    actual: FlatRelationData, expected: FlatRelationData
) -> None:
    assert type(actual).__name__ == type(expected).__name__
    assert actual.schema == expected.schema
    assert torch.equal(actual.x, expected.x)
    assert torch.equal(actual.node_sizes, expected.node_sizes)
    assert torch.equal(actual.object_sizes, expected.object_sizes)
    assert torch.equal(actual.object_indices, expected.object_indices)
    actual_target_entity_sizes = getattr(actual, "target_entity_sizes", None)
    expected_target_entity_sizes = getattr(expected, "target_entity_sizes", None)
    if actual_target_entity_sizes is None or expected_target_entity_sizes is None:
        assert actual_target_entity_sizes is expected_target_entity_sizes
    else:
        assert torch.equal(actual_target_entity_sizes, expected_target_entity_sizes)
    actual_target_entity_indices = getattr(actual, "target_entity_indices", None)
    expected_target_entity_indices = getattr(expected, "target_entity_indices", None)
    if actual_target_entity_indices is None or expected_target_entity_indices is None:
        assert actual_target_entity_indices is expected_target_entity_indices
    else:
        assert torch.equal(
            actual_target_entity_indices,
            expected_target_entity_indices,
        )
    actual_target_sizes = getattr(actual, "target_sizes", None)
    expected_target_sizes = getattr(expected, "target_sizes", None)
    if actual_target_sizes is None or expected_target_sizes is None:
        assert actual_target_sizes is expected_target_sizes
    else:
        assert torch.equal(actual_target_sizes, expected_target_sizes)
    for field_name in (
        "target_positions",
        "target_indices",
        "target_candidate_ids",
        "target_group_ids",
    ):
        actual_field = getattr(actual, field_name, None)
        expected_field = getattr(expected, field_name, None)
        if actual_field is None or expected_field is None:
            assert actual_field is expected_field
        else:
            assert torch.equal(actual_field, expected_field)
    assert torch.equal(actual.relation_counts, expected.relation_counts)
    assert torch.equal(actual.relation_args, expected.relation_args)
    expected_batch = getattr(expected, "batch", None)
    actual_batch = getattr(actual, "batch", None)
    if torch.is_tensor(expected_batch):
        assert torch.equal(actual_batch, expected_batch)
    else:
        assert actual_batch is expected_batch
    expected_ptr = getattr(expected, "ptr", None)
    actual_ptr = getattr(actual, "ptr", None)
    if torch.is_tensor(expected_ptr):
        assert torch.equal(actual_ptr, expected_ptr)
    else:
        assert actual_ptr is expected_ptr
    assert getattr(actual, "node_names", None) == getattr(expected, "node_names", None)
    assert getattr(actual, "object_names", None) == getattr(
        expected, "object_names", None
    )
    assert getattr(actual, "target_names", None) == getattr(
        expected, "target_names", None
    )
    assert getattr(actual, "target_groups", None) == getattr(
        expected, "target_groups", None
    )
    assert getattr(actual, "target_symbol_prefix", None) == getattr(
        expected, "target_symbol_prefix", None
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
    expected = flat_relation_data_from_pyg(
        Batch.from_data_list([encoder.encode_pyg(state) for state in states])
    )

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


def test_flat_relation_batch_flattened_relations_group_by_relation(small_blocks):
    space, domain, problem = small_blocks
    encoder = FlatRelationEncoder(domain)
    states = [
        problem.get_initial_state(),
        space._advanced_state_space_sampler.sample_state_n_steps_from_goal(0),
    ]

    batch = encoder.encode_batch(states).as_pyg(as_batch=True)
    combined = batch.flattened_relations_view()
    per_graph = [
        batch.flattened_relations_view(graph_index=graph_index)
        for graph_index in range(batch.num_graphs)
    ]

    for relation_name in batch.schema.names:
        expected = torch.cat(
            [graph_view[relation_name] for graph_view in per_graph],
            dim=0,
        )
        assert torch.equal(combined[relation_name], expected)


def test_flat_relation_encoder_supports_explicit_actions_with_target_entities(
    small_blocks,
):
    space, domain, problem = small_blocks
    encoder = FlatRelationEncoder(domain)
    state = problem.get_initial_state()
    action = _first_action(space, state)
    adv = adv_action(action)
    formatter = mifrost.RelationFormatter

    data = encoder.encode_pyg(state, actions=[action])

    assert data.target_entity_sizes.tolist() == [1]
    assert data.target_entity_indices.tolist() == [int(data.object_sizes[0].item())]
    target_entity_index = int(data.target_entity_indices[0].item())
    assert data.graph_target_entity_names(0) == [formatter.format_action(adv)]

    flattened = data.flattened_relations
    action_schema = formatter.format_action_schema(adv.get_action())
    assert action_schema in flattened
    action_relation = flattened[action_schema]
    assert action_relation.shape == (1, len(adv.get_objects()) + 1)
    assert int(action_relation[0, 0].item()) == target_entity_index

    action_arg_names = [
        data.graph_node_names(0)[int(idx)] for idx in action_relation[0, 1:].tolist()
    ]
    assert action_arg_names == [obj.get_name() for obj in adv.get_objects()]


def test_flat_relation_batch_matches_from_data_list_with_actions(small_blocks):
    space, domain, problem = small_blocks
    encoder = FlatRelationEncoder(domain)
    state = problem.get_initial_state()
    action = _first_action(space, state)

    actual = encoder.encode_batch([state, state], actions=[[action], None]).as_pyg(
        as_batch=True
    )
    expected = flat_relation_data_from_pyg(
        Batch.from_data_list(
            [
                encoder.encode_pyg(state, actions=[action]),
                encoder.encode_pyg(state),
            ]
        )
    )

    _assert_flat_batch_equal(actual, expected)


def test_flat_relation_batch_accepts_generator_actions(small_blocks):
    space, domain, problem = small_blocks
    encoder = FlatRelationEncoder(domain)
    state = problem.get_initial_state()
    action = _first_action(space, state)

    generated = encoder.encode_batch(
        [state],
        actions=(item for item in [action]),
    ).as_pyg(as_batch=False)
    listed = encoder.encode_batch([state], actions=[action]).as_pyg(as_batch=False)

    _assert_flat_batch_equal(generated, listed)


def test_flat_relation_batch_accepts_per_state_none_action_entries(small_blocks):
    space, domain, problem = small_blocks
    encoder = FlatRelationEncoder(domain)
    state = problem.get_initial_state()
    action = _first_action(space, state)

    actual = encoder.encode_batch([state, state], actions=[None, [action]]).as_pyg(
        as_batch=True
    )
    expected = encoder.encode_batch([state, state], actions=[[], [action]]).as_pyg(
        as_batch=True
    )

    _assert_flat_batch_equal(actual, expected)


def test_flat_relation_action_target_metadata_enabled_for_action_source(
    small_blocks,
):
    space, domain, problem = small_blocks
    state = problem.get_initial_state()
    action = _first_action(space, state)
    adv = adv_action(action)
    formatter = mifrost.RelationFormatter

    encoder = FlatRelationEncoder(
        domain,
        target_sources=[mifrost.TargetSource.Actions],
    )
    encoding = encoder.encode_batch([state], actions=[[action]])
    data = encoding.as_pyg(as_batch=True)

    assert encoding.has_field("target_sizes")
    assert encoding.has_field("target_positions")
    assert encoding.has_field("target_indices")
    assert encoding.has_field("target_candidate_ids")
    assert encoding.has_field("target_group_ids")
    assert data.target_sizes.tolist() == [1]
    assert data.target_positions.tolist() == data.target_entity_indices.tolist()
    assert data.target_indices.tolist() == [0]
    assert data.target_candidate_ids.tolist() == [0]
    assert data.target_group_ids.tolist() == [0]
    assert (
        data.graph_target_positions(0).tolist() == data.target_entity_indices.tolist()
    )
    assert data.graph_target_names(0) == [formatter.format_action(adv)]
    assert list(data.target_groups) == ["action"]
    assert data.target_symbol_prefix == "target:"


def test_flat_relation_action_target_metadata_can_be_disabled(small_blocks):
    space, domain, problem = small_blocks
    state = problem.get_initial_state()
    action = _first_action(space, state)

    encoder = FlatRelationEncoder(domain)
    encoding = encoder.encode_batch([state], actions=[[action]])
    data = encoding.as_pyg(as_batch=True)

    assert not encoding.has_field("target_sizes")
    assert not encoding.has_field("target_positions")
    assert not encoding.has_field("target_indices")
    assert not hasattr(data, "target_positions")
    assert not hasattr(data, "target_indices")
    assert not hasattr(data, "target_names")


def test_flat_relation_action_target_metadata_preserves_duplicates_and_empty_graphs(
    small_blocks,
):
    space, domain, problem = small_blocks
    state = problem.get_initial_state()
    action = _first_action(space, state)
    action_name = mifrost.RelationFormatter.format_action(adv_action(action))

    encoder = FlatRelationEncoder(
        domain,
        target_sources=[mifrost.TargetSource.Actions],
    )
    actual = encoder.encode_batch(
        [state, state],
        actions=[[action, action], []],
    ).as_pyg(as_batch=True)
    expected = flat_relation_data_from_pyg(
        Batch.from_data_list(
            [
                encoder.encode_pyg(state, actions=[action, action]),
                encoder.encode_pyg(state, actions=[]),
            ]
        )
    )

    assert actual.target_sizes.tolist() == [2, 0]
    assert actual.target_indices.tolist() == [0, 1]
    assert actual.target_candidate_ids.tolist() == [0, 1]
    assert actual.target_positions.tolist()[0] == actual.target_positions.tolist()[1]
    assert actual.graph_target_names(0) == [action_name, action_name]
    _assert_flat_batch_equal(actual, expected)


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
