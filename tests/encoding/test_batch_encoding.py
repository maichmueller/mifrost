from __future__ import annotations

import torch
from torch_geometric.data import Batch, HeteroData

from mifrost.encoders import HGraphEncoder, _parts_to_pyg


def _assert_tensor_or_list_equal(actual, expected):
    if torch.is_tensor(expected):
        assert torch.is_tensor(actual)
        assert torch.equal(actual, expected)
        return
    assert actual == expected


def _assert_hetero_batch_equal(actual: HeteroData, expected: HeteroData) -> None:
    assert set(actual.node_types) == set(expected.node_types)
    assert set(actual.edge_types) == set(expected.edge_types)
    for node_type in expected.node_types:
        actual_store = actual[node_type]
        expected_store = expected[node_type]
        assert set(actual_store.keys()) == set(expected_store.keys())
        for key in expected_store.keys():
            _assert_tensor_or_list_equal(actual_store[key], expected_store[key])
    for edge_type in expected.edge_types:
        actual_store = actual[edge_type]
        expected_store = expected[edge_type]
        assert set(actual_store.keys()) == set(expected_store.keys())
        for key in expected_store.keys():
            _assert_tensor_or_list_equal(actual_store[key], expected_store[key])

    expected_globals = set(expected._global_store.keys())
    actual_globals = set(actual._global_store.keys())
    assert expected_globals == actual_globals
    for key in expected_globals:
        _assert_tensor_or_list_equal(
            actual._global_store[key], expected._global_store[key]
        )
    assert actual.num_graphs == expected.num_graphs


def test_encode_batch_matches_from_data_list_hetero(small_blocks):
    space, domain, problem = small_blocks
    encoder = HGraphEncoder(domain)

    states = [
        problem.get_initial_state(),
        space._advanced_state_space_sampler.sample_state_n_steps_from_goal(0),
    ]
    data_list = [encoder.encode(state) for state in states]
    expected_batch = Batch.from_data_list(data_list)

    actual_batch = encoder.encode_batch(states)

    _assert_hetero_batch_equal(actual_batch, expected_batch)


def test_stream_matches_encode_batch(small_blocks):
    space, domain, problem = small_blocks
    encoder = HGraphEncoder(domain)

    states = [
        problem.get_initial_state(),
        space._advanced_state_space_sampler.sample_state_n_steps_from_goal(0),
    ]
    expected_batch = encoder.encode_batch(states)

    stream = encoder.stream()
    for state in states:
        stream.append(state)
    actual_batch = stream.flush(as_batch=True)

    _assert_hetero_batch_equal(actual_batch, expected_batch)


def test_encode_batch_parts_roundtrip(small_blocks):
    space, domain, problem = small_blocks
    encoder = HGraphEncoder(domain)

    states = [
        problem.get_initial_state(),
        space._advanced_state_space_sampler.sample_state_n_steps_from_goal(0),
    ]
    parts = encoder.encode_batch_parts(states)
    actual_batch = _parts_to_pyg(parts, as_batch=True)
    expected_batch = encoder.encode_batch(states)

    _assert_hetero_batch_equal(actual_batch, expected_batch)


def test_encode_batch_without_metadata(small_blocks):
    space, domain, problem = small_blocks
    encoder = HGraphEncoder(domain)

    states = [
        problem.get_initial_state(),
        space._advanced_state_space_sampler.sample_state_n_steps_from_goal(0),
    ]
    batch = encoder.encode_batch(states, include_metadata=False)

    for node_type in batch.node_types:
        assert "node_names" not in batch[node_type]
    assert "object_names" not in batch._global_store


def test_include_empty_edge_types_flag_drops_empty_edges(small_blocks):
    space, domain, problem = small_blocks
    states = [
        problem.get_initial_state(),
        space._advanced_state_space_sampler.sample_state_n_steps_from_goal(0),
    ]

    dense_encoder = HGraphEncoder(domain, include_empty_edge_types=True)
    sparse_encoder = HGraphEncoder(domain, include_empty_edge_types=False)

    dense_batch = dense_encoder.encode_batch(states)
    sparse_batch = sparse_encoder.encode_batch(states)

    assert set(sparse_batch.edge_types).issubset(set(dense_batch.edge_types))

    for edge_type in sparse_batch.edge_types:
        edge_index = sparse_batch[edge_type].edge_index
        assert edge_index.numel() > 0

    for edge_type in set(dense_batch.edge_types) - set(sparse_batch.edge_types):
        edge_index = dense_batch[edge_type].edge_index
        assert edge_index.numel() == 0
