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
    def _store_keys(store):
        return {key for key in store.keys() if key != "num_nodes"}

    assert set(actual.node_types) == set(expected.node_types)
    assert set(actual.edge_types) == set(expected.edge_types)
    for node_type in expected.node_types:
        actual_store = actual[node_type]
        expected_store = expected[node_type]
        assert _store_keys(actual_store) == _store_keys(expected_store)
        for key in _store_keys(expected_store):
            _assert_tensor_or_list_equal(actual_store[key], expected_store[key])
    for edge_type in expected.edge_types:
        actual_store = actual[edge_type]
        expected_store = expected[edge_type]
        assert _store_keys(actual_store) == _store_keys(expected_store)
        for key in _store_keys(expected_store):
            _assert_tensor_or_list_equal(actual_store[key], expected_store[key])

    expected_globals = set(expected._global_store.keys())
    actual_globals = set(actual._global_store.keys())
    assert expected_globals == actual_globals
    for key in expected_globals:
        _assert_tensor_or_list_equal(
            actual._global_store[key], expected._global_store[key]
        )
    assert int(getattr(actual, "num_graphs", 1)) == int(
        getattr(expected, "num_graphs", 1)
    )


def test_encode_batch_matches_from_data_list_hetero(small_blocks):
    space, domain, problem = small_blocks
    encoder = HGraphEncoder(domain)

    states = [
        problem.get_initial_state(),
        space._advanced_state_space_sampler.sample_state_n_steps_from_goal(0),
    ]
    data_list = [encoder.encode_pyg(state) for state in states]
    expected_batch = Batch.from_data_list(data_list)

    actual_batch = encoder.encode_batch(states).as_pyg(as_batch=True)

    _assert_hetero_batch_equal(actual_batch, expected_batch)


def test_stream_matches_encode_batch(small_blocks):
    space, domain, problem = small_blocks
    encoder = HGraphEncoder(domain)

    states = [
        problem.get_initial_state(),
        space._advanced_state_space_sampler.sample_state_n_steps_from_goal(0),
    ]
    expected_batch = encoder.encode_batch(states).as_pyg(as_batch=True)

    stream = encoder.stream()
    for state in states:
        stream.append(state)
    actual_batch = stream.flush_pyg(as_batch=True)

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
    expected_batch = encoder.encode_batch(states).as_pyg(as_batch=True)

    _assert_hetero_batch_equal(actual_batch, expected_batch)


def test_encode_batch_without_metadata(small_blocks):
    space, domain, problem = small_blocks
    encoder = HGraphEncoder(domain)

    states = [
        problem.get_initial_state(),
        space._advanced_state_space_sampler.sample_state_n_steps_from_goal(0),
    ]
    batch = encoder.encode_batch_pyg(states, include_metadata=False)

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

    dense_batch = dense_encoder.encode_batch_pyg(states)
    sparse_batch = sparse_encoder.encode_batch_pyg(states)

    assert set(sparse_batch.edge_types).issubset(set(dense_batch.edge_types))

    for edge_type in sparse_batch.edge_types:
        edge_index = sparse_batch[edge_type].edge_index
        assert edge_index.numel() > 0

    for edge_type in set(dense_batch.edge_types) - set(sparse_batch.edge_types):
        edge_index = dense_batch[edge_type].edge_index
        assert edge_index.numel() == 0


def test_encode_native_matches_encode_pyg(small_blocks):
    space, domain, problem = small_blocks
    encoder = HGraphEncoder(domain)

    state = problem.get_initial_state()
    native = encoder.encode(state)
    from_native = native.as_pyg()
    direct = encoder.encode_pyg(state)

    _assert_hetero_batch_equal(from_native, direct)


def test_export_node_names_flag_disables_metadata(small_blocks):
    _space, domain, problem = small_blocks
    encoder = HGraphEncoder(domain, export_node_names=False)

    state = problem.get_initial_state()
    encoding = encoder.encode(state)
    parts = encoding.to_parts()

    assert parts.get("node_names", {}) == {}
    assert list(parts.get("object_names", [])) == []
