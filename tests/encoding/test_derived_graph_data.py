from __future__ import annotations

import torch
from torch_geometric.data import Batch

from mifrost.encoders.derived_graph_data import (
    DerivedGraphData,
    normalize_derived_graph_batch_metadata,
)


def test_core_batching_concatenates_channels() -> None:
    data = DerivedGraphData(
        x_ids=torch.tensor([[0], [1], [2]]),
        edge_index=torch.tensor([[0, 1], [1, 2]]),
        edge_attr_ids=torch.tensor([[5], [7]]),
    )
    batch = Batch.from_data_list([data, data])
    assert isinstance(batch, DerivedGraphData)
    assert batch.x_ids.shape == (6, 1)
    assert torch.equal(batch.x_ids, torch.tensor([[0], [1], [2], [0], [1], [2]]))
    assert torch.equal(batch.edge_index, torch.tensor([[0, 1, 3, 4], [1, 2, 4, 5]]))
    assert batch.edge_attr_ids.shape == (4, 1)
    assert batch.num_nodes == 6


def test_hyperedge_batching_offsets_node_and_hyperedge_rows() -> None:
    first = DerivedGraphData(
        x_ids=torch.arange(2).unsqueeze(1),
        edge_index=torch.empty((2, 0), dtype=torch.long),
        edge_attr_ids=torch.empty((0, 1), dtype=torch.long),
        hyperedge_index=torch.tensor([[0, 1, 0], [0, 0, 1]]),
        hyperedge_attr_ids=torch.tensor([[10], [20]]),
    )
    second = DerivedGraphData(
        x_ids=torch.arange(3).unsqueeze(1),
        edge_index=torch.empty((2, 0), dtype=torch.long),
        edge_attr_ids=torch.empty((0, 1), dtype=torch.long),
        hyperedge_index=torch.tensor([[2], [0]]),
        hyperedge_attr_ids=torch.tensor([[30]]),
    )
    batch = Batch.from_data_list([first, second])
    assert torch.equal(
        batch.hyperedge_index, torch.tensor([[0, 1, 0, 4], [0, 0, 1, 2]])
    )
    assert torch.equal(batch.hyperedge_attr_ids, torch.tensor([[10], [20], [32]]))


def test_tuple_batching_and_padded_tuple_matrix() -> None:
    first = DerivedGraphData(
        x_ids=torch.arange(3).unsqueeze(1),
        edge_index=torch.empty((2, 0), dtype=torch.long),
        tuple_args=torch.tensor([10, 20, 30]),
        tuple_ptr=torch.tensor([0, 2, 3]),
        tuple_rel_ids=torch.tensor([4, 5]),
    )
    second = DerivedGraphData(
        x_ids=torch.arange(4).unsqueeze(1),
        edge_index=torch.empty((2, 0), dtype=torch.long),
        tuple_args=torch.tensor([40, 50, 60]),
        tuple_ptr=torch.tensor([0, 1, 3]),
        tuple_rel_ids=torch.tensor([6, 7]),
    )
    batch = Batch.from_data_list([first, second])
    assert torch.equal(batch.tuple_args, torch.tensor([10, 20, 30, 43, 53, 63]))
    assert torch.equal(batch.tuple_ptr, torch.tensor([0, 2, 3, 2, 3, 5]))
    assert torch.equal(batch.tuple_rel_ids, torch.tensor([4, 5, 8, 9]))

    args_matrix, mask = second.padded_tuple_matrix()
    assert args_matrix.shape == (2, 2)
    assert torch.equal(args_matrix, torch.tensor([[40, -1], [50, 60]]))
    assert torch.equal(mask, torch.tensor([[True, False], [True, True]]))

    ragged = DerivedGraphData(
        x_ids=torch.arange(2).unsqueeze(1),
        edge_index=torch.empty((2, 0), dtype=torch.long),
        tuple_args=torch.tensor([1, 2, 3, 4]),
        tuple_ptr=torch.tensor([0, 3, 4]),
        tuple_rel_ids=torch.tensor([0, 0]),
    )
    ragged_args, ragged_mask = ragged.padded_tuple_matrix(fill_value=9)
    assert torch.equal(ragged_args, torch.tensor([[1, 2, 3], [4, 9, 9]]))
    assert torch.equal(
        ragged_mask, torch.tensor([[True, True, True], [True, False, False]])
    )

    empty_data, empty_mask = DerivedGraphData().padded_tuple_matrix()
    assert empty_data.shape == (0, 0)
    assert empty_mask.shape == (0, 0)
    assert empty_data.dtype == torch.long
    assert empty_mask.dtype == torch.bool


def test_spd_batching_shifts_endpoints_by_num_nodes() -> None:
    first = DerivedGraphData(
        x_ids=torch.arange(2).unsqueeze(1),
        edge_index=torch.empty((2, 0), dtype=torch.long),
        spd_src=torch.tensor([0, 1]),
        spd_dst=torch.tensor([1, 0]),
        spd_dist=torch.tensor([5, 6]),
    )
    second = DerivedGraphData(
        x_ids=torch.arange(3).unsqueeze(1),
        edge_index=torch.empty((2, 0), dtype=torch.long),
        spd_src=torch.tensor([0]),
        spd_dst=torch.tensor([2]),
        spd_dist=torch.tensor([7]),
    )
    batch = Batch.from_data_list([first, second])
    assert torch.equal(batch.spd_src, torch.tensor([0, 1, 2]))
    assert torch.equal(batch.spd_dst, torch.tensor([1, 0, 4]))
    assert torch.equal(batch.spd_dist, torch.tensor([5, 6, 7]))


def test_metadata_lists_survive_batching_via_normalization() -> None:
    data = DerivedGraphData(
        x_ids=torch.arange(2).unsqueeze(1),
        edge_index=torch.empty((2, 0), dtype=torch.long),
        vocab_roles=["entity"],
        vocab_predicates=["on", "clear"],
        vocab_edge_kinds=["pre", "post"],
        channel_names=["role"],
        edge_channel_names=["kind"],
    )
    batch = Batch.from_data_list([data, data])
    assert batch.vocab_roles == [["entity"], ["entity"]]
    normalized = normalize_derived_graph_batch_metadata(batch)
    assert normalized.vocab_roles == ["entity"]
    assert normalized.vocab_predicates == ["on", "clear"]
    assert normalized.vocab_edge_kinds == ["pre", "post"]
    assert normalized.channel_names == ["role"]
    assert normalized.edge_channel_names == ["kind"]

    single = normalize_derived_graph_batch_metadata(data)
    assert single.vocab_predicates == ["on", "clear"]


def test_schema_summary_counts_all_channels() -> None:
    data = DerivedGraphData(
        x_ids=torch.arange(3).unsqueeze(1),
        edge_index=torch.tensor([[0, 1], [1, 2]]),
        edge_attr_ids=torch.tensor([[0], [1]]),
        hyperedge_index=torch.tensor([[0, 1], [0, 0]]),
        hyperedge_attr_ids=torch.tensor([[0], [1]]),
        tuple_args=torch.tensor([0, 1]),
        tuple_ptr=torch.tensor([0, 1, 2]),
        tuple_rel_ids=torch.tensor([0, 1]),
        spd_src=torch.tensor([0]),
        spd_dst=torch.tensor([1]),
        spd_dist=torch.tensor([4]),
    )
    assert data.schema_summary == {
        "nodes": 3,
        "edges": 2,
        "hyperedges": 2,
        "tuples": 2,
        "spd_pairs": 1,
    }
