import pytest
import numpy as np
import torch
import mifrost
from torch_geometric.data import Batch


def test_batch_builder_basics():
    builder = mifrost.BatchBuilder()

    # 1. Add some data
    x = torch.randn(10, 5)
    builder.add_node_features("atom", "x", x)

    # 2. Add edges
    src = torch.tensor([0, 1, 2], dtype=torch.int64)
    dst = torch.tensor([1, 2, 0], dtype=torch.int64)
    builder.add_edges("atom", "rel", "atom", src, dst)

    # 3. Next graph
    builder.next_graph()

    # 4. Add second graph (offset logic implicit)
    builder.add_node_features("atom", "x", x)
    builder.add_edges("atom", "rel", "atom", src, dst)  # Should be offset by 10
    builder.next_graph()

    # 5. Build
    batch = builder.build().as_pyg()
    assert isinstance(batch, Batch)
    assert "atom" in batch.node_types

    # Verify content
    out_x = batch["atom"].x
    assert out_x.shape == (20, 5)
    assert torch.allclose(out_x[0:10], x)
    assert torch.allclose(out_x[10:20], x)

    # Verify edge offsets
    out_src = batch[("atom", "rel", "atom")].edge_index[0]
    # First graph: 0, 1, 2
    # Second graph: 10, 11, 12
    expected_src = torch.cat([src, src + 10])
    assert torch.equal(out_src, expected_src)

    ptr = batch["atom"].ptr
    assert torch.equal(ptr, torch.tensor([0, 10, 20], dtype=torch.int64))


def test_hgraph_encoder_instantiation():
    # Needs domain binding or mock
    pass


def _single_graph_with_stack_field(value: float):
    builder = mifrost.BatchBuilder()
    builder.set_graph_kind("hetero")
    builder.register_graph_field(
        "goal_distance",
        {
            "dtype": "f32",
            "mode": "stack",
            "dim": 1,
            "inc": {"kind": "none"},
        },
    )
    builder.add_node_features("atom", "x", torch.zeros(1, 1))
    builder.set_graph_field("goal_distance", value)
    builder.next_graph()
    return builder.build()


def _single_graph_with_ragged_i64_field(values):
    builder = mifrost.BatchBuilder()
    builder.set_graph_kind("hetero")
    builder.register_graph_field(
        "target_indices",
        {
            "dtype": "i64",
            "mode": "ragged_cat",
            "dim": 1,
            "inc": {"kind": "none"},
        },
    )
    builder.add_node_features("atom", "x", torch.zeros(1, 1))
    builder.set_graph_field("target_indices", list(values))
    builder.next_graph()
    return builder.build()


def _single_graph_with_inc_ragged_field(symbol_count: int, positions):
    builder = mifrost.BatchBuilder()
    builder.set_graph_kind("hetero")
    builder.register_graph_field(
        "target_positions",
        {
            "dtype": "i64",
            "mode": "ragged_cat",
            "dim": 1,
            "inc": {"kind": "node_offset", "node_type": "symbol"},
        },
    )
    builder.add_node_features("symbol", "x", torch.zeros(symbol_count, 1))
    builder.set_graph_field("target_positions", list(positions))
    builder.next_graph()
    return builder.build()


def _single_graph_with_cat_i64_field(values):
    builder = mifrost.BatchBuilder()
    builder.set_graph_kind("hetero")
    builder.register_graph_field(
        "target_concat",
        {
            "dtype": "i64",
            "mode": "cat",
            "dim": 1,
            "inc": {"kind": "none"},
        },
    )
    builder.add_node_features("atom", "x", torch.zeros(1, 1))
    if values is not None:
        builder.set_graph_field("target_concat", list(values))
    builder.next_graph()
    return builder.build()


def _single_graph_with_const_i64_field(value: int):
    builder = mifrost.BatchBuilder()
    builder.set_graph_kind("hetero")
    builder.register_graph_field(
        "problem_id",
        {
            "dtype": "i64",
            "mode": "const",
            "dim": 1,
            "inc": {"kind": "none"},
        },
    )
    builder.add_node_features("atom", "x", torch.zeros(1, 1))
    builder.set_graph_field("problem_id", value)
    builder.next_graph()
    return builder.build()


def _single_graph_with_inc_cat_field(symbol_count: int, values):
    builder = mifrost.BatchBuilder()
    builder.set_graph_kind("hetero")
    builder.register_graph_field(
        "target_concat",
        {
            "dtype": "i64",
            "mode": "cat",
            "dim": 1,
            "inc": {"kind": "node_offset", "node_type": "symbol"},
        },
    )
    builder.add_node_features("symbol", "x", torch.zeros(symbol_count, 1))
    builder.set_graph_field("target_concat", list(values))
    builder.next_graph()
    return builder.build()


def _single_graph_with_matrix_field(values, *, mode: str = "cat", cat_dim: int = 1):
    builder = mifrost.BatchBuilder()
    builder.set_graph_kind("hetero")
    builder.register_graph_field(
        "target_matrix",
        {
            "dtype": "i64",
            "mode": mode,
            "dim": 2,
            "cat_dim": cat_dim,
            "inc": {"kind": "none"},
        },
    )
    builder.add_node_features("atom", "x", torch.zeros(1, 1))
    if values is not None:
        builder.set_graph_field("target_matrix", values)
    builder.next_graph()
    return builder.build()


def test_batch_encodings_collates_stack_graph_field():
    enc0 = _single_graph_with_stack_field(1.0)
    enc1 = _single_graph_with_stack_field(2.0)

    builder = mifrost.BatchBuilder()
    builder.set_graph_kind("hetero")
    builder.append_batch_encoding(enc0)
    builder.append_batch_encoding(enc1)
    manual = builder.build().as_dict()
    manual_goal_distance = torch.as_tensor(manual["tensors"]["__graph__/goal_distance"])
    assert torch.allclose(
        manual_goal_distance, torch.tensor([1.0, 2.0], dtype=torch.float32)
    )
    assert enc0.schema_fingerprint() == enc1.schema_fingerprint()

    batched = mifrost.batch_encodings([enc0, enc1])
    out = batched.as_dict()
    tensors = out["tensors"]

    assert "__graph__/goal_distance" in tensors
    goal_distance = torch.as_tensor(tensors["__graph__/goal_distance"])
    assert goal_distance.ndim == 1
    assert torch.allclose(goal_distance, torch.tensor([1.0, 2.0], dtype=torch.float32))

    schema = out["schema"]
    assert "graph_tensors" in schema
    assert any(entry["attr"] == "goal_distance" for entry in schema["graph_tensors"])


def test_batch_encodings_collates_ragged_graph_field_with_ptr():
    enc0 = _single_graph_with_ragged_i64_field([5, 6])
    enc1 = _single_graph_with_ragged_i64_field([7])

    batched = mifrost.batch_encodings([enc0, enc1])
    out = batched.as_dict()
    tensors = out["tensors"]

    assert "__graph__/target_indices" in tensors
    assert "__graph__/target_indices/ptr" in tensors

    values = torch.as_tensor(tensors["__graph__/target_indices"])
    ptr = torch.as_tensor(tensors["__graph__/target_indices/ptr"])
    assert torch.equal(values, torch.tensor([5, 6, 7], dtype=torch.int64))
    assert torch.equal(ptr, torch.tensor([0, 2, 3], dtype=torch.int64))


def test_batch_encodings_applies_node_offset_inc_for_ragged_i64_field():
    enc0 = _single_graph_with_inc_ragged_field(10, [2, 7])
    enc1 = _single_graph_with_inc_ragged_field(5, [0, 3])

    batched = mifrost.batch_encodings([enc0, enc1])
    tensors = batched.as_dict()["tensors"]
    values = torch.as_tensor(tensors["__graph__/target_positions"])
    ptr = torch.as_tensor(tensors["__graph__/target_positions/ptr"])

    assert torch.equal(values, torch.tensor([2, 7, 10, 13], dtype=torch.int64))
    assert torch.equal(ptr, torch.tensor([0, 2, 4], dtype=torch.int64))


def test_batch_encodings_collates_cat_graph_field():
    enc0 = _single_graph_with_cat_i64_field([5, 6])
    enc1 = _single_graph_with_cat_i64_field([7])

    batched = mifrost.batch_encodings([enc0, enc1])
    out = batched.as_dict()
    tensors = out["tensors"]
    assert "__graph__/target_concat" in tensors
    assert "__graph__/target_concat/ptr" not in tensors
    values = torch.as_tensor(tensors["__graph__/target_concat"])
    assert torch.equal(values, torch.tensor([5, 6, 7], dtype=torch.int64))


def test_append_batch_encoding_cat_missing_treated_as_empty():
    enc0 = _single_graph_with_cat_i64_field([5, 6])
    enc1 = _single_graph_with_cat_i64_field(None)

    builder = mifrost.BatchBuilder()
    builder.set_graph_kind("hetero")
    builder.append_batch_encoding(enc0)
    builder.append_batch_encoding(enc1)
    out = builder.build().as_dict()["tensors"]
    values = torch.as_tensor(out["__graph__/target_concat"])
    assert torch.equal(values, torch.tensor([5, 6], dtype=torch.int64))


def test_batch_encodings_applies_node_offset_inc_for_cat_i64_field():
    enc0 = _single_graph_with_inc_cat_field(10, [2, 7])
    enc1 = _single_graph_with_inc_cat_field(5, [0, 3])

    batched = mifrost.batch_encodings([enc0, enc1])
    out = batched.as_dict()["tensors"]
    values = torch.as_tensor(out["__graph__/target_concat"])
    assert torch.equal(values, torch.tensor([2, 7, 10, 13], dtype=torch.int64))


def test_batch_encodings_collates_const_graph_field():
    enc0 = _single_graph_with_const_i64_field(42)
    enc1 = _single_graph_with_const_i64_field(42)

    batched = mifrost.batch_encodings([enc0, enc1])
    out = batched.as_dict()["tensors"]
    values = torch.as_tensor(out["__graph__/problem_id"])
    assert torch.equal(values, torch.tensor([42], dtype=torch.int64))


def test_append_batch_encoding_missing_const_raises():
    enc_with_const = _single_graph_with_const_i64_field(42)

    builder_without_const = mifrost.BatchBuilder()
    builder_without_const.set_graph_kind("hetero")
    builder_without_const.add_node_features("atom", "x", torch.zeros(1, 1))
    builder_without_const.next_graph()
    enc_without_const = builder_without_const.build()

    builder = mifrost.BatchBuilder()
    builder.set_graph_kind("hetero")
    builder.append_batch_encoding(enc_with_const)
    with pytest.raises(ValueError, match="missing required graph field 'problem_id'"):
        builder.append_batch_encoding(enc_without_const)


def test_batch_encodings_collates_cat_dim1_cat_graph_field():
    enc0 = _single_graph_with_matrix_field([[10, 20], [30, 40]], mode="cat", cat_dim=1)
    enc1 = _single_graph_with_matrix_field([[50], [60]], mode="cat", cat_dim=1)

    batched = mifrost.batch_encodings([enc0, enc1]).as_dict()
    values = torch.as_tensor(batched["tensors"]["__graph__/target_matrix"])
    assert values.shape == (2, 3)
    assert torch.equal(
        values, torch.tensor([[10, 20, 50], [30, 40, 60]], dtype=torch.int64)
    )
    schema = batched["schema"]
    entry = next(x for x in schema["graph_tensors"] if x["attr"] == "target_matrix")
    assert entry["cat_dim"] == 1


def test_batch_encodings_collates_cat_dim1_ragged_graph_field_with_ptr():
    enc0 = _single_graph_with_matrix_field(
        [[10, 20], [30, 40]], mode="ragged_cat", cat_dim=1
    )
    enc1 = _single_graph_with_matrix_field([[50], [60]], mode="ragged_cat", cat_dim=1)

    batched = mifrost.batch_encodings([enc0, enc1]).as_dict()
    values = torch.as_tensor(batched["tensors"]["__graph__/target_matrix"])
    ptr = torch.as_tensor(batched["tensors"]["__graph__/target_matrix/ptr"])
    assert values.shape == (2, 3)
    assert torch.equal(
        values, torch.tensor([[10, 20, 50], [30, 40, 60]], dtype=torch.int64)
    )
    assert torch.equal(ptr, torch.tensor([0, 2, 3], dtype=torch.int64))


def test_graph_field_cat_dim1_dim_gt_1_requires_2d_input():
    builder = mifrost.BatchBuilder()
    builder.set_graph_kind("hetero")
    builder.register_graph_field(
        "target_matrix",
        {
            "dtype": "i64",
            "mode": "cat",
            "dim": 2,
            "cat_dim": 1,
            "inc": {"kind": "none"},
        },
    )
    builder.add_node_features("atom", "x", torch.zeros(1, 1))
    with pytest.raises(ValueError, match="requires a 2D value shaped \\[dim, N\\]"):
        builder.set_graph_field("target_matrix", [10, 20, 30, 40])


def test_batch_encodings_schema_fingerprint_mismatch_on_cat_dim():
    enc0 = _single_graph_with_matrix_field([[10, 20]], mode="cat", cat_dim=0)
    enc1 = _single_graph_with_matrix_field([[10], [20]], mode="cat", cat_dim=1)

    with pytest.raises(ValueError, match="schema_fingerprint mismatch"):
        mifrost.batch_encodings([enc0, enc1])
