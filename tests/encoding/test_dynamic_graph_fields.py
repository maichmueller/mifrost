from __future__ import annotations

import pytest
import torch

import mifrost
from mifrost.encoders import _encoding_dict_to_pyg
from mifrost.graph_fields import DType, GraphFieldSpec, Mode


def test_dtype_enum_exposes_pyobj_member():
    assert "PYOBJ" in DType.__members__
    assert DType("pyobj") is DType.PYOBJ


def test_graph_field_spec_rejects_pyobj_dtype_for_native_fields():
    with pytest.raises(ValueError, match="dtype='pyobj'"):
        GraphFieldSpec(mode=Mode.STACK, dtype=DType.PYOBJ)


def test_encoded_graph_assignment_batch_graphs_and_as_pyg(small_blocks):
    space, domain, problem = small_blocks
    encoder = mifrost.HGraphEncoder(domain)
    encoder.register_fields(
        {
            "goal_distance": GraphFieldSpec(mode=Mode.STACK, dtype=DType.F32),
            "target_indices": GraphFieldSpec(mode=Mode.RAGGED_CAT, dtype=DType.I64),
        }
    )

    state0 = problem.get_initial_state()
    state1 = space._advanced_state_space_sampler.sample_state_n_steps_from_goal(0)

    g0 = encoder.encode_graph(state0)
    g1 = encoder.encode_graph(state1)
    g0.goal_distance = 45
    g0.target_indices = [0, 3, 5]
    g1.goal_distance = 12
    g1.target_indices = [1]

    batch_enc = encoder.batch_graphs([g0, g1])
    data = batch_enc.as_pyg()
    assert data.goal_distance.shape == (2,)
    assert torch.equal(
        data.goal_distance, torch.tensor([45.0, 12.0], dtype=torch.float32)
    )
    assert torch.equal(
        data.target_indices, torch.tensor([0, 3, 5, 1], dtype=torch.int64)
    )
    assert torch.equal(
        data.target_indices_ptr, torch.tensor([0, 3, 4], dtype=torch.int64)
    )

    data_from_dict = _encoding_dict_to_pyg(batch_enc.as_dict(), as_batch=True)
    assert torch.equal(data_from_dict.goal_distance, data.goal_distance)
    assert torch.equal(data_from_dict.target_indices, data.target_indices)
    assert torch.equal(data_from_dict.target_indices_ptr, data.target_indices_ptr)


def test_encoded_graph_cat_mode_batches_without_ptr(small_blocks):
    space, domain, problem = small_blocks
    encoder = mifrost.HGraphEncoder(domain)
    encoder.register_fields(
        {
            "target_concat": GraphFieldSpec(mode=Mode.CAT, dtype=DType.I64),
        }
    )

    state0 = problem.get_initial_state()
    state1 = space._advanced_state_space_sampler.sample_state_n_steps_from_goal(0)

    g0 = encoder.encode_graph(state0)
    g1 = encoder.encode_graph(state1)
    g0.target_concat = [2, 4]
    g1.target_concat = [1]

    batch_enc = encoder.batch_graphs([g0, g1])
    data = batch_enc.as_pyg()
    assert torch.equal(data.target_concat, torch.tensor([2, 4, 1], dtype=torch.int64))
    assert not hasattr(data, "target_concat_ptr")


def test_no_dynamic_graph_fields_emit_no_graph_tensor_schema(small_blocks):
    _, domain, problem = small_blocks
    encoder = mifrost.HGraphEncoder(domain)
    encoding = encoder.encode(problem.get_initial_state())
    payload = encoding.as_dict()

    schema = payload["schema"]
    tensors = payload["tensors"]
    assert "graph_tensors" not in schema
    assert not any(str(key).startswith("__graph__/") for key in tensors)


def test_encoded_graph_cat_dim1_for_matrix_field(small_blocks):
    space, domain, problem = small_blocks
    encoder = mifrost.HGraphEncoder(domain)
    encoder.register_fields(
        {
            "target_matrix": GraphFieldSpec(
                mode=Mode.CAT, dtype=DType.I64, dim=2, cat_dim=1
            ),
        }
    )

    state0 = problem.get_initial_state()
    state1 = space._advanced_state_space_sampler.sample_state_n_steps_from_goal(0)

    g0 = encoder.encode_graph(state0)
    g1 = encoder.encode_graph(state1)
    g0.target_matrix = [[10, 20], [30, 40]]
    g1.target_matrix = [[50], [60]]

    batch_enc = encoder.batch_graphs([g0, g1])
    data = batch_enc.as_pyg()
    assert torch.equal(
        data.target_matrix,
        torch.tensor([[10, 20, 50], [30, 40, 60]], dtype=torch.int64),
    )


def test_encoder_exposes_registered_graph_field_specs(small_blocks):
    _, domain, _ = small_blocks
    encoder = mifrost.HGraphEncoder(domain)
    encoder.register_fields(
        {
            "goal_distance": GraphFieldSpec(mode=Mode.STACK, dtype=DType.F32),
            "target_matrix": GraphFieldSpec(
                mode=Mode.CAT, dtype=DType.I64, dim=2, cat_dim=1
            ),
        }
    )

    assert encoder.field_keys == ["goal_distance", "target_matrix"]
    specs = encoder.field_specs
    assert specs["goal_distance"].mode is Mode.STACK
    assert specs["target_matrix"].cat_dim == 1


def test_encoder_register_graph_fields_accepts_dict_specs(small_blocks):
    _, domain, problem = small_blocks
    encoder = mifrost.HGraphEncoder(domain)
    encoder.register_fields(
        {
            "goal_distance": {
                "mode": "stack",
                "dtype": "f32",
                "dim": 1,
                "inc": {"kind": "none"},
            }
        }
    )

    specs = encoder.field_specs
    assert specs["goal_distance"].mode is Mode.STACK
    assert specs["goal_distance"].dtype is DType.F32

    graph = encoder.encode_graph(problem.get_initial_state())
    graph.goal_distance = 7
    data = encoder.batch_graphs([graph]).as_pyg(as_batch=False)
    assert torch.equal(data.goal_distance, torch.tensor([7.0], dtype=torch.float32))


def test_encoded_graph_accepts_zero_dim_tensors_for_graph_fields(small_blocks):
    _, domain, problem = small_blocks
    encoder = mifrost.HGraphEncoder(domain)
    encoder.register_fields(
        {
            "reward": GraphFieldSpec(mode=Mode.RAGGED_CAT, dtype=DType.F32),
            "done": GraphFieldSpec(mode=Mode.RAGGED_CAT, dtype=DType.I64),
        }
    )

    graph = encoder.encode_graph(problem.get_initial_state())
    graph.reward = torch.tensor(0.0)
    graph.done = torch.tensor(0)
    data = encoder.batch_graphs([graph]).as_pyg(as_batch=False)

    assert torch.equal(data.reward, torch.tensor([0.0], dtype=torch.float32))
    assert torch.equal(data.done, torch.tensor([0], dtype=torch.int64))


def test_dynamic_graph_field_batch_encoding_dumps_loads_roundtrip(small_blocks):
    _, domain, problem = small_blocks
    encoder = mifrost.HGraphEncoder(domain)
    encoder.register_fields(
        {
            "idx": {
                "mode": "stack",
                "dtype": "i64",
                "dim": 1,
                "cat_dim": 0,
                "inc": {"kind": "none"},
            }
        }
    )

    graph = encoder.encode_graph(problem.get_initial_state())
    graph.idx = 0
    sample = graph.finalize()

    loaded = mifrost.BatchEncoding.loads(sample.dumps(include_metadata=True))
    assert torch.equal(loaded.get_field("idx"), torch.tensor([0], dtype=torch.int64))
