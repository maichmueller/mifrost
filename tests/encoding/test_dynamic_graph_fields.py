from __future__ import annotations

import torch

import mifrost
from mifrost.encoders import _encoding_dict_to_pyg
from mifrost.graph_fields import DType, GraphFieldSpec, Mode


def test_encoded_graph_assignment_batch_graphs_and_as_pyg(small_blocks):
    space, domain, problem = small_blocks
    encoder = mifrost.HGraphEncoder(domain)
    encoder.register_graph_fields(
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
    encoder.register_graph_fields(
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
    encoder.register_graph_fields(
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
