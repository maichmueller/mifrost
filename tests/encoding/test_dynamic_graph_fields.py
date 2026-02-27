from __future__ import annotations

import pytest
import torch

import mifrost
from mifrost.encoders import _encoding_dict_to_pyg
from mifrost.graph_fields import CollateSpec, DType, GraphFieldSpec, Mode


def test_dtype_enum_exposes_python_side_members():
    assert "PYOBJ" in DType.__members__
    assert "STR" in DType.__members__
    assert DType("pyobj") is DType.PYOBJ
    assert DType("str") is DType.STR


def test_graph_field_spec_rejects_pyobj_dtype_for_native_fields():
    with pytest.raises(ValueError, match="native HGraph dynamic fields"):
        GraphFieldSpec(mode=Mode.STACK, dtype=DType.PYOBJ)


def test_batch_encodings_collate_spec_numeric_attrs_and_as_pyg(small_blocks):
    space, domain, problem = small_blocks
    encoder = mifrost.HGraphEncoder(domain)

    state0 = problem.get_initial_state()
    state1 = space._advanced_state_space_sampler.sample_state_n_steps_from_goal(0)

    enc0 = encoder.encode(state0)
    enc1 = encoder.encode(state1)
    enc0.goal_distance = 45
    enc0.target_indices = [0, 3, 5]
    enc1.goal_distance = 12
    enc1.target_indices = [1]

    batch_enc = mifrost.batch_encodings(
        [enc0, enc1],
        collate_spec={
            "goal_distance": CollateSpec(mode=Mode.STACK, dtype=DType.F32),
            "target_indices": CollateSpec(mode=Mode.RAGGED_CAT, dtype=DType.I64),
        },
    )
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

    # Dynamic collated attrs are Python attrs on BatchEncoding; they are not part of
    # the native as_dict tensor payload.
    data_from_dict = _encoding_dict_to_pyg(batch_enc.as_dict(), as_batch=True)
    assert not hasattr(data_from_dict, "goal_distance")
    assert not hasattr(data_from_dict, "target_indices")
    assert not hasattr(data_from_dict, "target_indices_ptr")


def test_batch_encodings_default_shallow_dict_collation(small_blocks):
    space, domain, problem = small_blocks
    encoder = mifrost.HGraphEncoder(domain)

    state0 = problem.get_initial_state()
    state1 = space._advanced_state_space_sampler.sample_state_n_steps_from_goal(0)

    enc0 = encoder.encode(state0)
    enc1 = encoder.encode(state1)
    enc0.transition_info = {"depth": 0, "tag": "root"}
    enc1.transition_info = {"depth": 1, "tag": "child"}

    batch_enc = mifrost.batch_encodings([enc0, enc1])
    assert batch_enc.transition_info == {
        "depth": [0, 1],
        "tag": ["root", "child"],
    }


def test_no_dynamic_graph_fields_emit_no_graph_tensor_schema(small_blocks):
    _, domain, problem = small_blocks
    encoder = mifrost.HGraphEncoder(domain)
    encoding = encoder.encode(problem.get_initial_state())
    payload = encoding.as_dict()

    schema = payload["schema"]
    tensors = payload["tensors"]
    assert "graph_tensors" not in schema
    assert not any(str(key).startswith("__graph__/") for key in tensors)


def test_encode_batch_accepts_pre_collated_batch_attrs_and_collate_spec(small_blocks):
    space, domain, problem = small_blocks
    encoder = mifrost.HGraphEncoder(domain)

    state0 = problem.get_initial_state()
    state1 = space._advanced_state_space_sampler.sample_state_n_steps_from_goal(0)
    returns = torch.tensor([0.5, 1.5], dtype=torch.float32)

    batch_enc = encoder.encode_batch(
        [state0, state1],
        batch_attrs={"returns": returns, "domain_path": "domain.pddl"},
        collate_spec={
            "returns": {"dtype": "f32", "mode": "stack", "dim": 1},
            "domain_path": {"dtype": "pyobj", "mode": "const"},
        },
    )
    assert torch.equal(batch_enc.returns, returns)
    assert batch_enc.domain_path == "domain.pddl"
    assert batch_enc.collate_spec()["returns"]["dtype"] == "f32"
    assert batch_enc.collate_spec()["domain_path"]["mode"] == "const"


def test_stored_collate_spec_metadata_is_not_reused_implicitly(small_blocks):
    _, domain, problem = small_blocks
    encoder = mifrost.HGraphEncoder(domain)
    state = problem.get_initial_state()

    enc0 = encoder.encode_batch(
        [state],
        batch_attrs={"targets": ["a0", "a1"]},
        collate_spec={"targets": {"dtype": "pyobj", "mode": "ragged_cat"}},
    )
    enc1 = encoder.encode_batch(
        [state],
        batch_attrs={"targets": ["b0"]},
        collate_spec={"targets": {"dtype": "pyobj", "mode": "ragged_cat"}},
    )

    rebatched = mifrost.batch_encodings([enc0, enc1])
    assert rebatched.targets == [["a0", "a1"], ["b0"]]
    assert rebatched.collate_spec() == {}


def test_encode_batch_rejects_reserved_batch_attr_keys(small_blocks):
    _, domain, problem = small_blocks
    encoder = mifrost.HGraphEncoder(domain)
    state = problem.get_initial_state()

    with pytest.raises(ValueError, match="collides with reserved/native key"):
        encoder.encode_batch([state], batch_attrs={"x": [1]})
