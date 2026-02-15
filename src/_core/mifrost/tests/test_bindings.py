import gc
import inspect
import io
import os
import pickle
import subprocess
import sys
from pathlib import Path

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


def test_batch_builder_exposes_registered_graph_field_specs():
    builder = mifrost.BatchBuilder()
    builder.set_graph_kind("hetero")
    builder.register_graph_field(
        "a",
        {"dtype": "f32", "mode": "stack", "dim": 1, "inc": {"kind": "none"}},
    )
    builder.register_graph_field(
        "m",
        {
            "dtype": "i64",
            "mode": "cat",
            "dim": 2,
            "cat_dim": -1,
            "inc": {"kind": "none"},
        },
    )

    assert builder.graph_field_keys() == ["a", "m"]
    specs = builder.graph_field_specs()
    assert specs["a"]["dtype"] == "f32"
    assert specs["a"]["mode"] == "stack"
    assert specs["m"]["dtype"] == "i64"
    assert specs["m"]["mode"] == "cat"
    assert specs["m"]["dim"] == 2
    assert specs["m"]["cat_dim"] == 1  # -1 normalized to 1


def test_batch_builder_map_views_are_read_only_and_dict_like():
    builder = mifrost.BatchBuilder()
    builder.set_schema_flag("edge_features", True)
    builder.add_node_features("atom", "x", torch.zeros(2, 3))

    flags_view = builder.schema_flags_view()
    dims_view = builder.node_feature_dims_view()

    assert isinstance(flags_view, mifrost.MapView)
    assert isinstance(flags_view, mifrost.MapView[str, bool])
    assert isinstance(dims_view, mifrost.MapView[str, int])
    assert flags_view.key_type is str
    assert flags_view.value_type is bool
    assert dims_view.key_type is str
    assert dims_view.value_type is int
    assert len(flags_view) == 1
    assert bool(flags_view)
    assert "edge_features" in flags_view
    assert flags_view["edge_features"] is True
    assert list(flags_view) == ["edge_features"]
    assert list(flags_view.keys()) == ["edge_features"]
    assert list(flags_view.values()) == [True]
    assert list(flags_view.items()) == [("edge_features", True)]

    # Base map-view API should expose type-erased helpers beyond len/bool/as_dict.
    flags_impl = flags_view._impl
    assert isinstance(flags_impl, mifrost._core.MapViewBase)
    assert hasattr(mifrost._core.MapViewBase, "__getitem__")
    assert hasattr(mifrost._core.MapViewBase, "__contains__")
    assert "edge_features" in flags_impl
    assert flags_impl["edge_features"] is True
    assert flags_impl.contains("edge_features")
    assert flags_impl.contains("missing") is False
    assert flags_impl.contains(1) is False
    assert flags_impl.at("edge_features") is True
    with pytest.raises(KeyError):
        flags_impl.at("missing")
    missing = object()
    assert flags_impl.get("edge_features", False) is True
    assert flags_impl.get("missing", missing) is missing
    assert flags_impl.keys_list() == ["edge_features"]
    assert flags_impl.values_list() == [True]
    assert flags_impl.items_list() == [("edge_features", True)]

    assert len(dims_view) == 1
    assert dims_view["atom"] == 3
    assert list(dims_view.keys()) == ["atom"]
    assert list(dims_view.values()) == [3]
    assert list(dims_view.items()) == [("atom", 3)]

    with pytest.raises(TypeError):
        flags_view["edge_features"] = False
    with pytest.raises(TypeError):
        del flags_view["edge_features"]

    copied = flags_view.as_dict()
    copied["edge_features"] = False
    assert flags_view["edge_features"] is True


def test_map_views_keep_owner_alive():
    builder = mifrost.BatchBuilder()
    builder.set_schema_flag("predicate_nodes", False)
    builder.add_node_features("atom", "x", torch.zeros(2, 4))
    builder.next_graph()
    encoding = builder.build()

    dims_view = encoding.node_feature_dims_view()
    flags_view = encoding.schema_flags_view()
    del encoding
    gc.collect()

    assert dims_view["atom"] == 4
    assert flags_view["predicate_nodes"] is False


def test_batch_encoding_schema_flags_property_uses_map_view():
    builder = mifrost.BatchBuilder()
    builder.set_graph_kind("hetero")
    builder.set_schema_flag("supports_literals", True)
    builder.add_node_features("atom", "x", torch.zeros(2, 3))
    builder.next_graph()
    encoding = builder.build()

    flags = encoding.schema_flags
    assert isinstance(flags, mifrost._core.MapViewBase)
    assert flags["supports_literals"] is True
    assert flags.as_dict() == {"supports_literals": True}


def test_schema_flags_view_is_read_only():
    schema_dict = mifrost.Schema().to_dict()
    schema_dict["graph_kind"] = "hetero"
    schema_dict["flags"] = {"supports_literals": True}
    schema = mifrost.Schema.from_dict(schema_dict)

    flags_view = schema.flags_view()
    assert isinstance(flags_view, mifrost.MapView)
    assert isinstance(flags_view, mifrost.MapView[str, bool])
    assert flags_view["supports_literals"] is True
    assert list(flags_view.items()) == [("supports_literals", True)]

    with pytest.raises(TypeError):
        flags_view["supports_literals"] = False


def test_batch_encoding_save_load_roundtrip(tmp_path):
    encoding = _single_graph_with_stack_field(3.5)
    path = tmp_path / "encoding.pkl"
    encoding.save(str(path), include_metadata=True)

    loaded = mifrost.BatchEncoding.load(str(path))
    assert loaded.num_graphs == encoding.num_graphs
    assert loaded.graph_kind == encoding.graph_kind
    assert loaded.schema_fingerprint() == encoding.schema_fingerprint()
    _assert_tensor_payload_equal(loaded, encoding)


def test_batch_encoding_dumps_loads_roundtrip():
    encoding = _single_graph_with_stack_field(2.0)
    payload = encoding.dumps(include_metadata=True)
    assert isinstance(payload, bytes)

    loaded = mifrost.BatchEncoding.loads(payload)
    assert loaded.schema_fingerprint() == encoding.schema_fingerprint()
    _assert_tensor_payload_equal(loaded, encoding)


def test_batch_encoding_dumps_default_includes_metadata_and_python_attrs():
    builder = mifrost.BatchBuilder()
    builder.set_graph_kind("hetero")
    builder.add_node_features("atom", "x", torch.zeros(2, 1))
    builder.set_node_names("atom", ["o0", "o1"])
    builder.set_object_names(["o0", "o1"])
    builder.next_graph()
    encoding = builder.build()
    encoding.transition_info = {"depth": 1}

    payload = encoding.dumps()
    loaded = mifrost.BatchEncoding.loads(payload)
    loaded_dict = loaded.as_dict()
    assert loaded_dict["node_names"]["atom"] == ["o0", "o1"]
    assert loaded_dict["object_names"] == ["o0", "o1"]
    assert loaded.transition_info == {"depth": 1}


def test_batch_encoding_pickle_and_torch_pickle_roundtrip():
    script = """
import inspect
import io
import pickle
import torch
import mifrost

b = mifrost.BatchBuilder()
b.set_graph_kind("hetero")
b.register_graph_field(
    "goal_distance",
    {"dtype": "f32", "mode": "stack", "dim": 1, "inc": {"kind": "none"}},
)
b.add_node_features("atom", "x", torch.zeros(1, 1))
b.set_graph_field("goal_distance", 7.0)
b.next_graph()
encoding = b.build()

restored = pickle.loads(pickle.dumps(encoding))
assert isinstance(restored, mifrost.BatchEncoding)
assert restored.schema_fingerprint() == encoding.schema_fingerprint()

buffer = io.BytesIO()
torch.save(encoding, buffer)
buffer.seek(0)
load_kwargs = {}
if "weights_only" in inspect.signature(torch.load).parameters:
    load_kwargs["weights_only"] = False
restored_torch = torch.load(buffer, **load_kwargs)
assert isinstance(restored_torch, mifrost.BatchEncoding)
assert restored_torch.schema_fingerprint() == encoding.schema_fingerprint()
print("ok")
"""

    env = os.environ.copy()
    source_root = Path(__file__).resolve().parents[3]
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{source_root}:{existing}" if existing else str(source_root)
    result = subprocess.run(
        [sys.executable, "-c", script],
        text=True,
        capture_output=True,
        env=env,
        check=False,
    )
    assert result.returncode == 0, result.stderr + result.stdout


def test_batch_encoding_dynamic_python_attrs_roundtrip(tmp_path):
    encoding = _single_graph_with_stack_field(4.0)
    encoding.targets = ["s0", "s1"]
    encoding.transition_info = {"depth": 2, "source": "demo"}

    payload = encoding.dumps(include_metadata=True)
    loaded = mifrost.BatchEncoding.loads(payload)
    assert loaded.targets == ["s0", "s1"]
    assert loaded.transition_info == {"depth": 2, "source": "demo"}

    path = tmp_path / "encoding_with_python_attrs.pkl"
    encoding.save(str(path), include_metadata=True)
    loaded_from_file = mifrost.BatchEncoding.load(str(path))
    assert loaded_from_file.targets == ["s0", "s1"]
    assert loaded_from_file.transition_info == {"depth": 2, "source": "demo"}

    restored_pickle = pickle.loads(pickle.dumps(encoding))
    assert restored_pickle.targets == ["s0", "s1"]
    assert restored_pickle.transition_info == {"depth": 2, "source": "demo"}

    buffer = io.BytesIO()
    torch.save(encoding, buffer)
    buffer.seek(0)
    load_kwargs = {}
    if "weights_only" in inspect.signature(torch.load).parameters:
        load_kwargs["weights_only"] = False
    restored_torch = torch.load(buffer, **load_kwargs)
    assert restored_torch.targets == ["s0", "s1"]
    assert restored_torch.transition_info == {"depth": 2, "source": "demo"}


def test_batch_encodings_collates_python_fields_with_specs():
    enc0 = _single_graph_with_stack_field(1.0)
    enc1 = _single_graph_with_stack_field(2.0)

    enc0.targets = [10, 11]
    enc1.targets = [12]
    enc0.transition_label = "left"
    enc1.transition_label = "right"
    enc0.domain_path = "domain.pddl"
    enc1.domain_path = "domain.pddl"

    batched = mifrost.batch_encodings(
        [enc0, enc1],
        graph_field_specs={
            "targets": {"dtype": "pyobj", "mode": "ragged_cat"},
            "transition_label": {"dtype": "pyobj", "mode": "stack"},
            "domain_path": {"dtype": "pyobj", "mode": "const"},
        },
    )

    assert batched.targets == [10, 11, 12]
    assert batched.targets_ptr == [0, 2, 3]
    assert batched.transition_label == ["left", "right"]
    assert batched.domain_path == "domain.pddl"
    assert batched.graph_field_specs()["targets"]["mode"] == "ragged_cat"
    assert batched.graph_field_specs()["domain_path"]["dtype"] == "pyobj"


def test_batch_encodings_collates_registered_python_graph_field_specs():
    enc0 = _single_graph_with_stack_field(1.0)
    enc1 = _single_graph_with_stack_field(2.0)

    enc0.targets = ["a0", "a1"]
    enc1.targets = ["b0"]
    enc0.returns = 3.0
    enc1.returns = 4.0

    specs = {
        "targets": {"dtype": "pyobj", "mode": "ragged_cat"},
        "returns": {"mode": "stack"},
    }
    enc0.register_graph_field_specs(specs)
    enc1.register_graph_field_specs(specs)

    batched = mifrost.batch_encodings([enc0, enc1])
    assert batched.targets == ["a0", "a1", "b0"]
    assert batched.targets_ptr == [0, 2, 3]
    assert batched.returns == [3.0, 4.0]
    assert batched.graph_field_specs()["targets"]["dtype"] == "pyobj"
    assert batched.graph_field_specs()["returns"]["mode"] == "stack"


def test_batch_encodings_accepts_legacy_py_field_specs_alias():
    enc0 = _single_graph_with_stack_field(1.0)
    enc1 = _single_graph_with_stack_field(2.0)
    enc0.targets = ["a0"]
    enc1.targets = ["b0", "b1"]

    batched = mifrost.batch_encodings(
        [enc0, enc1],
        py_field_specs={"targets": {"dtype": "pyobj", "mode": "ragged_cat"}},
    )
    assert batched.targets == ["a0", "b0", "b1"]
    assert batched.targets_ptr == [0, 1, 3]


def test_batch_encodings_without_python_attrs_keeps_python_specs_empty():
    enc0 = _single_graph_with_stack_field(1.0)
    enc1 = _single_graph_with_stack_field(2.0)

    batched = mifrost.batch_encodings([enc0, enc1])
    assert batched.graph_field_specs() == {}
    assert "__mifrost_graph_field_specs__" not in batched.__dict__


def test_batch_encodings_mixed_python_attrs_infers_stack_and_pads_missing():
    enc0 = _single_graph_with_stack_field(1.0)
    enc1 = _single_graph_with_stack_field(2.0)

    enc0.transition_label = "left"

    batched = mifrost.batch_encodings([enc0, enc1])
    assert batched.transition_label == ["left", None]
    assert batched.graph_field_specs()["transition_label"]["mode"] == "stack"


def test_batch_encodings_mixed_registered_ragged_spec_handles_missing_values():
    enc0 = _single_graph_with_stack_field(1.0)
    enc1 = _single_graph_with_stack_field(2.0)

    enc0.targets = ["a0", "a1"]
    enc0.register_graph_field_specs(
        {"targets": {"dtype": "pyobj", "mode": "ragged_cat"}}
    )

    batched = mifrost.batch_encodings([enc0, enc1])
    assert batched.targets == ["a0", "a1"]
    assert batched.targets_ptr == [0, 2, 2]
    assert batched.graph_field_specs()["targets"]["mode"] == "ragged_cat"


def test_batch_encodings_mixed_explicit_specs_collate_per_mode():
    enc0 = _single_graph_with_stack_field(1.0)
    enc1 = _single_graph_with_stack_field(2.0)

    enc0.targets = [10, 11]
    enc0.transition_label = "left"
    enc1.transition_label = "right"
    enc0.domain_path = "domain.pddl"
    enc1.domain_path = "domain.pddl"

    batched = mifrost.batch_encodings(
        [enc0, enc1],
        graph_field_specs={
            "targets": {"dtype": "pyobj", "mode": "ragged_cat"},
            "transition_label": {"dtype": "pyobj", "mode": "stack"},
            "domain_path": {"dtype": "pyobj", "mode": "const"},
        },
    )

    assert batched.targets == [10, 11]
    assert batched.targets_ptr == [0, 2, 2]
    assert batched.transition_label == ["left", "right"]
    assert batched.domain_path == "domain.pddl"


def test_batch_encodings_const_python_field_requires_presence_on_all_inputs():
    enc0 = _single_graph_with_stack_field(1.0)
    enc1 = _single_graph_with_stack_field(2.0)
    enc0.domain_path = "domain.pddl"

    with pytest.raises(ValueError, match="missing value for encoding index 1"):
        mifrost.batch_encodings(
            [enc0, enc1],
            graph_field_specs={"domain_path": {"dtype": "pyobj", "mode": "const"}},
        )


def test_batch_encodings_collates_const_tensor_python_field():
    enc0 = _single_graph_with_stack_field(1.0)
    enc1 = _single_graph_with_stack_field(2.0)

    enc0.reward_signature = torch.tensor([1, 2, 3], dtype=torch.int64)
    enc1.reward_signature = torch.tensor([1, 2, 3], dtype=torch.int64)

    batched = mifrost.batch_encodings(
        [enc0, enc1],
        graph_field_specs={"reward_signature": {"dtype": "pyobj", "mode": "const"}},
    )

    assert torch.equal(
        batched.reward_signature, torch.tensor([1, 2, 3], dtype=torch.int64)
    )


def test_batch_encodings_const_tensor_field_requires_exact_structure():
    enc0 = _single_graph_with_stack_field(1.0)
    enc1 = _single_graph_with_stack_field(2.0)

    enc0.reward_signature = torch.tensor([1, 2, 3], dtype=torch.int64)
    enc1.reward_signature = torch.tensor([1, 2, 3], dtype=torch.float32)

    with pytest.raises(ValueError, match="non-constant values"):
        mifrost.batch_encodings(
            [enc0, enc1],
            graph_field_specs={"reward_signature": {"dtype": "pyobj", "mode": "const"}},
        )


def test_batch_encodings_const_numpy_field_requires_exact_structure():
    enc0 = _single_graph_with_stack_field(1.0)
    enc1 = _single_graph_with_stack_field(2.0)

    enc0.reward_signature = np.array([1, 2, 3], dtype=np.int64)
    enc1.reward_signature = np.array([1, 2, 3], dtype=np.float32)

    with pytest.raises(ValueError, match="non-constant values"):
        mifrost.batch_encodings(
            [enc0, enc1],
            graph_field_specs={"reward_signature": {"dtype": "pyobj", "mode": "const"}},
        )


def test_batch_encoding_as_pyg_copies_python_attrs():
    encoding = _single_graph_with_stack_field(3.0)
    encoding.sample_labels = ["label-0"]
    encoding.register_graph_field_specs(
        {"sample_labels": {"dtype": "pyobj", "mode": "stack"}}
    )

    as_single = encoding.as_pyg(as_batch=False)
    as_batch = encoding.as_pyg(as_batch=True)

    assert as_single.sample_labels == "label-0"
    assert as_batch.sample_labels == ["label-0"]
    assert not hasattr(as_single, "__mifrost_graph_field_specs__")
    assert not hasattr(as_batch, "__mifrost_graph_field_specs__")


def test_batch_encoding_graph_field_accessors_and_introspection():
    encoding = _single_graph_with_ragged_i64_field([5, 6])
    encoding.label = "demo"
    encoding.register_graph_field_specs({"label": {"dtype": "pyobj", "mode": "stack"}})

    assert encoding.has_graph_field("target_indices")
    assert encoding.has_graph_field("target_indices_ptr")
    assert not encoding.has_graph_field("missing")

    assert torch.equal(
        encoding.get_graph_field("target_indices"),
        torch.tensor([5, 6], dtype=torch.int64),
    )
    assert torch.equal(
        encoding.get_graph_field("target_indices_ptr"),
        torch.tensor([0, 2], dtype=torch.int64),
    )

    keys = encoding.keys()
    assert "target_indices" in keys
    assert "target_indices_ptr" in keys
    assert "label" in keys
    assert "__mifrost_graph_field_specs__" not in keys

    items = dict(encoding.items())
    assert torch.equal(items["target_indices"], torch.tensor([5, 6], dtype=torch.int64))
    assert torch.equal(
        items["target_indices_ptr"], torch.tensor([0, 2], dtype=torch.int64)
    )
    assert items["label"] == "demo"


def test_batch_encoding_get_graph_field_supports_stack_cat_const_ragged():
    stack = _single_graph_with_stack_field(1.5)
    cat = _single_graph_with_cat_i64_field([2, 4])
    const = _single_graph_with_const_i64_field(42)
    ragged = _single_graph_with_ragged_i64_field([7, 8, 9])

    assert torch.allclose(
        stack.get_graph_field("goal_distance"), torch.tensor([1.5], dtype=torch.float32)
    )
    assert torch.equal(
        cat.get_graph_field("target_concat"), torch.tensor([2, 4], dtype=torch.int64)
    )
    assert torch.equal(
        const.get_graph_field("problem_id"), torch.tensor([42], dtype=torch.int64)
    )
    assert torch.equal(
        ragged.get_graph_field("target_indices"),
        torch.tensor([7, 8, 9], dtype=torch.int64),
    )
    assert torch.equal(
        ragged.get_graph_field("target_indices_ptr"),
        torch.tensor([0, 3], dtype=torch.int64),
    )


def test_batch_encoding_graph_field_introspection_is_stable_across_as_pyg_calls():
    encoding = _single_graph_with_ragged_i64_field([4, 5, 6])

    keys_before = set(encoding.keys())
    items_before = dict(encoding.items())
    assert torch.equal(
        encoding.get_graph_field("target_indices"), items_before["target_indices"]
    )
    assert torch.equal(
        encoding.get_graph_field("target_indices_ptr"),
        items_before["target_indices_ptr"],
    )

    _ = encoding.as_pyg(as_batch=True)
    _ = encoding.as_pyg(as_batch=False)

    keys_after = set(encoding.keys())
    items_after = dict(encoding.items())
    assert keys_after == keys_before
    assert torch.equal(items_after["target_indices"], items_before["target_indices"])
    assert torch.equal(
        items_after["target_indices_ptr"], items_before["target_indices_ptr"]
    )


def test_batch_encoding_as_pyg_native_graph_fields_win_on_python_attr_collision():
    encoding = _single_graph_with_ragged_i64_field([5, 6])
    encoding.__dict__["target_indices"] = ["shadowed"]
    encoding.__dict__["target_indices_ptr"] = [99]

    as_batch = encoding.as_pyg(as_batch=True)
    as_single = encoding.as_pyg(as_batch=False)
    expected_values = torch.tensor([5, 6], dtype=torch.int64)
    expected_ptr = torch.tensor([0, 2], dtype=torch.int64)

    assert torch.equal(as_batch.target_indices, expected_values)
    assert torch.equal(as_batch.target_indices_ptr, expected_ptr)
    assert torch.equal(as_single.target_indices, expected_values)
    assert torch.equal(as_single.target_indices_ptr, expected_ptr)


def test_batch_encoding_native_graph_field_attr_access_and_write_through():
    enc0 = _single_graph_with_ragged_i64_field([1, 2])
    enc1 = _single_graph_with_ragged_i64_field([3])
    batched = mifrost.batch_encodings([enc0, enc1])

    assert hasattr(batched, "target_indices")
    values = batched.target_indices
    assert torch.equal(values, torch.tensor([1, 2, 3], dtype=torch.int64))

    values[0] = 99
    roundtrip = batched.get_graph_field("target_indices")
    assert torch.equal(roundtrip, torch.tensor([99, 2, 3], dtype=torch.int64))


def test_batch_encoding_ragged_ptr_attr_is_read_only_snapshot():
    enc0 = _single_graph_with_ragged_i64_field([1])
    enc1 = _single_graph_with_ragged_i64_field([2, 3])
    batched = mifrost.batch_encodings([enc0, enc1])

    ptr = batched.target_indices_ptr
    ptr[0] = 123
    assert torch.equal(
        batched.get_graph_field("target_indices_ptr"),
        torch.tensor([0, 1, 3], dtype=torch.int64),
    )

    with pytest.raises(ValueError, match="Direct assignment to ragged ptr key"):
        batched.target_indices_ptr = [0, 1, 3]


def test_register_graph_field_specs_raises_on_native_key_collision():
    encoding = _single_graph_with_stack_field(1.0)
    with pytest.raises(ValueError, match="collides with native graph field key"):
        encoding.register_graph_field_specs(
            {"goal_distance": {"dtype": "pyobj", "mode": "stack"}}
        )


def test_batch_encodings_raises_on_python_specs_colliding_with_native_graph_fields():
    enc0 = _single_graph_with_ragged_i64_field([1])
    enc1 = _single_graph_with_ragged_i64_field([2, 3])
    enc0.__dict__["target_indices"] = ["shadow-a"]
    enc1.__dict__["target_indices"] = ["shadow-b"]
    enc0.tag = "a"
    enc1.tag = "b"

    with pytest.raises(ValueError, match="collides with native graph field key"):
        mifrost.batch_encodings(
            [enc0, enc1],
            graph_field_specs={
                "target_indices": {"dtype": "pyobj", "mode": "ragged_cat"},
                "tag": {"dtype": "pyobj", "mode": "stack"},
            },
        )


def test_batch_encoding_consumption_validates_malformed_graph_field_state():
    encoding = _single_graph_with_ragged_i64_field([4, 5])
    state = encoding.__getstate__()
    state["graph_fields"]["target_indices"]["ptr"] = [0]

    malformed = mifrost.BatchEncoding()
    malformed.__setstate__(state)

    with pytest.raises(ValueError, match="Invalid graph field 'target_indices'"):
        malformed.as_pyg(as_batch=True)
    with pytest.raises(ValueError, match="Invalid graph field 'target_indices'"):
        malformed.dumps()
    with pytest.raises(ValueError, match="Invalid graph field 'target_indices'"):
        mifrost.batch_encodings([malformed])

    builder = mifrost.BatchBuilder()
    builder.set_graph_kind("hetero")
    with pytest.raises(ValueError, match="append_batch_encoding invalid graph field"):
        builder.append_batch_encoding(malformed)


def test_map_view_methods_marked_in_core_are_wrapped():
    found_marked = set()
    found_wrapped = set()

    for attr_name in dir(mifrost._core):
        cls = getattr(mifrost._core, attr_name, None)
        if not isinstance(cls, type):
            continue

        marker = getattr(cls, "__mifrost_map_view_methods__", ())
        if isinstance(marker, str):
            marker = (marker,)

        for method_name in marker:
            if not isinstance(method_name, str):
                continue
            method = getattr(cls, method_name, None)
            if not callable(method):
                continue

            key = (cls.__name__, method_name)
            found_marked.add(key)
            if getattr(method, "__mifrost_map_view_wrapped__", False):
                found_wrapped.add(key)

    expected = {
        ("BatchBuilder", "schema_flags_view"),
        ("BatchBuilder", "node_feature_dims_view"),
        ("BatchEncoding", "schema_flags_view"),
        ("BatchEncoding", "node_feature_dims_view"),
        ("Schema", "flags_view"),
    }
    assert expected.issubset(found_marked)
    assert expected.issubset(found_wrapped)


def _assert_tensor_payload_equal(
    lhs: mifrost.BatchEncoding, rhs: mifrost.BatchEncoding
) -> None:
    lhs_tensors = lhs.as_dict()["tensors"]
    rhs_tensors = rhs.as_dict()["tensors"]
    assert lhs_tensors.keys() == rhs_tensors.keys()
    for key in lhs_tensors:
        assert np.array_equal(lhs_tensors[key], rhs_tensors[key]), key


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


def test_append_batch_encoding_legacy_edge_index_schema_fallback():
    src = torch.tensor([0], dtype=torch.int64)
    dst = torch.tensor([1], dtype=torch.int64)

    single = mifrost.BatchBuilder()
    single.set_graph_kind("hetero")
    single.add_node_features("atom", "x", torch.zeros(2, 1))
    single.add_edges("atom", "rel", "atom", src, dst)
    single.next_graph()
    encoding = single.build()

    state = encoding.__getstate__()
    state["schema"]["edge_tensors"] = []

    legacy = mifrost.BatchEncoding()
    legacy.__setstate__(state)

    builder = mifrost.BatchBuilder()
    builder.set_graph_kind("hetero")
    builder.append_batch_encoding(legacy)
    builder.append_batch_encoding(legacy)
    tensors = builder.build().as_dict()["tensors"]

    out_src = torch.as_tensor(tensors["atom|rel|atom/edge_index_0"])
    out_dst = torch.as_tensor(tensors["atom|rel|atom/edge_index_1"])
    assert torch.equal(out_src, torch.tensor([0, 2], dtype=torch.int64))
    assert torch.equal(out_dst, torch.tensor([1, 3], dtype=torch.int64))
