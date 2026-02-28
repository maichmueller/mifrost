from __future__ import annotations

import time

import numpy as np
import pytest
import torch

import mifrost
from mifrost.encoders import ColorEncoder, HGraphEncoder


def _assert_tensor_equal(actual: torch.Tensor, expected: torch.Tensor) -> None:
    assert torch.equal(actual, expected)


def _non_cpu_device() -> torch.device | None:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return None


def _hetero_encoding_with_missing_x_column() -> mifrost.BatchEncoding:
    builder = mifrost.BatchBuilder()
    builder.set_graph_kind("hetero")
    builder.add_node_features("src", "x", torch.zeros(2, 1))
    builder.add_node_features("dst", "x", torch.zeros(2, 1))
    builder.add_edges(
        "src",
        "rel",
        "dst",
        torch.tensor([0, 1], dtype=torch.int64),
        torch.tensor([1, 0], dtype=torch.int64),
    )
    builder.next_graph()
    encoding = builder.build()
    state = encoding.__getstate__()
    state["columns"].pop("src/x")
    out = mifrost.BatchEncoding()
    out.__setstate__(state)
    return out


def _homo_encoding_with_missing_x_column() -> mifrost.BatchEncoding:
    builder = mifrost.BatchBuilder()
    builder.set_graph_kind("homo")
    builder.add_node_features("atom", "x", torch.zeros(2, 1))
    builder.add_edges(
        "atom",
        "rel",
        "atom",
        torch.tensor([0, 1], dtype=torch.int64),
        torch.tensor([1, 0], dtype=torch.int64),
    )
    builder.next_graph()
    encoding = builder.build()
    state = encoding.__getstate__()
    state["columns"].pop("atom/x")
    out = mifrost.BatchEncoding()
    out.__setstate__(state)
    return out


def _expected_edge_attr_dict_from_hetero_batch(
    data,
) -> dict[tuple[str, str, str], torch.Tensor]:
    out: dict[tuple[str, str, str], torch.Tensor] = {}
    for edge_type in data.edge_types:
        store = data[edge_type]
        if "edge_attr" in store:
            out[edge_type] = store.edge_attr
    return out


def test_as_hetero_parity_with_as_pyg_batch(small_blocks):
    space, domain, problem = small_blocks
    encoder = HGraphEncoder(domain)
    states = [
        problem.get_initial_state(),
        space._advanced_state_space_sampler.sample_state_n_steps_from_goal(0),
    ]
    encoding = encoder.encode_batch(states)

    view = encoding.as_hetero()
    batch = encoding.as_pyg(as_batch=True)

    assert view.graph_kind == "hetero"
    assert view.num_graphs == encoding.num_graphs
    assert view.num_nodes == encoding.num_nodes
    assert view.num_edges == encoding.num_edges
    assert list(view.object_names) == list(encoding.as_dict().get("object_names", []))
    assert set(view.node_types) == set(batch.node_types)
    assert set(view.edge_types) == set(batch.edge_types)

    assert set(view.x_dict.keys()) == set(batch.x_dict.keys())
    for node_type, value in view.x_dict.items():
        _assert_tensor_equal(value, batch.x_dict[node_type])

    assert set(view.edge_index_dict.keys()) == set(batch.edge_index_dict.keys())
    for edge_type, value in view.edge_index_dict.items():
        _assert_tensor_equal(value, batch.edge_index_dict[edge_type])

    expected_edge_attr_dict = _expected_edge_attr_dict_from_hetero_batch(batch)
    assert set(view.edge_attr_dict.keys()) == set(expected_edge_attr_dict.keys())
    for edge_type, value in view.edge_attr_dict.items():
        _assert_tensor_equal(value, expected_edge_attr_dict[edge_type])

    expected_batch_dict = {
        node_type: batch[node_type].batch for node_type in batch.node_types
    }
    expected_ptr_dict = {
        node_type: batch[node_type].ptr for node_type in batch.node_types
    }
    assert set(view.batch_dict.keys()) == set(expected_batch_dict.keys())
    assert set(view.ptr_dict.keys()) == set(expected_ptr_dict.keys())
    for node_type, value in view.batch_dict.items():
        _assert_tensor_equal(value, expected_batch_dict[node_type])
    for node_type, value in view.ptr_dict.items():
        _assert_tensor_equal(value, expected_ptr_dict[node_type])


def test_as_homo_parity_with_as_pyg_batch(small_blocks):
    _space, domain, problem = small_blocks
    encoder = ColorEncoder(domain)
    states = [problem.get_initial_state(), problem.get_initial_state()]
    encoding = encoder.encode_batch(states)

    view = encoding.as_homo()
    batch = encoding.as_pyg(as_batch=True)
    assert not hasattr(batch, "node_types")

    assert view.graph_kind == "homo"
    assert view.num_graphs == encoding.num_graphs
    assert view.num_nodes == encoding.num_nodes
    assert view.num_edges == encoding.num_edges
    assert list(view.object_names) == list(encoding.as_dict().get("object_names", []))

    assert view.x is not None
    assert view.edge_index is not None
    assert view.batch is not None
    assert view.ptr is not None
    edge_type = view.edge_types[0] if view.edge_types else None
    _assert_tensor_equal(view.x, batch.x)
    if edge_type is not None:
        _assert_tensor_equal(view.edge_index, batch.edge_index)
    _assert_tensor_equal(view.batch, batch.batch)
    _assert_tensor_equal(view.ptr, batch.ptr)
    if edge_type is None or getattr(batch, "edge_attr", None) is None:
        assert view.edge_attr is None
    else:
        assert view.edge_attr is not None
        _assert_tensor_equal(view.edge_attr, batch.edge_attr)


def test_facades_expose_underlying_batch_encoding(small_blocks):
    space, domain, problem = small_blocks

    hetero_encoder = HGraphEncoder(domain)
    hetero_encoding = hetero_encoder.encode_batch(
        [
            problem.get_initial_state(),
            space._advanced_state_space_sampler.sample_state_n_steps_from_goal(0),
        ]
    )
    hetero_view = hetero_encoding.as_hetero()
    assert hetero_view.base is hetero_encoding

    homo_encoder = ColorEncoder(domain)
    homo_encoding = homo_encoder.encode_batch(
        [problem.get_initial_state(), problem.get_initial_state()]
    )
    homo_view = homo_encoding.as_homo()
    assert homo_view.base is homo_encoding


def test_facade_kind_gates_raise_on_mismatch(small_blocks):
    _space, domain, problem = small_blocks
    hetero_encoding = HGraphEncoder(domain).encode(problem.get_initial_state())
    homo_encoding = ColorEncoder(domain).encode(problem.get_initial_state())

    with pytest.raises(ValueError, match="expected 'hetero'"):
        homo_encoding.as_hetero()
    with pytest.raises(ValueError, match="expected 'homo'"):
        hetero_encoding.as_homo()


def test_as_homo_rejects_multi_type_schema():
    builder = mifrost.BatchBuilder()
    builder.set_graph_kind("homo")
    builder.add_node_features("a", "x", np.zeros((1, 1), dtype=np.float32))
    builder.add_node_features("b", "x", np.zeros((1, 1), dtype=np.float32))
    builder.next_graph()
    encoding = builder.build()

    with pytest.raises(
        ValueError, match="expects schema with at most one node type and one edge type"
    ):
        encoding.as_homo()


def test_facade_properties_are_cached_and_avoid_as_pyg(small_blocks):
    space, domain, problem = small_blocks
    encoder = HGraphEncoder(domain)
    states = [
        problem.get_initial_state(),
        space._advanced_state_space_sampler.sample_state_n_steps_from_goal(0),
    ]
    encoding = encoder.encode_batch(states)

    # If facade properties route through as_pyg/as_dict, this will fail.
    encoding.as_pyg = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("as_pyg should not be used by facades")
    )
    encoding.as_dict = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("as_dict should not be used by facades")
    )

    view = encoding.as_hetero()
    x1 = view.x_dict
    x2 = view.x_dict
    e1 = view.edge_index_dict
    e2 = view.edge_index_dict
    b1 = view.batch_dict
    b2 = view.batch_dict

    assert x1 is x2
    assert e1 is e2
    assert b1 is b2


def test_hetero_facade_to_cpu_is_eager_and_in_place(small_blocks):
    space, domain, problem = small_blocks
    encoder = HGraphEncoder(domain)
    states = [
        problem.get_initial_state(),
        space._advanced_state_space_sampler.sample_state_n_steps_from_goal(0),
    ]
    encoding = encoder.encode_batch(states)
    view = encoding.as_hetero()

    out = view.to(torch.device("cpu"))
    assert out is view

    cache = encoding.__dict__.get("__mifrost_tensor_cache__")
    assert isinstance(cache, dict)
    assert len(cache) > 0

    for tensor in view.x_dict.values():
        assert tensor.device.type == "cpu"
    for tensor in view.edge_index_dict.values():
        assert tensor.device.type == "cpu"
    for tensor in view.batch_dict.values():
        assert tensor.device.type == "cpu"
    for tensor in view.ptr_dict.values():
        assert tensor.device.type == "cpu"
    for tensor in view.edge_attr_dict.values():
        assert tensor.device.type == "cpu"


def test_homo_facade_to_cpu_is_eager_and_in_place(small_blocks):
    _space, domain, problem = small_blocks
    encoder = ColorEncoder(domain)
    states = [problem.get_initial_state(), problem.get_initial_state()]
    encoding = encoder.encode_batch(states)
    view = encoding.as_homo()

    out = view.to(torch.device("cpu"))
    assert out is view

    cache = encoding.__dict__.get("__mifrost_tensor_cache__")
    assert isinstance(cache, dict)
    assert len(cache) > 0

    if view.x is not None:
        assert view.x.device.type == "cpu"
    if view.edge_index is not None:
        assert view.edge_index.device.type == "cpu"
    if view.batch is not None:
        assert view.batch.device.type == "cpu"
    if view.ptr is not None:
        assert view.ptr.device.type == "cpu"
    if view.edge_attr is not None:
        assert view.edge_attr.device.type == "cpu"


def test_hetero_facade_to_non_cpu_moves_fallback_x_with_edge_index():
    device = _non_cpu_device()
    if device is None:
        pytest.skip("No non-CPU device available")

    encoding = _hetero_encoding_with_missing_x_column()
    view = encoding.as_hetero().to(device)
    edge_type = ("src", "rel", "dst")

    assert view.x_dict["src"].device.type == device.type
    assert view.x_dict["dst"].device.type == device.type
    assert view.edge_index_dict[edge_type].device.type == device.type

    if device.type == "cuda":
        from torch_geometric.nn import SimpleConv

        conv = SimpleConv()
        out = conv(
            (view.x_dict["src"], view.x_dict["dst"]), view.edge_index_dict[edge_type]
        )
        assert out.device.type == "cuda"


def test_homo_facade_to_non_cpu_moves_fallback_x_with_edge_index():
    device = _non_cpu_device()
    if device is None:
        pytest.skip("No non-CPU device available")

    encoding = _homo_encoding_with_missing_x_column()
    view = encoding.as_homo().to(device)

    assert view.x is not None
    assert view.edge_index is not None
    assert view.x.device.type == device.type
    assert view.edge_index.device.type == device.type

    if device.type == "cuda":
        from torch_geometric.nn import SimpleConv

        conv = SimpleConv()
        out = conv(view.x, view.edge_index)
        assert out.device.type == "cuda"


def test_facade_access_is_faster_than_repeated_as_pyg(small_blocks):
    space, domain, problem = small_blocks
    encoder = HGraphEncoder(domain)
    states = [
        problem.get_initial_state(),
        space._advanced_state_space_sampler.sample_state_n_steps_from_goal(0),
        space._advanced_state_space_sampler.sample_state_n_steps_from_goal(1),
        space._advanced_state_space_sampler.sample_state_n_steps_from_goal(2),
    ]
    encoding = encoder.encode_batch(states)

    view = encoding.as_hetero()
    # Warm up lazy caches.
    _ = view.x_dict
    _ = view.edge_index_dict
    _ = view.batch_dict

    t0 = time.perf_counter()
    for _ in range(200):
        _ = view.x_dict
        _ = view.edge_index_dict
        _ = view.batch_dict
    facade_elapsed = time.perf_counter() - t0

    t1 = time.perf_counter()
    for _ in range(20):
        _ = encoding.as_pyg(as_batch=True)
    pyg_elapsed = time.perf_counter() - t1

    assert facade_elapsed < pyg_elapsed
