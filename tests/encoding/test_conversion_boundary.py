from __future__ import annotations

import torch

import mifrost
import mifrost.encoders as encoders

from .test_utils import as_pyg, hetero_data_equal


def _conversion():
    from mifrost.encoders import conversion

    return conversion


def _to_torch(value):
    if torch.is_tensor(value):
        return value
    return torch.utils.dlpack.from_dlpack(value)


def _assert_tensor_dict_equal(actual, expected) -> None:
    assert set(actual) == set(expected)
    for key in actual:
        assert torch.equal(actual[key], expected[key])


def _assert_metadata_matches(actual, expected) -> None:
    assert getattr(actual, "object_names", None) == getattr(
        expected, "object_names", None
    )
    for node_type in actual.node_types:
        assert list(getattr(actual[node_type], "node_names", [])) == list(
            getattr(expected[node_type], "node_names", [])
        )


def _homo_dict_payload(*, with_edges: bool) -> dict:
    """Build a small schema/payload pair without going through native code."""

    tensors = {
        "node/x": torch.tensor([[1.0], [2.0], [3.0]]),
        "node/ptr": torch.tensor([0, 2, 3], dtype=torch.long),
        "node/batch": torch.tensor([0, 0, 1], dtype=torch.long),
    }
    edge_tensors = [
        {
            "edge_type": 0,
            "attr": "edge_index",
            "part": "0",
            "key": "node|edge|node/edge_index_0",
        },
        {
            "edge_type": 0,
            "attr": "edge_index",
            "part": "1",
            "key": "node|edge|node/edge_index_1",
        },
        {
            "edge_type": 0,
            "attr": "edge_attr",
            "key": "node|edge|node/edge_attr",
        },
    ]
    if with_edges:
        tensors.update(
            {
                "node|edge|node/edge_index_0": torch.tensor([0, 2]),
                "node|edge|node/edge_index_1": torch.tensor([2, 1]),
                "node|edge|node/edge_attr": torch.tensor(
                    [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]
                ),
            }
        )
    return {
        "tensors": tensors,
        "schema": {
            "version": 1,
            "graph_kind": "homo",
            "node_types": ["node"],
            "edge_types": [{"src": "node", "rel": "edge", "dst": "node"}],
            "node_tensors": [
                {"node_type": "node", "attr": "x", "key": "node/x"},
                {"node_type": "node", "attr": "ptr", "key": "node/ptr"},
                {"node_type": "node", "attr": "batch", "key": "node/batch"},
            ],
            "edge_tensors": edge_tensors,
            "flags": {"include_reverse_edges": False},
            "extensions": {},
        },
        "node_names": {"node": ["a", "b", "c"]},
        "object_names": ["a", "b", "c"],
        "graph_attrs": {"vocab_categories": ["static", "fluent"]},
        "num_graphs": 2,
    }


def test_dict_homo_conversion_honors_metadata_and_explicit_reverse_flag():
    conversion = _conversion()
    data = conversion.to_pyg(_homo_dict_payload(with_edges=True), as_batch=True)

    # The schema explicitly owns edge direction; False must not trigger a
    # second reverse expansion in the Python dictionary path.
    assert data.edge_index.tolist() == [[0, 2], [2, 1]]
    assert data.node_names == [["a", "b"], ["c"]]
    assert data.object_names == [["a", "b"], ["c"]]
    assert data.vocab_categories == ["static", "fluent"]


def test_dict_homo_conversion_supplies_rank_stable_empty_edges():
    conversion = _conversion()
    data = conversion.to_pyg(_homo_dict_payload(with_edges=False), as_batch=False)

    assert data.edge_index.shape == (2, 0)
    assert data.edge_attr.shape == (0, 3)


def test_to_pyg_matches_native_batch_encoding_as_batch(small_blocks):
    space, domain, problem = small_blocks
    encoder = mifrost.HGraphEncoder(domain)

    state0 = problem.get_initial_state()
    state1 = space._advanced_state_space_sampler.sample_state_n_steps_from_goal(0)
    encoding = encoder.encode_batch([state0, state1])
    conversion = _conversion()

    actual = conversion.to_pyg(encoding, as_batch=True)
    expected = as_pyg(encoding, as_batch=True)

    assert hetero_data_equal(actual, expected)
    _assert_metadata_matches(actual, expected)


def test_to_pyg_dict_without_metadata_omits_node_and_object_names(small_blocks):
    space, domain, problem = small_blocks
    encoder = mifrost.HGraphEncoder(domain)

    state0 = problem.get_initial_state()
    state1 = space._advanced_state_space_sampler.sample_state_n_steps_from_goal(0)
    encoding = encoder.encode_batch([state0, state1])
    conversion = _conversion()

    actual = conversion.to_pyg(
        encoding.as_dict(), as_batch=True, include_metadata=False
    )

    for node_type in actual.node_types:
        assert "node_names" not in actual[node_type]
    assert not hasattr(actual, "object_names")


def test_to_tensor_payload_normalizes_keys_and_tensors(small_blocks):
    _space, domain, problem = small_blocks
    encoder = mifrost.HGraphEncoder(domain)

    encoding = encoder.encode(problem.get_initial_state())
    conversion = _conversion()
    payload = encoding.as_dict()

    actual = conversion.to_tensor_payload(payload)
    expected_payload = encoding.as_dict()["tensors"]

    assert actual
    assert all(isinstance(key, str) for key in actual)
    assert all(torch.is_tensor(value) for value in actual.values())
    assert set(actual) == {str(key) for key in expected_payload}
    for key, value in actual.items():
        assert torch.equal(value, _to_torch(expected_payload[key]))


def test_legacy_encoder_namespace_conversion_exports_remain_callable(small_blocks):
    conversion = _conversion()
    _space, domain, problem = small_blocks
    encoder = mifrost.HGraphEncoder(domain)
    encoding = encoder.encode(problem.get_initial_state())
    payload0 = encoding.as_dict()
    payload1 = encoding.as_dict()
    payload2 = encoding.as_dict()

    assert hetero_data_equal(
        encoders._encoding_dict_to_pyg(payload0, as_batch=True),
        conversion.to_pyg(payload1, as_batch=True),
    )
    _assert_tensor_dict_equal(
        encoders.encoding_to_tensors(payload2),
        conversion.to_tensor_payload(encoding.as_dict()),
    )
