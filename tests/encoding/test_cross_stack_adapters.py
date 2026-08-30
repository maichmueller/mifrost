"""Structural round-trip tests for the DGL / Jraph export adapters.

Both target stacks are optional dependencies. Each test class skips
cleanly (with an install hint) when its library is missing or its wheel is
broken; the adapters themselves guarantee structural equality only — the
deep behavioural contract stays with the PyG conformance suite.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from mifrost.encoders import cross_stack
from mifrost.encoders.cross_stack import to_dgl, to_jraph
from mifrost.encoders.derived import (
    HypergraphIncidenceEncoder,
    StarGraphEncoder,
    TransformerBiasEncoder,
    TupleTensorEncoder,
)
from mifrost.encoders.object_feature import ObjectFeatureEncoder

try:
    from tests.conftest import load_problem
except ImportError:  # pragma: no cover - wheel test layouts
    from conftest import load_problem  # type: ignore[no-redef]


def _import_or_skip(module_name: str, install_hint: str):
    """Import ``module_name`` or skip cleanly with an install hint.

    Delegates the plain-missing case to :func:`pytest.importorskip` and
    additionally skips when a wheel that is present fails mid-import with a
    non-ImportError (e.g. DGL's graphbolt dylib mismatching newer torch
    releases raises bare ``OSError`` subclasses).
    """
    try:
        return pytest.importorskip(module_name)
    except pytest.skip.Exception:
        raise
    except Exception as exc:
        pytest.skip(
            f"{module_name} is present but not importable here "
            f"({type(exc).__name__}: {exc}); install hint: {install_hint}"
        )


@pytest.fixture(scope="module")
def blocks_small():
    _domain, problem, state, _domain_path, _problem_path = load_problem(
        "blocks", "small"
    )
    return problem, state


def test_metadata_collector_includes_categories_and_object_names() -> None:
    data = SimpleNamespace(
        vocab_categories=["static", "fluent"], object_names=["a", "b"]
    )

    assert cross_stack._collect_metadata(data) == {
        "vocab_categories": ["static", "fluent"],
        "object_names": ["a", "b"],
    }


class TestToDgl:
    def test_star_round_trip_structure(self, blocks_small) -> None:
        _import_or_skip("dgl", "pip install dgl")
        import torch

        problem, state = blocks_small
        data = StarGraphEncoder(problem).encode_pyg(state)
        graph, metadata = to_dgl(data)

        assert graph.idtype == torch.int64
        assert graph.device == torch.device("cpu")
        assert graph.num_nodes() == data.num_nodes
        assert graph.num_edges() == data.edge_index.size(1)

        src, dst = graph.edges()
        assert torch.equal(src, data.edge_index[0].long())
        assert torch.equal(dst, data.edge_index[1].long())

        assert torch.equal(graph.ndata["x_ids"], data.x_ids)
        assert torch.equal(graph.edata["edge_attr"], data.edge_attr)
        role_ids = graph.ndata["role_ids"]
        assert role_ids.dtype == torch.int64
        assert torch.equal(role_ids, data.x_ids[:, 0].long())

    def test_metadata_vocabs_match_data_attrs(self, blocks_small) -> None:
        _import_or_skip("dgl", "pip install dgl")
        import torch

        problem, state = blocks_small
        data = StarGraphEncoder(problem).encode_pyg(state)
        _graph, metadata = to_dgl(data)

        for attr in (
            "vocab_roles",
            "vocab_categories",
            "vocab_predicates",
            "vocab_edge_kinds",
            "channel_names",
            "edge_channel_names",
            "node_names",
            "object_names",
        ):
            expected = getattr(data, attr)
            assert metadata[attr] == expected
        assert set(metadata) == {
            "vocab_roles",
            "vocab_categories",
            "vocab_predicates",
            "vocab_edge_kinds",
            "channel_names",
            "edge_channel_names",
            "node_names",
            "object_names",
        }
        assert all(not isinstance(value, torch.Tensor) for value in metadata.values())

    def test_hyper_facade_bipartite_membership(self, blocks_small) -> None:
        _import_or_skip("dgl", "pip install dgl")
        import torch

        problem, state = blocks_small
        data = HypergraphIncidenceEncoder(problem).encode_pyg(state)
        graph, metadata = to_dgl(data)

        hyperedge_index = data.hyperedge_index.long()
        bipartite = metadata["hyperedge_bipartite"]
        num_anchors = int(hyperedge_index[1].max().item()) + 1
        assert bipartite.num_nodes() == data.num_nodes + num_anchors
        assert bipartite.num_edges() == hyperedge_index.size(1)

        src, dst = bipartite.edges()
        assert torch.equal(src, hyperedge_index[0])
        assert torch.equal(dst, hyperedge_index[1] + data.num_nodes)
        assert torch.equal(metadata["hyperedge_index"], hyperedge_index)
        # The main graph still carries every object/atom edge unchanged.
        assert graph.num_edges() == data.edge_index.size(1)

    def test_tuple_and_spd_pass_through(self, blocks_small) -> None:
        _import_or_skip("dgl", "pip install dgl")
        import torch

        problem, state = blocks_small
        tuple_data = TupleTensorEncoder(problem).encode_pyg(state)
        _graph, tuple_meta = to_dgl(tuple_data)
        for attr in ("tuple_args", "tuple_ptr", "tuple_rel_ids", "tuple_role_ids"):
            assert isinstance(tuple_meta[attr], torch.Tensor)
            assert torch.equal(tuple_meta[attr], getattr(tuple_data, attr))

        spd_data = TransformerBiasEncoder(problem).encode_pyg(state)
        _graph, spd_meta = to_dgl(spd_data)
        for attr in ("spd_src", "spd_dst", "spd_dist"):
            assert isinstance(spd_meta[attr], torch.Tensor)
            assert torch.equal(spd_meta[attr], getattr(spd_data, attr))

    def test_facade_delegation_matches_module_function(self, blocks_small) -> None:
        _import_or_skip("dgl", "pip install dgl")

        problem, state = blocks_small
        encoder = StarGraphEncoder(problem)
        data = encoder.encode_pyg(state)
        via_facade = encoder.to_dgl(data)
        via_module = to_dgl(data)

        assert via_facade[0].num_nodes() == via_module[0].num_nodes()
        assert via_facade[0].num_edges() == via_module[0].num_edges()
        assert via_facade[1].keys() == via_module[1].keys()


class TestToJraph:
    def test_star_round_trip_structure(self, blocks_small) -> None:
        jraph = _import_or_skip("jraph", "pip install jraph jax")

        problem, state = blocks_small
        data = StarGraphEncoder(problem).encode_pyg(state)
        graphs_tuple, metadata = to_jraph(data)
        assert isinstance(graphs_tuple, jraph.GraphsTuple)

        assert graphs_tuple.n_node.tolist() == [data.num_nodes]
        assert graphs_tuple.n_edge.tolist() == [data.edge_index.size(1)]
        assert sum(int(entry) for entry in graphs_tuple.n_node) == data.num_nodes
        assert sum(int(entry) for entry in graphs_tuple.n_edge) == data.edge_index.size(
            1
        )

    def test_senders_receivers_and_channels_match(self, blocks_small) -> None:
        _import_or_skip("jraph", "pip install jraph jax")
        import numpy as np

        problem, state = blocks_small
        data = StarGraphEncoder(problem).encode_pyg(state)
        graphs_tuple, _metadata = to_jraph(data)

        np.testing.assert_array_equal(
            np.asarray(graphs_tuple.senders), data.edge_index[0].numpy()
        )
        np.testing.assert_array_equal(
            np.asarray(graphs_tuple.receivers), data.edge_index[1].numpy()
        )
        np.testing.assert_allclose(np.asarray(graphs_tuple.nodes), data.x_ids.numpy())
        np.testing.assert_allclose(
            np.asarray(graphs_tuple.edges), data.edge_attr.numpy()
        )

    def test_globals_and_metadata_carry_vocabs_and_extras(self, blocks_small) -> None:
        _import_or_skip("jraph", "pip install jraph jax")

        problem, state = blocks_small
        data = HypergraphIncidenceEncoder(problem).encode_pyg(state)
        graphs_tuple, metadata = to_jraph(data)

        for attr in (
            "vocab_roles",
            "vocab_categories",
            "vocab_predicates",
            "vocab_edge_kinds",
            "channel_names",
            "edge_channel_names",
            "node_names",
            "object_names",
        ):
            assert graphs_tuple.globals[attr] == getattr(data, attr)
            assert metadata[attr] == getattr(data, attr)
        assert metadata["hyperedge_index"] is graphs_tuple.globals["hyperedge_index"]

    def test_object_feature_encoder_output_converts(self, blocks_small) -> None:
        _import_or_skip("jraph", "pip install jraph jax")

        problem, state = blocks_small
        data = ObjectFeatureEncoder(problem).encode_pyg(state)
        graphs_tuple, metadata = to_jraph(data)

        assert graphs_tuple.n_node.tolist() == [data.num_nodes]
        assert graphs_tuple.n_edge.tolist() == [data.edge_index.size(1)]
        assert "vocab_roles" not in metadata
        assert metadata["vocab_predicates"] == data.vocab_predicates
        assert metadata["node_names"] == data.node_names
        assert metadata["object_names"] == data.object_names

    def test_graph_convolution_applies(self, blocks_small) -> None:
        jax = _import_or_skip("jax", "pip install jax")
        jraph = _import_or_skip("jraph", "pip install jraph jax")

        problem, state = blocks_small
        data = StarGraphEncoder(problem).encode_pyg(state)
        graphs_tuple, _metadata = to_jraph(data)

        conv = jraph.GraphConvolution(update_node_fn=jax.nn.relu)
        out = conv(graphs_tuple._replace(globals=None))
        assert out.nodes.shape[0] == data.num_nodes

    def test_facade_delegation_matches_module_function(self, blocks_small) -> None:
        _import_or_skip("jraph", "pip install jraph jax")

        problem, state = blocks_small
        encoder = StarGraphEncoder(problem)
        data = encoder.encode_pyg(state)
        via_facade = encoder.to_jraph(data)
        via_module = to_jraph(data)

        assert via_facade[0].n_node.tolist() == via_module[0].n_node.tolist()
        assert via_facade[0].n_edge.tolist() == via_module[0].n_edge.tolist()
        assert via_facade[1].keys() == via_module[1].keys()


class TestInstallHints:
    """The lazy imports must fail with actionable hints when stacks are absent."""

    @pytest.mark.parametrize(
        "function_name,hint", [("to_dgl", "dgl"), ("to_jraph", "jraph")]
    )
    def test_missing_stack_raises_import_error_with_hint(
        self, monkeypatch, function_name, hint
    ) -> None:
        from mifrost.encoders import cross_stack

        monkeypatch.setitem(sys.modules, "dgl", None)
        monkeypatch.setitem(sys.modules, "jraph", None)
        converter = getattr(cross_stack, function_name)
        with pytest.raises(ImportError, match=f"pip install.*{hint}"):
            converter(object())
