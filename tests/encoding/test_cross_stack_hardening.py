"""Hardening tests for the cross-stack export adapters (review round two).

Covers input validation (PyG ``Batch`` rejection) and the
``hyperedge_bipartite`` construction in :func:`to_dgl`: the hyperedge
count must be derived from ``hyperedge_index[1]`` (max + 1), never from
attribute lengths, and an empty membership tensor must yield an empty
bipartite graph instead of crashing.

The bipartite-math tests run against a recording ``dql``/dgl stand-in so
they exercise the real adapter code even where the real DGL wheel cannot
import; one integration test mirrors the true-DGL behaviour when it is
available.
"""

from __future__ import annotations

import sys
from typing import Any

import pytest

torch = pytest.importorskip("torch")

from torch_geometric.data import Batch, Data  # noqa: E402

from mifrost.encoders.cross_stack import to_dgl, to_jraph  # noqa: E402


def _carrier(hyperedge_index: Any = None, *, num_nodes: int = 3) -> Data:
    """A minimal derived-style PyG carrier."""

    data = Data(
        x_ids=torch.arange(num_nodes, dtype=torch.float32).reshape(-1, 1).repeat(1, 6),
        edge_index=torch.zeros((2, 0), dtype=torch.long),
        edge_attr=torch.zeros((0, 3)),
    )
    data.num_nodes = num_nodes
    if hyperedge_index is not None:
        data.hyperedge_index = hyperedge_index
    return data


class _FakeDGLGraph:
    def __init__(self) -> None:
        self.ndata: dict[str, Any] = {}
        self.edata: dict[str, Any] = {}
        self.edge_pairs: Any = None
        self.node_count: int | None = None
        self.idtype: Any = None


class _FakeDGL:
    """Records ``graph(...)`` calls; stands in for the real library."""

    def __init__(self) -> None:
        self.graphs: list[_FakeDGLGraph] = []

    def graph(self, edges: Any, **kwargs: Any) -> _FakeDGLGraph:
        record = _FakeDGLGraph()
        record.edge_pairs = edges
        record.node_count = int(kwargs["num_nodes"])
        record.idtype = kwargs.get("idtype")
        self.graphs.append(record)
        return record


@pytest.fixture
def fake_dgl(monkeypatch: pytest.MonkeyPatch) -> _FakeDGL:
    fake = _FakeDGL()
    monkeypatch.setitem(sys.modules, "dgl", fake)
    return fake


class TestBatchRejection:
    def test_to_dgl_rejects_batch(self) -> None:
        batch = Batch.from_data_list([_carrier()])
        with pytest.raises(
            TypeError, match=r"call \.to_data_list\(\) or index the batch first"
        ):
            to_dgl(batch)

    def test_to_jraph_rejects_batch(self) -> None:
        batch = Batch.from_data_list([_carrier()])
        with pytest.raises(
            TypeError, match=r"call \.to_data_list\(\) or index the batch first"
        ):
            to_jraph(batch)


class TestHyperedgeBipartiteMath:
    def test_hyperedge_count_derived_from_row_one_max(self, fake_dgl) -> None:
        # Anchors use ids {0, 5}: six hyperedges exist (max + 1) even though
        # only three membership entries and two distinct ids appear.
        index = torch.tensor([[0, 1, 2], [0, 5, 5]])
        graph, metadata = to_dgl(_carrier(index))

        bipartite = metadata["hyperedge_bipartite"]
        assert bipartite is fake_dgl.graphs[-1]
        assert bipartite.node_count == 3 + 6  # num_nodes + (max anchor + 1)
        members, anchors = bipartite.edge_pairs
        assert torch.equal(members, index[0])
        assert torch.equal(anchors, index[1] + 3)
        # The main graph is unaffected.
        assert fake_dgl.graphs[0].node_count == 3

    def test_contiguous_membership_matches_membership_width(self, fake_dgl) -> None:
        index = torch.tensor([[0, 1], [0, 1]])
        _graph, metadata = to_dgl(_carrier(index))
        assert metadata["hyperedge_bipartite"].node_count == 3 + 2

    def test_empty_membership_yields_empty_entry(self, fake_dgl) -> None:
        index = torch.empty((2, 0), dtype=torch.long)
        graph, metadata = to_dgl(_carrier(index))

        bipartite = metadata["hyperedge_bipartite"]
        assert bipartite.node_count == 3
        members, anchors = bipartite.edge_pairs
        assert members.numel() == 0 and anchors.numel() == 0
        assert graph is fake_dgl.graphs[0]

    def test_no_membership_attr_skips_bipartite(self, fake_dgl) -> None:
        _graph, metadata = to_dgl(_carrier())
        assert "hyperedge_bipartite" not in metadata
        assert len(fake_dgl.graphs) == 1


class TestRealDglIntegration:
    @staticmethod
    def _import_or_skip_dgl() -> None:
        """Skip cleanly when the DGL wheel is missing *or* broken.

        Broken wheels can raise non-ImportError exceptions mid-import
        (e.g. graphbolt dylib mismatches raising ``OSError`` subclasses),
        which :func:`pytest.importorskip` does not catch.
        """
        try:
            pytest.importorskip("dgl")
        except Exception as exc:  # noqa: BLE001 - any import failure skips
            pytest.skip(f"dgl is present but not importable here: {exc}")

    def test_bipartite_structure_with_real_dgl(self) -> None:
        self._import_or_skip_dgl()
        index = torch.tensor([[0, 1, 2], [0, 5, 5]])
        graph, metadata = to_dgl(_carrier(index))

        bipartite = metadata["hyperedge_bipartite"]
        assert bipartite.num_nodes() == 9
        src, dst = bipartite.edges()
        assert torch.equal(src, index[0])
        assert torch.equal(dst, index[1] + 3)
        assert graph.num_edges() == 0
