"""Visualization tests for the derived-graph encoder family."""

from __future__ import annotations

import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import networkx as nx  # noqa: E402
import torch  # noqa: E402

try:
    from tests.conftest import load_problem
except ImportError:  # pragma: no cover - wheel-test layout
    from conftest import load_problem  # type: ignore[no-redef]
from mifrost.encoders._derived_visualization import (  # noqa: E402
    EDGE_KIND_NAMES,
    ROLE_NAMES,
)

from mifrost import (  # noqa: E402
    AtomLineGraphEncoder,
    HypergraphIncidenceEncoder,
    ObjectGraphEncoder,
    StarGraphEncoder,
)


@pytest.fixture(scope="module")
def blocks_state():
    _domain, problem, state, _domain_path, _problem_path = load_problem(
        "blocks", "small"
    )
    return problem, state


@pytest.mark.parametrize(
    ("facade", "kwargs"),
    [
        (StarGraphEncoder, {}),
        (ObjectGraphEncoder, {"atom_expansion": "clique"}),
        (ObjectGraphEncoder, {"atom_expansion": "chain"}),
        (AtomLineGraphEncoder, {}),
        (HypergraphIncidenceEncoder, {}),
    ],
)
def test_to_networkx_decodes_roles_and_kinds(blocks_state, facade, kwargs) -> None:
    problem, state = blocks_state
    data = facade(problem, **kwargs).encode_pyg(state)
    encoder = facade(problem, **kwargs)
    graph = encoder.to_networkx(data)
    assert isinstance(graph, nx.MultiDiGraph)
    full = encoder.to_networkx(
        data,
        include_hyperedges=False,
        include_reverse_edges=True,
        include_line_shares=True,
        include_self_loops=True,
    )
    assert full.number_of_nodes() == data.num_nodes
    assert full.number_of_edges() == int(data.edge_index.size(1))
    stripped = encoder.to_networkx(
        data,
        include_hyperedges=False,
        include_reverse_edges=False,
        include_line_shares=False,
        include_self_loops=False,
    )
    assert stripped.number_of_edges() <= full.number_of_edges()
    roles = {attrs["role"] for _, attrs in graph.nodes(data=True)}
    assert roles <= set(ROLE_NAMES)
    kinds = {attrs["kind"] for _, _, attrs in graph.edges(data=True)}
    assert kinds and set(kinds) <= set(EDGE_KIND_NAMES) | {"membership"}


def test_hyperedge_membership_nodes_appear(blocks_state) -> None:
    problem, state = blocks_state
    encoder = HypergraphIncidenceEncoder(problem)
    data = encoder.encode_pyg(state)
    with_membership = encoder.to_networkx(data)
    without = encoder.to_networkx(data, include_hyperedges=False)
    membership_count = sum(
        1
        for _, _, attrs in with_membership.edges(data=True)
        if attrs["kind"] == "membership"
    )
    assert membership_count == int(data.hyperedge_index.size(1))
    assert without.number_of_nodes() == data.num_nodes


def test_draw_renders_all_facades(blocks_state) -> None:
    problem, state = blocks_state
    encoders = [
        StarGraphEncoder(problem),
        ObjectGraphEncoder(problem),
        AtomLineGraphEncoder(problem),
        HypergraphIncidenceEncoder(problem),
    ]
    figure, axes = plt.subplots(1, len(encoders), figsize=(6 * len(encoders), 5))
    for encoder, ax in zip(encoders, axes, strict=True):
        returned = encoder.draw(encoder.encode_pyg(state), ax=ax)
        assert returned is ax
    figure.tight_layout()
    figure.savefig("/tmp/mifrost_derived_test_render.png", dpi=72)
    plt.close(figure)


def test_summarize_reports_histograms(blocks_state) -> None:
    problem, state = blocks_state
    data = StarGraphEncoder(problem).encode_pyg(state)
    summary = StarGraphEncoder(problem).summarize(data)
    assert "roles:" in summary and "kinds:" in summary
    assert torch.is_tensor(data.edge_attr)
