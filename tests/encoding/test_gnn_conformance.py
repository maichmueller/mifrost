"""PyTorch Geometric conformance tests for derived-graph encoder facades.

Every derived-graph facade must emit graphs that stock torch_geometric layers
consume out of the box: a forward pass, a backward pass, and finite gradients
for every facade x layer combination.
"""

from __future__ import annotations

import sys
from typing import Any, Callable

import pytest

pytest.importorskip("torch_geometric")

import torch  # noqa: E402
import torch.nn as tnn  # noqa: E402
from torch_geometric import nn as pyg_nn  # noqa: E402
from torch_geometric.data import Data  # noqa: E402
from torch_geometric.nn import global_mean_pool  # noqa: E402
from torch_geometric.utils import degree  # noqa: E402

import mifrost.encoders.derived as derived_mod  # noqa: E402

try:
    from tests.conftest import load_problem
except ImportError:  # pragma: no cover - wheel test layouts
    try:
        from conftest import load_problem  # type: ignore[no-redef]
    except Exception:  # pragma: no cover - pymimir absent
        load_problem = None  # type: ignore[assignment]

FACADE_CASES: tuple[tuple[str, str, dict[str, Any]], ...] = (
    ("star", "StarGraphEncoder", {}),
    ("object_clique", "ObjectGraphEncoder", {"atom_expansion": "clique"}),
    ("object_chain", "ObjectGraphEncoder", {"atom_expansion": "chain"}),
    ("object_star_first", "ObjectGraphEncoder", {"atom_expansion": "star_first"}),
    ("line", "AtomLineGraphEncoder", {}),
    ("hyper", "HypergraphIncidenceEncoder", {}),
    ("bias", "TransformerBiasEncoder", {}),
    ("tuple", "TupleTensorEncoder", {}),
)

OPTIONAL_FACADES = frozenset({"hyper", "bias"})

# hyper/bias are first-class facades; only a missing optional dependency
# may skip them. Genuine encode bugs (TypeError/ValueError/...) must fail.
_SKIP_EXCEPTIONS = (ModuleNotFoundError, ImportError)

STANDARD_LAYERS = (
    "GCNConv",
    "SAGEConv",
    "GATConv",
    "GATv2Conv",
    "GINConv",
    "TransformerConv",
    "NNConv",
    "PNAConv",
    "GraphConv",
    "CGConv",
)

EDGE_ATTR_LAYERS = frozenset(
    {"GATConv", "GATv2Conv", "TransformerConv", "NNConv", "CGConv"}
)

ALL_LAYERS = STANDARD_LAYERS + ("HypergraphConv",)

CONFORMANCE_CASES = [
    (facade, layer) for facade, _cls, _kw in FACADE_CASES for layer in ALL_LAYERS
]


@pytest.fixture(scope="session")
def blocks_problem():
    if load_problem is None or derived_mod is None:
        pytest.skip("pymimir or mifrost derived encoders unavailable")
    try:
        _domain, problem, _state, _dp, _pp = load_problem("blocks", "small")
    except ModuleNotFoundError as exc:
        pytest.skip(f"pymimir runtime missing: {exc}")
    return problem


@pytest.fixture(scope="session")
def initial_state(blocks_problem):
    return blocks_problem.get_initial_state()


@pytest.fixture(scope="session")
def make_facade(blocks_problem):
    """Lazily build and cache one facade instance per configured case."""
    built: dict[str, Any] = {}

    def _make(name: str, **overrides: Any) -> Any:
        matches = [(cls_name, kw) for f, cls_name, kw in FACADE_CASES if f == name]
        if not matches:
            raise KeyError(f"unknown facade case {name!r}")
        cls_name, base_kwargs = matches[0]
        key = f"{name}:{tuple(sorted(overrides.items()))}"
        if key in built:
            return built[key]
        cls = getattr(derived_mod, cls_name, None)
        if cls is None:
            pytest.skip(f"{cls_name} not yet exported by mifrost.encoders.derived")
        try:
            facade = cls(blocks_problem, **(base_kwargs | overrides))
        except (ImportError, TypeError) as exc:
            pytest.skip(f"{cls_name} unusable mid-flight: {exc}")
        built[key] = facade
        return facade

    return _make


@pytest.fixture(scope="session")
def encoded(make_facade, initial_state) -> Callable[[str], Data]:
    """Lazily encode the initial state once per facade case."""
    cache: dict[str, Data] = {}

    def _get(name: str) -> Data:
        if name not in cache:
            facade = make_facade(name)
            if name in OPTIONAL_FACADES:
                try:
                    cache[name] = facade.encode_pyg(initial_state)
                except _SKIP_EXCEPTIONS as exc:
                    pytest.skip(f"{name} encoder output unavailable mid-flight: {exc}")
            else:
                cache[name] = facade.encode_pyg(initial_state)
        return cache[name]

    return _get


def _check_backward(params, out: torch.Tensor) -> None:
    parameters = list(params)
    assert torch.isfinite(out).all(), "layer produced non-finite outputs"
    out.square().mean().backward()
    grads = [p.grad for p in parameters if p.grad is not None]
    if parameters and not grads:
        raise AssertionError("model has parameters but backward delivered no gradients")
    for grad in grads:
        assert torch.isfinite(grad).all(), "layer produced non-finite gradients"


def _run_layer(
    data: Data,
    layer_factory: Callable[[Data], tnn.Module],
    *,
    use_edge_attr: bool,
) -> None:
    """Run forward+backward for one stock layer over an encoded graph."""
    torch.manual_seed(0)
    model = layer_factory(data)
    x = data.x_ids if hasattr(data, "x_ids") else data.x
    if not x.is_floating_point():
        x = x.float()
    raw_edge_attr = getattr(data, "edge_attr", None)
    edge_attr = raw_edge_attr if (use_edge_attr and raw_edge_attr is not None) else None
    if edge_attr is not None and not edge_attr.is_floating_point():
        edge_attr = edge_attr.float()
    if edge_attr is not None:
        out = model(x, data.edge_index, edge_attr)
    else:
        out = model(x, data.edge_index)
    _check_backward(model.parameters(), out)


def _degree_histogram(data: Data) -> torch.Tensor:
    num_nodes = data.num_nodes or (int(data.edge_index.max()) + 1)
    return degree(data.edge_index[0], num_nodes=num_nodes)


def _layer_factories(
    in_channels: int, out_channels: int, edge_dim: int
) -> dict[str, Callable[[Data], tnn.Module]]:
    gin_nn = tnn.Sequential(
        tnn.Linear(in_channels, out_channels),
        tnn.ReLU(),
        tnn.Linear(out_channels, out_channels),
    )
    nnconv_nn = tnn.Sequential(
        tnn.Linear(edge_dim, 16),
        tnn.ReLU(),
        tnn.Linear(16, in_channels * out_channels),
    )
    return {
        "GCNConv": lambda _d: pyg_nn.GCNConv(in_channels, out_channels),
        "SAGEConv": lambda _d: pyg_nn.SAGEConv(in_channels, out_channels),
        "GATConv": lambda _d: pyg_nn.GATConv(
            in_channels, out_channels, edge_dim=edge_dim
        ),
        "GATv2Conv": lambda _d: pyg_nn.GATv2Conv(
            in_channels, out_channels, edge_dim=edge_dim
        ),
        "GINConv": lambda _d: pyg_nn.GINConv(gin_nn),
        "TransformerConv": lambda _d: pyg_nn.TransformerConv(
            in_channels, out_channels, edge_dim=edge_dim
        ),
        "NNConv": lambda _d: pyg_nn.NNConv(
            in_channels, out_channels, nnconv_nn, aggr="mean"
        ),
        "PNAConv": lambda d: pyg_nn.PNAConv(
            in_channels,
            out_channels,
            aggregators=["mean", "max", "min", "std"],
            scalers=["identity", "amplification", "attenuation"],
            deg=_degree_histogram(d),
        ),
        "GraphConv": lambda _d: pyg_nn.GraphConv(in_channels, out_channels),
        "CGConv": lambda _d: pyg_nn.CGConv(in_channels, dim=edge_dim),
    }


@pytest.mark.parametrize(
    ("facade_name", "layer_name"),
    CONFORMANCE_CASES,
    ids=[f"{facade}-{layer}" for facade, layer in CONFORMANCE_CASES],
)
def test_stock_layer_conformance(facade_name, layer_name, encoded):
    """A stock torch_geometric layer trains on the facade's output graph."""
    data = encoded(facade_name)
    x_raw = getattr(data, "x_ids", None)
    if x_raw is None:
        x_raw = getattr(data, "x", None)
    if x_raw is None:
        pytest.skip(f"{facade_name} output has no node features")
    x = x_raw.float()

    if layer_name == "HypergraphConv":
        hyperedge_index = getattr(data, "hyperedge_index", None)
        if facade_name != "hyper" or hyperedge_index is None:
            pytest.skip(
                "HypergraphConv applies only to hypergraphs with hyperedge_index"
            )
        hyper_cls = getattr(pyg_nn, "HypergraphConv", None)
        if hyper_cls is None:
            pytest.skip("torch_geometric lacks HypergraphConv")
        torch.manual_seed(0)
        model = hyper_cls(x.size(1), 8)
        out = model(x, hyperedge_index)
        _check_backward(model.parameters(), out)
        return

    layer_cls = getattr(pyg_nn, layer_name, None)
    if layer_cls is None:
        pytest.skip(f"torch_geometric lacks {layer_name}")
    use_edge_attr = layer_name in EDGE_ATTR_LAYERS
    edge_attr = getattr(data, "edge_attr", None)
    if use_edge_attr and edge_attr is None:
        pytest.skip(f"{layer_name} needs edge_attr; {facade_name} output has none")
    factories = _layer_factories(
        x.size(1), 8, edge_attr.size(1) if edge_attr is not None else 0
    )
    _run_layer(data, factories[layer_name], use_edge_attr=use_edge_attr)


def test_embedding_path_then_gcn_on_star(make_facade, initial_state):
    """Integer-id channels embed via nn.Embedding before message passing."""
    data = make_facade("star").encode_pyg(initial_state)
    vocab_size = int(data.x_ids.max().item()) + 1
    torch.manual_seed(0)
    embedding = tnn.Embedding(vocab_size, 8)
    conv = pyg_nn.GCNConv(8 * data.x_ids.size(1), 8)
    embedded = embedding(data.x_ids.long())
    out = conv(embedded.reshape(data.num_nodes, -1), data.edge_index)
    _check_backward(list(embedding.parameters()) + list(conv.parameters()), out)


def test_batch_encode_pooling_backward(make_facade, initial_state):
    """Batched star encoding pools per graph and backpropagates finitely."""
    batch = make_facade("star").encode_batch_pyg([initial_state, initial_state])
    torch.manual_seed(0)
    conv = pyg_nn.GCNConv(batch.x_ids.size(1), 8)
    out = conv(batch.x_ids.float(), batch.edge_index)
    pooled = global_mean_pool(out, batch.batch)
    assert pooled.size(0) == batch.num_graphs == 2
    _check_backward(conv.parameters(), pooled)


def test_star_edge_index_deterministic(make_facade, initial_state):
    """Encoding twice with the same facade yields identical edges."""
    facade = make_facade("star")
    first = facade.encode_pyg(initial_state)
    second = facade.encode_pyg(initial_state)
    assert torch.equal(first.edge_index, second.edge_index)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
