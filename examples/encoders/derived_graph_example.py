"""
Example: Encode planning states with the derived-graph family for vanilla GNNs.

The derived-graph encoders turn a planning problem's objects and atoms into
plain homogeneous PyTorch Geometric graphs consumable by stock message-passing
layers. This script encodes a blocks-world state with three facades, prints
size and channel histograms, and (when torch_geometric is installed) runs a
small embedding + GCNConv forward pass on the star output.
"""

from __future__ import annotations

import collections

from mifrost.encoders.derived import (
    HypergraphIncidenceEncoder,
    ObjectGraphEncoder,
    StarGraphEncoder,
)

from _helpers import load_problem


def _histogram(values, vocab) -> dict[str, int]:
    counts = collections.Counter(int(value) for value in values)
    return {vocab[kind]: count for kind, count in sorted(counts.items())}


def _summarize(name: str, data) -> None:
    print(
        f"{name}: nodes={data.num_nodes} edges={data.edge_index.size(1)} "
        f"x_ids={tuple(data.x_ids.shape)} edge_attr={tuple(data.edge_attr.shape)}"
    )
    print(f"  roles: {_histogram(data.x_ids[:, 0], data.vocab_roles)}")
    print(f"  edge kinds: {_histogram(data.edge_attr[:, 0], data.vocab_edge_kinds)}")


def main():
    _domain, problem, state = load_problem(domain="blocks", problem="smedium")

    star_encoder = StarGraphEncoder(problem)
    star = star_encoder.encode_pyg(state)
    _summarize("StarGraphEncoder", star)

    chain_encoder = ObjectGraphEncoder(problem, atom_expansion="chain")
    chain = chain_encoder.encode_pyg(state)
    _summarize("ObjectGraphEncoder(chain)", chain)

    try:
        hyper_encoder = HypergraphIncidenceEncoder(problem)
        hyper = hyper_encoder.encode_pyg(state)
        memberships = hyper.hyperedge_index.size(1)
        hyperedges = hyper.hyperedge_attr_ids.size(0)
        print(
            f"HypergraphIncidenceEncoder: nodes={hyper.num_nodes} "
            f"hyperedges={hyperedges} memberships={memberships}"
        )
    except Exception as exc:
        print(f"HypergraphIncidenceEncoder unavailable: {type(exc).__name__}: {exc}")

    try:
        import torch
        import torch.nn as tnn
        from torch_geometric.nn import GCNConv
    except ImportError as exc:
        print(f"torch_geometric unavailable, skipping GNN smoke test: {exc}")
        _render(problem, state, star)
        return

    torch.manual_seed(0)
    num_channels = star.x_ids.size(1)
    embeddings = tnn.ModuleList(
        [
            tnn.Embedding(int(star.x_ids[:, c].max().item()) + 2, 8)
            for c in range(num_channels)
        ]
    )
    feats = torch.cat(
        [embeddings[c](star.x_ids[:, c].long()) for c in range(num_channels)],
        dim=-1,
    )
    conv = GCNConv(feats.size(-1), 16)
    out = conv(feats, star.edge_index)
    object_mask = star.x_ids[:, 0] == 0
    pooled = out[object_mask].mean(dim=0)
    print(
        "GCNConv smoke test: "
        f"embedded={tuple(feats.shape)} out={tuple(out.shape)} "
        f"object-pooled={tuple(pooled.shape)}"
    )
    _render(problem, state, star)


def _render(problem, state, star) -> None:
    """Render one figure per facade when matplotlib is available."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        print(f"matplotlib unavailable, skipping rendering: {exc}")
        return

    from mifrost.encoders.derived import AtomLineGraphEncoder, TupleTensorEncoder

    facades = [
        ("star", StarGraphEncoder(problem), star),
        (
            "object_clique",
            ObjectGraphEncoder(problem, atom_expansion="clique"),
            ObjectGraphEncoder(problem, atom_expansion="clique").encode_pyg(state),
        ),
        (
            "line",
            AtomLineGraphEncoder(problem),
            AtomLineGraphEncoder(problem).encode_pyg(state),
        ),
        (
            "hyperedge",
            HypergraphIncidenceEncoder(problem),
            HypergraphIncidenceEncoder(problem).encode_pyg(state),
        ),
        (
            "tuples",
            TupleTensorEncoder(problem),
            TupleTensorEncoder(problem).encode_pyg(state),
        ),
    ]
    figure, axes = plt.subplots(1, len(facades), figsize=(6 * len(facades), 5.5))
    for (name, encoder, data), ax in zip(facades, axes, strict=True):
        encoder.draw(data, ax=ax, font_size=7)
        ax.set_title(name)
    figure.tight_layout()
    figure.savefig("derived_graph_example.png", dpi=110)
    plt.close(figure)
    print("rendered derived_graph_example.png")


if __name__ == "__main__":
    main()
