"""
Example: Encode planning states with the derived-graph family for vanilla GNNs.

The derived-graph encoders turn a planning problem's objects and atoms into
plain homogeneous PyTorch Geometric graphs consumable by stock message-passing
layers. This script encodes two fixtures -- blocks-world and a ternary/nullary
toy domain -- with *every* facade of the family, printing size and channel
histograms, running an embedding + GCNConv forward pass, and rendering one
panel per facade.

Two things this script exists to keep honest:

* **Column 4 (``history_dt``) is the one SIGNED channel.** The naive recipe
  ``Embedding(int(x_ids[:, c].max()) + 2, d)`` followed by
  ``emb(x_ids[:, c].long())`` raises ``IndexError`` the moment any history is
  supplied, because history dt is negative. The encoder exports a per-graph
  ``history_dt_offset`` for exactly this; see ``_channel_ids``.
* **Every encoded node must appear in its own picture.** ``_render`` asserts
  it per panel, so a node whose only edge is a self-loop cannot silently
  vanish again.

Every encode below supplies goals, a subgoal layer, two history steps and a
grounded action, so none of those code paths can rot unnoticed.
"""

from __future__ import annotations

import collections
from pathlib import Path

from mifrost.encoders._derived_visualization import core_nodes
from mifrost.encoders.derived import (
    AtomLineGraphEncoder,
    HypergraphIncidenceEncoder,
    ObjectGraphEncoder,
    StarGraphEncoder,
    TransformerBiasEncoder,
    TupleTensorEncoder,
)

from _helpers import first_transition, load_problem, load_space

FIXTURES: tuple[tuple[str, str], ...] = (("blocks", "smedium"), ("tri", "p1"))
FACADE_ORDER: tuple[str, ...] = (
    "star",
    "object_clique",
    "line",
    "hyperedge",
    "tuples",
    "spd",
)


def _facades(problem):
    """Build every derived facade for one problem, with what it adds."""
    return [
        ("star", StarGraphEncoder(problem), "reified fact nodes, hub edges"),
        (
            "object_clique",
            ObjectGraphEncoder(problem, atom_expansion="clique"),
            "objects only; labeled clique/self edges + tuple instances",
        ),
        ("line", AtomLineGraphEncoder(problem), "star + line_share fact shortcuts"),
        (
            "hyperedge",
            HypergraphIncidenceEncoder(problem),
            "star + one hyperedge per instance (anchor member for arity 0)",
        ),
        (
            "tuples",
            TupleTensorEncoder(problem),
            "star + ordered tuple slots per instance",
        ),
        (
            "spd",
            TransformerBiasEncoder(problem),
            "objects only + shortest-path-distance object pairs",
        ),
    ]


def _histogram(values, vocab) -> dict[str, int]:
    counts = collections.Counter(int(value) for value in values)
    return {
        (vocab[kind] if 0 <= kind < len(vocab) else str(kind)): count
        for kind, count in sorted(counts.items())
    }


def _summarize(name: str, data, note: str) -> None:
    print(
        f"{name}: nodes={data.num_nodes} edges={data.edge_index.size(1)} "
        f"x_ids={tuple(data.x_ids.shape)} edge_attr={tuple(data.edge_attr.shape)}"
    )
    print(f"  {note}")
    print(f"  roles: {_histogram(data.x_ids[:, 0], data.vocab_roles)}")
    print(f"  edge kinds: {_histogram(data.edge_attr[:, 0], data.vocab_edge_kinds)}")
    relations = getattr(data, "vocab_relations", None) or data.vocab_predicates
    present = [int(v) - 1 for v in data.x_ids[:, 1].tolist() if int(v) > 0]
    print(f"  relations: {_histogram(present, relations)}")
    extras = []
    for key in (
        "hyperedge_index",
        "tuple_rel_ids",
        "spd_dist",
        "instance_node_indices",
    ):
        value = getattr(data, key, None)
        if value is not None:
            extras.append(f"{key}={tuple(value.shape)}")
    if extras:
        print(f"  instance channels: {', '.join(extras)}")


def _channel_ids(data, column: int):
    """Return embeddable non-negative ids for one ``x_ids`` column.

    ``history_dt`` is the single signed channel of the contract. Shifting it
    by the encoder's own per-graph ``history_dt_offset`` is what makes the
    documented ``Embedding`` recipe correct; without it, any state carrying
    history raises ``IndexError: index out of range in self``.
    """
    ids = data.x_ids[:, column].long()
    names = list(getattr(data, "channel_names", ()) or ())
    if column < len(names) and names[column] == "history_dt":
        offset = getattr(data, "history_dt_offset", None)
        if offset is not None:
            offset = offset.long().reshape(-1)
            batch = getattr(data, "batch", None)
            ids = ids + (offset[batch] if batch is not None else int(offset[0]))
    return ids


def _gnn_smoke_test(data) -> None:
    """Embed every channel and run one GCNConv pass over the star view."""
    import torch
    import torch.nn as tnn
    from torch_geometric.nn import GCNConv

    torch.manual_seed(0)
    num_channels = data.x_ids.size(1)
    ids = [_channel_ids(data, c) for c in range(num_channels)]
    assert all(int(column.min()) >= 0 for column in ids), "channel ids must be >= 0"
    embeddings = tnn.ModuleList(
        [tnn.Embedding(int(column.max().item()) + 2, 8) for column in ids]
    )
    feats = torch.cat(
        [embeddings[c](ids[c]) for c in range(num_channels)],
        dim=-1,
    )
    conv = GCNConv(feats.size(-1), 16)
    out = conv(feats, data.edge_index)
    object_mask = data.x_ids[:, 0] == 0
    pooled = out[object_mask].mean(dim=0)
    raw_dt = data.x_ids[:, 4].long()
    shifted = ids[4]
    assert torch.equal(shifted - int(data.history_dt_offset[0]), raw_dt)
    print(
        "GCNConv smoke test: "
        f"embedded={tuple(feats.shape)} out={tuple(out.shape)} "
        f"object-pooled={tuple(pooled.shape)}"
    )
    print(
        f"  signed history_dt in [{int(raw_dt.min())}, {int(raw_dt.max())}], "
        f"history_dt_offset={int(data.history_dt_offset[0])} -> "
        f"embeddable ids in [{int(shifted.min())}, {int(shifted.max())}]"
    )


def main():
    per_fixture: list[tuple[str, list]] = []
    for domain, problem_name in FIXTURES:
        _domain, problem, state = load_problem(domain=domain, problem=problem_name)
        space, _d, _p = load_space(domain=domain, problem=problem_name)
        goals = list(problem.get_goal_condition().get_literals())
        inputs = {
            "goals": goals,
            "subgoal_layers": [[goals[0]]] if goals else None,
            "history_subgoals": (
                [(-1, [goals[0]]), (-2, goals[:1])] if goals else None
            ),
            "actions": [first_transition(space, state)[0]],
        }
        print(
            f"=== {domain}/{problem_name} "
            f"(goals={len(goals)}, 1 subgoal layer, 2 history steps, 1 action)"
        )
        panels = []
        for name, encoder, note in _facades(problem):
            data = encoder.encode_pyg(state, **inputs)
            _summarize(f"  {name}", data, note)
            panels.append((name, note, encoder, data))
        per_fixture.append((f"{domain}/{problem_name}", panels))
        print()

    try:
        import torch_geometric  # noqa: F401
    except ImportError as exc:
        print(f"torch_geometric unavailable, skipping GNN smoke test: {exc}")
    else:
        _gnn_smoke_test(per_fixture[0][1][0][3])

    _render(per_fixture)


def _render(per_fixture) -> None:
    """Render one figure per fixture and verify nothing was dropped.

    One fixture per file, six large panels each: twelve panels crammed into a
    single strip are unreadable at 100%, which is the size a reader actually
    opens the PNG at. Each figure carries ONE shared legend -- the role and
    edge-kind vocabularies are identical in every panel, so per-panel legends
    both repeat themselves and cover the drawings.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        print(f"matplotlib unavailable, skipping rendering: {exc}")
        return

    from mifrost.encoders._derived_visualization import legend_handles

    plots_dir = Path(__file__).resolve().parent / "_plots"
    plots_dir.mkdir(exist_ok=True)
    columns = 3
    rows = -(-len(FACADE_ORDER) // columns)

    for index, (fixture, panels) in enumerate(per_fixture):
        figure, axes = plt.subplots(
            rows, columns, figsize=(8.4 * columns, 8.0 * rows), squeeze=False
        )
        flat = [ax for row in axes for ax in row]
        graphs = []
        for (name, note, encoder, data), ax in zip(panels, flat, strict=False):
            graph = encoder.to_networkx(data, include_line_shares=True)
            graphs.append(graph)
            encoder.draw(data, ax=ax, font_size=8, legend=False, node_size=520)
            drawn = sum(len(collection.get_offsets()) for collection in ax.collections)
            core = len(core_nodes(graph))
            # D11 guard: a node that exists in the encoding must exist in its
            # picture. Edge filtering must never delete nodes.
            assert core == int(data.num_nodes), (
                f"{fixture} {name}: {core} core nodes rendered from "
                f"{int(data.num_nodes)} encoded"
            )
            assert drawn == graph.number_of_nodes(), (
                f"{fixture} {name}: {drawn} markers drawn from "
                f"{graph.number_of_nodes()} nodes"
            )
            overlay = graph.number_of_nodes() - core
            ax.set_title(
                f"{name} -- {note}\n"
                f"{int(data.num_nodes)} encoded nodes"
                + (f" + {overlay} instance nodes" if overlay else "")
                + f", {int(data.edge_index.size(1))} edges",
                fontsize=10,
            )
        for ax in flat[len(panels) :]:
            ax.set_axis_off()

        handles = legend_handles(graphs)
        figure.legend(
            handles=handles,
            loc="lower center",
            ncol=min(9, len(handles)),
            fontsize=10,
            frameon=False,
        )
        figure.suptitle(
            f"derived-graph encoders on {fixture} "
            "(goals + 1 subgoal layer + 2 history steps + 1 action)",
            fontsize=14,
        )
        figure.tight_layout(rect=(0.0, 0.055, 1.0, 0.97))
        suffix = "" if index == 0 else f"_{fixture.split('/')[0]}"
        out_path = plots_dir / f"derived_graph_example{suffix}.png"
        figure.savefig(out_path, dpi=110)
        plt.close(figure)
        print(f"rendered {out_path} ({len(panels)} facades, node counts verified)")


if __name__ == "__main__":
    main()
