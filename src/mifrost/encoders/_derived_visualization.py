"""Rendering helpers for the derived-graph encoder family.

Mirrors the hgraph/color visualization idioms: ``derived_to_networkx``
converts one PyG :class:`DerivedGraphData` into a NetworkX multigraph with
stable display names and decoded role/kind attributes, and
``draw_derived`` renders it with role-shaped, kind-styled matplotlib passes.
"""

from __future__ import annotations

from typing import Any, Iterable

import torch

ROLE_NAMES: tuple[str, ...] = (
    "object",
    "fact",
    "goal",
    "subgoal",
    "history",
    "action",
    "hyperedge",
)

EDGE_KIND_NAMES: tuple[str, ...] = (
    "arg_fwd",
    "arg_bwd",
    "clique_fwd",
    "clique_bwd",
    "chain_fwd",
    "chain_bwd",
    "star_first_fwd",
    "star_first_bwd",
    "nullary_self",
    "action_fwd",
    "action_bwd",
    "line_share",
)

REVERSE_KINDS: frozenset[str] = frozenset(
    {"arg_bwd", "clique_bwd", "chain_bwd", "star_first_bwd", "action_bwd"}
)

ROLE_COLORS: dict[str, str] = {
    "object": "#4C72B0",
    "fact": "#DD8452",
    "goal": "#55A868",
    "subgoal": "#C44E52",
    "history": "#8172B3",
    "action": "#937860",
    "hyperedge": "#DA8BC3",
}

KIND_STYLES: dict[str, tuple[str, float, float, str]] = {
    "arg_fwd": ("#666666", 1.0, 0.9, "-"),
    "arg_bwd": ("#BBBBBB", 0.5, 0.35, ":"),
    "clique_fwd": ("#1F77B4", 1.4, 0.9, "-"),
    "clique_bwd": ("#1F77B4", 0.6, 0.3, ":"),
    "chain_fwd": ("#2CA02C", 1.4, 0.9, "-"),
    "chain_bwd": ("#2CA02C", 0.6, 0.3, ":"),
    "star_first_fwd": ("#FF7F0E", 1.4, 0.9, "-"),
    "star_first_bwd": ("#FF7F0E", 0.6, 0.3, ":"),
    "nullary_self": ("#888888", 0.8, 0.7, "-"),
    "action_fwd": ("#8C564B", 1.4, 0.9, "-"),
    "action_bwd": ("#8C564B", 0.6, 0.3, ":"),
    "line_share": ("#17BECF", 1.0, 0.6, "--"),
    "membership": ("#DA8BC3", 1.0, 0.6, ":"),
}

ROLE_SHAPES: dict[str, str] = {
    "object": "o",
    "fact": "o",
    "goal": "D",
    "subgoal": "D",
    "history": "s",
    "action": "s",
    "hyperedge": "P",
}


def derived_to_networkx(
    data: Any,
    *,
    include_hyperedges: bool = True,
    include_reverse_edges: bool = True,
    include_line_shares: bool = False,
    include_self_loops: bool = False,
) -> Any:
    """Convert derived-graph PyG data into a labeled NetworkX multigraph.

    Node attributes carry the decoded integer channels (``role``,
    ``predicate``, ``sign``, ``goal_level``); edge attributes carry the
    decoded kind name plus both argument positions. When
    ``include_hyperedges`` is set and the data carries stacked membership,
    one auxiliary ``hyperedge`` node per hyperedge is inserted and wired to
    its member objects with ``membership`` edges.
    """
    import networkx as nx

    graph = nx.MultiDiGraph()
    node_names = getattr(data, "node_names", None)
    count = int(getattr(data, "num_nodes", 0))
    x_ids = getattr(data, "x_ids", None)
    vocab_predicates = tuple(getattr(data, "vocab_predicates", ()) or ())
    for index in range(count):
        name = (
            str(node_names[index])
            if node_names and index < len(node_names)
            else str(index)
        )
        attrs: dict[str, Any] = {
            "role": "object",
            "predicate": "",
            "sign": 0,
            "goal_level": 0,
        }
        if torch.is_tensor(x_ids) and x_ids.dim() == 2 and index < x_ids.size(0):
            row = x_ids[index].long().tolist()
            attrs = {
                "role": ROLE_NAMES[int(row[0])]
                if int(row[0]) < len(ROLE_NAMES)
                else str(row[0]),
                "predicate": (
                    vocab_predicates[int(row[1]) - 1]
                    if 0 < int(row[1]) <= len(vocab_predicates)
                    else ""
                ),
                "sign": int(row[2]),
                "goal_level": int(row[3]),
                "channel_ids": row,
            }
        graph.add_node(index, label=name, **attrs)

    edge_index = getattr(data, "edge_index", None)
    edge_attr = getattr(data, "edge_attr", None)
    if torch.is_tensor(edge_index) and edge_index.numel() > 0:
        kinds = edge_attr[:, 0].long() if torch.is_tensor(edge_attr) else None
        positions = edge_attr[:, 1:].long() if torch.is_tensor(edge_attr) else None
        for column in range(edge_index.size(1)):
            kind_id = int(kinds[column].item()) if kinds is not None else 0
            kind = (
                EDGE_KIND_NAMES[kind_id]
                if kind_id < len(EDGE_KIND_NAMES)
                else str(kind_id)
            )
            if kind in REVERSE_KINDS and not include_reverse_edges:
                continue
            if kind == "line_share" and not include_line_shares:
                continue
            if kind == "nullary_self" and not include_self_loops:
                continue
            pos_a = int(positions[column, 0].item()) if positions is not None else -1
            pos_b = int(positions[column, 1].item()) if positions is not None else -1
            source = int(edge_index[0, column].item())
            target = int(edge_index[1, column].item())
            label = kind if pos_a < 0 and pos_b < 0 else f"{kind} [{pos_a},{pos_b}]"
            graph.add_edge(
                source,
                target,
                key=column,
                kind=kind,
                position=(pos_a, pos_b),
                label=label,
            )

    membership = getattr(data, "hyperedge_index", None)
    if include_hyperedges and torch.is_tensor(membership) and membership.numel() > 0:
        hyperedge_attrs = getattr(data, "hyperedge_attr_ids", None)
        pairs = membership.long().tolist()
        seen: set[int] = set()
        for member, hyperedge in zip(pairs[0], pairs[1], strict=True):
            hyperedge = int(hyperedge)
            if hyperedge not in seen:
                seen.add(hyperedge)
                role = ""
                if (
                    torch.is_tensor(hyperedge_attrs)
                    and hyperedge < hyperedge_attrs.numel()
                ):
                    role_id = int(hyperedge_attrs[hyperedge].item())
                    role = (
                        ROLE_NAMES[role_id]
                        if role_id < len(ROLE_NAMES)
                        else str(role_id)
                    )
                graph.add_node(
                    ("h", hyperedge),
                    label=f"h{hyperedge}",
                    role="hyperedge",
                    instance_role=role,
                    predicate="",
                    sign=0,
                    goal_level=0,
                )
            graph.add_edge(
                int(member),
                ("h", hyperedge),
                key=("m", hyperedge, int(member)),
                kind="membership",
                position=(-1, -1),
                label="membership",
            )
    return graph


def draw_derived(
    graph: Any,
    *,
    ax: Any | None = None,
    with_labels: bool = True,
    edge_labels: bool = False,
    hide_reverse_edges: bool = True,
    layout: Any | None = None,
    node_size: int | None = None,
    font_size: int = 7,
    legend: bool = True,
    node_kwargs: dict | None = None,
    edge_kwargs: dict | None = None,
) -> Any:
    """Render a derived-graph NetworkX multigraph and return the axis.

    Nodes are drawn per role with distinct colors and marker shapes
    (objects largest, anchors as squares/diamonds, hyperedges as pentagons);
    edges are styled per kind, with reverse-direction kinds hidden by
    default since forward kinds already carry position labels.
    """
    import matplotlib.pyplot as plt
    import networkx as nx

    node_kwargs = node_kwargs or {}
    edge_kwargs = edge_kwargs or {}
    if ax is None:
        _, ax = plt.subplots()

    render_graph = graph
    if hide_reverse_edges:
        render_graph = graph.edge_subgraph(
            [
                (u, v, k)
                for u, v, k, attrs in graph.edges(keys=True, data=True)
                if attrs.get("kind") not in REVERSE_KINDS
            ]
        ).copy()

    positions = layout or nx.spring_layout(render_graph, seed=0, k=1.4)

    role_groups: dict[str, list[Any]] = {}
    for node, attrs in render_graph.nodes(data=True):
        role_groups.setdefault(str(attrs.get("role", "object")), []).append(node)

    base_size = node_size or 420
    size_by_role: dict[str, int] = {
        "object": base_size,
        "fact": max(160, base_size // 2),
        "hyperedge": max(140, base_size // 3),
        "goal": max(200, int(base_size * 0.6)),
        "subgoal": max(200, int(base_size * 0.6)),
        "history": max(180, int(base_size * 0.55)),
        "action": max(180, int(base_size * 0.55)),
    }
    for role, nodes in role_groups.items():
        nx.draw_networkx_nodes(
            render_graph,
            positions,
            nodelist=nodes,
            node_color=ROLE_COLORS.get(role, "#999999"),
            node_shape=ROLE_SHAPES.get(role, "o"),
            node_size=size_by_role.get(role, base_size // 2),
            ax=ax,
            **node_kwargs,
        )

    kind_groups: dict[str, list[tuple[Any, Any, Any]]] = {}
    for source, target, key, attrs in render_graph.edges(keys=True, data=True):
        kind_groups.setdefault(str(attrs.get("kind", "arg_fwd")), []).append(
            (source, target, key)
        )
    for kind, edges in kind_groups.items():
        color, width, alpha, style = KIND_STYLES.get(kind, ("#CCCCCC", 1.0, 0.8, "-"))
        nx.draw_networkx_edges(
            render_graph,
            positions,
            edgelist=[(u, v, k) for u, v, k in edges],
            edge_color=color,
            width=width,
            alpha=alpha,
            style=style,
            arrows=True,
            arrowsize=9,
            ax=ax,
            **edge_kwargs,
        )

    if with_labels:
        labels = {
            node: str(attrs.get("label", node))
            for node, attrs in render_graph.nodes(data=True)
        }
        anchor_roles = {"fact", "goal", "subgoal", "history", "action", "hyperedge"}
        anchor_nodes = [
            node
            for node, attrs in render_graph.nodes(data=True)
            if attrs.get("role") in anchor_roles
        ]
        nx.draw_networkx_labels(
            render_graph,
            positions,
            labels={node: labels[node] for node in anchor_nodes},
            font_size=font_size,
            ax=ax,
        )
        object_nodes = [
            node
            for node, attrs in render_graph.nodes(data=True)
            if attrs.get("role") == "object"
        ]
        if object_nodes:
            nx.draw_networkx_labels(
                render_graph,
                positions,
                labels={node: labels[node] for node in object_nodes},
                font_size=font_size + 1,
                font_weight="bold",
                ax=ax,
            )

    if edge_labels:
        edge_label_map = {
            (source, target): str(attrs.get("label", attrs.get("kind", "")))
            for source, target, _, attrs in render_graph.edges(keys=True, data=True)
        }
        nx.draw_networkx_edge_labels(
            render_graph,
            positions,
            edge_labels=edge_label_map,
            font_size=max(6, font_size - 1),
            ax=ax,
        )

    if legend:
        handles = [
            plt.Line2D(
                [0],
                [0],
                marker=ROLE_SHAPES.get(role, "o"),
                linestyle="",
                markerfacecolor=color,
                markeredgecolor=color,
                markersize=8,
                label=role,
            )
            for role, color in ROLE_COLORS.items()
            if role in role_groups
        ]
        present_kinds = [kind for kind in KIND_STYLES if kind in kind_groups]
        handles.extend(
            plt.Line2D(
                [0],
                [0],
                color=color,
                linewidth=width,
                alpha=min(1.0, alpha + 0.2),
                linestyle=style,
                label=kind,
            )
            for kind in present_kinds
            for color, width, alpha, style in [KIND_STYLES[kind]]
        )
        ax.legend(
            handles=handles, loc="best", fontsize=max(6, font_size - 1), framealpha=0.85
        )

    ax.set_axis_off()
    return ax


def summarize_derived(data: Any) -> str:
    """Return a short human-readable channel histogram for one graph."""
    lines: list[str] = []
    x_ids = getattr(data, "x_ids", None)
    if torch.is_tensor(x_ids):
        counts: dict[str, int] = {}
        for value in x_ids[:, 0].long().tolist():
            role = ROLE_NAMES[value] if value < len(ROLE_NAMES) else str(value)
            counts[role] = counts.get(role, 0) + 1
        lines.append(
            "roles: "
            + ", ".join(f"{role}={count}" for role, count in sorted(counts.items()))
        )
    edge_attr = getattr(data, "edge_attr", None)
    if torch.is_tensor(edge_attr):
        kinds: dict[str, int] = {}
        for value in edge_attr[:, 0].long().tolist():
            kind = (
                EDGE_KIND_NAMES[value] if value < len(EDGE_KIND_NAMES) else str(value)
            )
            kinds[kind] = kinds.get(kind, 0) + 1
        lines.append(
            "kinds: "
            + ", ".join(f"{kind}={count}" for kind, count in sorted(kinds.items()))
        )
    return "; ".join(lines)


def filter_nodes_by_roles(graph: Any, roles: Iterable[str]) -> list[Any]:
    """Return nodes whose role attribute is one of ``roles``."""
    wanted = set(roles)
    return [
        node for node, attrs in graph.nodes(data=True) if attrs.get("role") in wanted
    ]
