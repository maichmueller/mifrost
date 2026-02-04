from __future__ import annotations

from typing import Any

import networkx as nx


def canonical_graph(
    graph: nx.Graph,
) -> tuple[
    list[tuple[str, tuple[tuple[str, Any], ...]]],
    list[tuple[str, str, tuple[tuple[str, Any], ...]]],
]:
    nodes = sorted(
        (str(node), tuple(sorted(attrs.items())))
        for node, attrs in graph.nodes(data=True)
    )

    edges: list[tuple[str, str, tuple[tuple[str, Any], ...]]] = []
    if graph.is_multigraph():
        for u, v, _key, attrs in graph.edges(keys=True, data=True):
            u_label, v_label = str(u), str(v)
            if not graph.is_directed() and u_label > v_label:
                u_label, v_label = v_label, u_label
            edges.append((u_label, v_label, tuple(sorted(attrs.items()))))
    else:
        for u, v, attrs in graph.edges(data=True):
            u_label, v_label = str(u), str(v)
            if not graph.is_directed() and u_label > v_label:
                u_label, v_label = v_label, u_label
            edges.append((u_label, v_label, tuple(sorted(attrs.items()))))

    return nodes, sorted(edges)
