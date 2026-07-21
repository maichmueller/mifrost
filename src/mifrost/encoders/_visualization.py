"""Private NetworkX conversion and drawing helpers for encoder facades."""

from __future__ import annotations

import numbers
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from torch_geometric.data import HeteroData
from torch_geometric.utils import to_networkx as pyg_to_networkx

if TYPE_CHECKING:
    import networkx as nx


@dataclass(frozen=True)
class HGraphVisualizationContext:
    """Data-only rendering configuration for heterogeneous graphs."""

    symbol_type_id: str
    include_lgan_edges: bool
    lgan_edge_positions: frozenset[Any]


@dataclass(frozen=True)
class HorizonVisualizationContext:
    """Data-only metadata needed by horizon conversion and layout."""

    hgraph: HGraphVisualizationContext
    parent_relation: str
    sibling_relation: str
    cousin_relation: str
    target_symbol_prefix: str


def hgraph_to_networkx(data: HeteroData) -> nx.MultiDiGraph:
    """Convert heterogeneous PyG data to a graph with stable display names."""
    import networkx as nx

    graph = pyg_to_networkx(
        data, node_attrs=["node_names"], edge_attrs=[], to_multi=True
    )
    renaming: dict[object, str] = {}
    used_names: dict[str, int] = {}

    for node, attrs in graph.nodes(data=True):
        if isinstance(node, tuple) and len(node) == 2:
            node_type = str(node[0])
            attrs.setdefault("type", node_type)
        else:
            node_type = str(attrs.get("type", ""))
            attrs.setdefault("type", node_type)

        name = attrs.get("node_names")
        if name is None:
            if isinstance(node, tuple) and len(node) == 2:
                name = f"{node[0]}:{node[1]}"
            else:
                name = str(node)
        name = str(name)
        suffix = used_names.get(name, 0)
        display_name = f"{name}#{suffix}" if suffix > 0 else name
        used_names[name] = suffix + 1
        renaming[node] = display_name

    graph = nx.relabel_nodes(graph, renaming, copy=True)
    if graph.is_multigraph():
        for _, _, _, attrs in graph.edges(keys=True, data=True):
            edge_type = attrs.get("edge_type") or attrs.get("type")
            if isinstance(edge_type, tuple) and len(edge_type) > 1:
                attrs["position"] = edge_type[1]
    else:
        for _, _, attrs in graph.edges(data=True):
            edge_type = attrs.get("edge_type") or attrs.get("type")
            if isinstance(edge_type, tuple) and len(edge_type) > 1:
                attrs["position"] = edge_type[1]
    return graph


def target_index_from_name(name: str, target_symbol_prefix: str) -> int:
    """Extract an integer target index, or ``-1`` for a regular symbol."""
    if not name.startswith(target_symbol_prefix):
        return -1
    remainder = name[len(target_symbol_prefix) :]
    if not remainder.isdigit():
        return -1
    return int(remainder)


def horizon_to_networkx(
    data: HeteroData, context: HorizonVisualizationContext
) -> nx.MultiGraph:
    """Convert encoded horizon data while retaining target-tree metadata."""
    import networkx as nx

    graph = nx.MultiGraph()
    symbol_type = context.hgraph.symbol_type_id
    parent_type = getattr(data, "parent_relation", context.parent_relation)

    def _to_list(value):
        if value is None:
            return []
        if hasattr(value, "tolist"):
            return value.tolist()
        return list(value)

    node_names_by_type: dict[str, list[str]] = {}
    for node_type in data.node_types:
        storage = data[node_type]
        node_names_by_type[node_type] = list(getattr(storage, "node_names", []))

    symbol_nodes = node_names_by_type.get(symbol_type, [])
    target_names = list(getattr(data, "target_names", []))
    target_depths = _to_list(getattr(data, "target_depths", []))
    target_indices = _to_list(getattr(data, "target_indices", []))
    target_positions = _to_list(getattr(data, "target_positions", []))
    object_names = list(getattr(data, "object_names", []))
    object_name_set = {str(name) for name in object_names}

    target_info: dict[int, tuple[str, int | None, int | None]] = {}
    target_name_by_index: dict[int, str] = {}
    target_depth_by_index: dict[int, int] = {}
    for pos, sym_idx in enumerate(target_positions):
        if sym_idx < 0 or sym_idx >= len(symbol_nodes):
            continue
        target_name = (
            target_names[pos] if pos < len(target_names) else symbol_nodes[sym_idx]
        )
        depth = target_depths[pos] if pos < len(target_depths) else None
        index = target_indices[pos] if pos < len(target_indices) else None
        target_info[sym_idx] = (target_name, depth, index)
        if index is not None:
            target_name_by_index[index] = target_name
            if depth is not None:
                target_depth_by_index[index] = depth

    object_iter = iter(object_names)
    for idx, node_key in enumerate(symbol_nodes):
        target_name = depth = index = None
        if idx in target_info:
            target_name, depth, index = target_info[idx]
        else:
            key_index = target_index_from_name(node_key, context.target_symbol_prefix)
            if key_index >= 0 and str(node_key) not in object_name_set:
                index = key_index
                target_name = target_name_by_index.get(key_index, node_key)
                depth = target_depth_by_index.get(
                    key_index, 0 if key_index == 0 else None
                )
        if index is not None:
            graph.add_node(
                node_key,
                type=symbol_type,
                name=target_name,
                depth=depth,
                target_index=index,
            )
        else:
            graph.add_node(node_key, type=symbol_type, name=next(object_iter, node_key))

    for other_type, names in node_names_by_type.items():
        if other_type != symbol_type:
            for name in names:
                graph.add_node(name, type=other_type)

    for edge_type, edge_index in data.edge_index_dict.items():
        src_type, pos_str, dst_type = edge_type
        if src_type != symbol_type:
            continue
        src_names = node_names_by_type.get(src_type, [])
        dst_names = node_names_by_type.get(dst_type, [])
        if not src_names or not dst_names:
            continue
        try:
            position: int | str = int(pos_str)
        except (TypeError, ValueError):
            position = pos_str
        for src_idx, dst_idx in zip(edge_index[0].tolist(), edge_index[1].tolist()):
            if 0 <= src_idx < len(src_names) and 0 <= dst_idx < len(dst_names):
                graph.add_edge(
                    src_names[src_idx], dst_names[dst_idx], position=position
                )

    if parent_type in node_names_by_type:
        target_name_to_idx = {
            name: int(attrs["target_index"])
            for name, attrs in graph.nodes(data=True)
            if attrs.get("type") == symbol_type
            and attrs.get("target_index") is not None
        }
        for transition_name in node_names_by_type[parent_type]:
            parent_idx = child_idx = None
            for neighbor, edge_dict in graph[transition_name].items():
                for edge_data in edge_dict.values():
                    if edge_data.get("position") == 0:
                        parent_idx = target_name_to_idx.get(neighbor)
                    elif edge_data.get("position") == 1:
                        child_idx = target_name_to_idx.get(neighbor)
            if parent_idx is not None:
                graph.nodes[transition_name]["parent"] = parent_idx
            if child_idx is not None:
                graph.nodes[transition_name]["child"] = child_idx
    return graph


def draw_hgraph(
    graph: nx.Graph,
    *,
    context: HGraphVisualizationContext,
    ax=None,
    with_labels: bool = True,
    edge_labels: bool = True,
    node_kwargs: dict | None = None,
    edge_kwargs: dict | None = None,
    layout: dict | None = None,
    node_size: float | None = None,
    node_alpha: float | None = None,
    edge_width: float | None = None,
    edge_alpha: float | None = None,
    label_font_size: float | None = None,
    label_nodes: Iterable[str] | None = None,
    label_node_types: Iterable[str] | None = None,
    label_edges: Iterable[tuple[str, ...]] | None = None,
    symbol_node_scale: float = 1.5,
    non_symbol_linestyle: str | None = "--",
):
    import matplotlib.pyplot as plt
    import networkx as nx

    node_kwargs = node_kwargs or {}
    edge_kwargs = edge_kwargs or {}

    if ax is None:
        _, ax = plt.subplots()

    pos = layout or nx.spring_layout(graph)

    node_types = [graph.nodes[n]["type"] for n in graph.nodes]
    unique_types = list(dict.fromkeys(node_types))
    if unique_types:
        cmap = plt.get_cmap("tab20_r")
        type_to_color = {
            ntype: cmap(i / max(1, len(unique_types) - 1))
            for i, ntype in enumerate(unique_types)
        }

    base_node_kwargs = dict(node_kwargs)
    if node_size is not None:
        base_node_kwargs.setdefault("node_size", node_size)
    if node_alpha is not None:
        base_node_kwargs.setdefault("alpha", node_alpha)

    base_node_size_value = base_node_kwargs.get("node_size")
    if isinstance(base_node_size_value, Sequence) and not isinstance(
        base_node_size_value, (str, bytes)
    ):
        base_node_size_value = base_node_size_value[0] if base_node_size_value else None
    if base_node_size_value is None:
        inferred_size = node_size if node_size is not None else 300
        base_node_kwargs.setdefault("node_size", inferred_size)
        base_node_size_value = inferred_size
    elif not isinstance(base_node_size_value, numbers.Real):
        base_node_size_value = 300

    label_edge_set = None
    if label_edges is not None:
        label_edge_set = {tuple(edge) for edge in label_edges}

    symbol_nodes = [
        node
        for node in graph.nodes
        if graph.nodes[node]["type"] == context.symbol_type_id
    ]
    symbol_set = set(symbol_nodes)
    other_nodes = [node for node in graph.nodes if node not in symbol_set]

    other_kwargs = dict(base_node_kwargs)
    other_kwargs.setdefault("edgecolors", "#444444")
    other_kwargs.setdefault("linewidths", 1.2)

    if other_nodes:
        other_colors = [
            type_to_color[graph.nodes[node]["type"]] for node in other_nodes
        ]
        other_collection = nx.draw_networkx_nodes(
            graph,
            pos,
            nodelist=other_nodes,
            node_color=other_colors,
            ax=ax,
            **other_kwargs,
        )
        if (
            non_symbol_linestyle
            and other_collection is not None
            and hasattr(other_collection, "set_linestyle")
        ):
            other_collection.set_linestyle(non_symbol_linestyle)

    if symbol_nodes:
        symbol_colors = [type_to_color[context.symbol_type_id] for _ in symbol_nodes]
        symbol_kwargs = dict(base_node_kwargs)
        if isinstance(base_node_size_value, numbers.Real):
            symbol_kwargs["node_size"] = base_node_size_value * symbol_node_scale
        else:
            symbol_kwargs["node_size"] = 300 * symbol_node_scale
        symbol_kwargs.setdefault("edgecolors", "black")
        symbol_kwargs.setdefault("linewidths", 2.4)
        symbol_collection = nx.draw_networkx_nodes(
            graph,
            pos,
            nodelist=symbol_nodes,
            node_color=symbol_colors,
            ax=ax,
            **symbol_kwargs,
        )
        if symbol_collection is not None:
            symbol_collection.set_facecolor(symbol_colors)
            symbol_collection.set_edgecolor("black")

    labels_to_draw = {}
    explicit_labels = set(label_nodes or [])
    if label_node_types:
        type_set = {t for t in label_node_types}
        explicit_labels.update(
            node
            for node, data in graph.nodes(data=True)
            if data.get("type") in type_set
        )
    if explicit_labels:
        labels_to_draw = {node: node for node in graph.nodes if node in explicit_labels}
    elif with_labels:
        labels_to_draw = {node: node for node in graph.nodes}

    if labels_to_draw:
        label_kwargs = {}
        if label_font_size is not None:
            label_kwargs["font_size"] = label_font_size
        nx.draw_networkx_labels(
            graph, pos, labels=labels_to_draw, ax=ax, **label_kwargs
        )

    # Edge coloring by argument position
    edge_attr_name = "position"

    # Split edges into standard (numerical) and LGAN (structural - linegraph)
    standard_edges = []
    standard_colors = []
    lgan_edges = []
    lgan_colors = []

    if graph.is_multigraph():
        all_edge_iter = graph.edges(keys=True, data=True)
    else:
        all_edge_iter = graph.edges(data=True)

    all_positions = []
    for e_info in all_edge_iter:
        data_dict = e_info[-1]
        p_val = data_dict.get(edge_attr_name)
        if p_val is not None:
            all_positions.append(p_val)

    unique_positions = list(dict.fromkeys(p for p in all_positions))
    edge_pos_to_color = {}
    if unique_positions:
        cmap = plt.get_cmap("Dark2")
        edge_pos_to_color = {
            val: cmap(i / max(1, len(unique_positions) - 1))
            for i, val in enumerate(unique_positions)
        }

    # Re-iterate to separate by style
    if graph.is_multigraph():
        all_edge_iter = graph.edges(keys=True, data=True)
    else:
        all_edge_iter = graph.edges(data=True)

    for e_info in all_edge_iter:
        u, v = e_info[0], e_info[1]
        data_dict = e_info[-1]
        p_val = data_dict.get(edge_attr_name)
        color = edge_pos_to_color.get(p_val, "#666666")

        if p_val in context.lgan_edge_positions:
            lgan_edges.append((u, v))
            lgan_colors.append(color)
        else:
            standard_edges.append((u, v))
            standard_colors.append(color)

    if edge_width is not None:
        edge_kwargs.setdefault("width", edge_width)
    if edge_alpha is not None:
        edge_kwargs.setdefault("alpha", edge_alpha)

    # Draw standard edges
    if standard_edges:
        nx.draw_networkx_edges(
            graph,
            pos,
            edgelist=standard_edges,
            edge_color=standard_colors,
            arrows=graph.is_directed(),
            ax=ax,
            **edge_kwargs,
        )

    # Draw LGAN (linegraph) edges as dashed
    if lgan_edges:
        lg_kwargs = dict(edge_kwargs)
        lg_kwargs["style"] = "dashed"
        nx.draw_networkx_edges(
            graph,
            pos,
            edgelist=lgan_edges,
            edge_color=lgan_colors,
            arrows=graph.is_directed(),
            ax=ax,
            **lg_kwargs,
        )

    if unique_types:
        from matplotlib.patches import Patch

        legend_handles = [
            Patch(facecolor=type_to_color[ntype], edgecolor="none", label=str(ntype))
            for ntype in unique_types
        ]
        node_title = "Node Types"
        if context.include_lgan_edges:
            node_title += "\n(Relations = Linegraph Nodes)"
        node_legend = ax.legend(
            handles=legend_handles,
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            frameon=False,
            title=node_title,
        )
        ax.add_artist(node_legend)

    if unique_positions:
        from matplotlib.lines import Line2D

        edge_handles = [
            Line2D(
                [0],
                [0],
                color=edge_pos_to_color[p_val],
                linestyle=(
                    "dashed" if p_val in context.lgan_edge_positions else "solid"
                ),
                label=(
                    f"LGAN ({p_val})"
                    if p_val in context.lgan_edge_positions
                    else f"pos: {p_val}"
                ),
            )
            for p_val in unique_positions
        ]
        ax.legend(
            handles=edge_handles,
            loc="lower left",
            bbox_to_anchor=(1.02, 0.0),
            frameon=False,
            title="Edge Roles",
        )

    ax.figure.subplots_adjust(right=0.8)

    draw_edge_labels = edge_labels or label_edges is not None
    if draw_edge_labels and unique_positions:
        labels: dict[tuple[Any, ...], Any] = {}
        if graph.is_multigraph():
            for u, v, k, data in graph.edges(keys=True, data=True):
                if (
                    label_edge_set is not None
                    and (u, v, k) not in label_edge_set
                    and (u, v) not in label_edge_set
                ):
                    continue
                labels[(u, v, k)] = data.get(edge_attr_name)
        else:
            for u, v, data in graph.edges(data=True):
                if label_edge_set is not None and (u, v) not in label_edge_set:
                    continue
                labels[(u, v)] = data.get(edge_attr_name)
        labels = {edge: value for edge, value in labels.items() if value is not None}
        label_kwargs = {}
        if label_font_size is not None:
            label_kwargs["font_size"] = label_font_size
        nx.draw_networkx_edge_labels(
            graph,
            pos,
            edge_labels=labels,
            font_color="black",
            ax=ax,
            **label_kwargs,
        )

    ax.set_axis_off()
    return ax


def draw_horizon(
    graph: nx.MultiGraph,
    *,
    context: HorizonVisualizationContext,
    ax=None,
    with_labels: bool = True,
    edge_labels: bool = True,
    node_kwargs: dict | None = None,
    edge_kwargs: dict | None = None,
    layout: dict | None = None,
    node_size: float | None = None,
    node_alpha: float | None = None,
    edge_width: float | None = None,
    edge_alpha: float | None = None,
    label_font_size: float | None = None,
    label_nodes: Iterable[str] | None = None,
    label_node_types: Iterable[str] | None = None,
    label_edges: Iterable[tuple[str, ...]] | None = None,
    align_target_nodes: bool = True,
    target_x_spacing: float = 4.0,
    target_y_spacing: float = 2.0,
    layout_seed: int | None = 7,
    symbol_node_scale: float = 1.5,
    non_symbol_linestyle: str | None = "--",
):
    # Structured, bipartite-inspired layout: objects (left), atoms (middle),
    # and optionally the target-tree (right).
    if layout is None:
        # derive spacing scale from node size to keep enough room for
        # family relation nodes; larger nodes -> larger spacing
        nk = node_kwargs or {}
        base_node_size_value = nk.get("node_size", node_size)
        if isinstance(base_node_size_value, (list, tuple)):
            base_node_size_value = (
                base_node_size_value[0] if base_node_size_value else None
            )
        if base_node_size_value is None:
            base_node_size_value = 300.0
        try:
            size_scale = float(base_node_size_value) / 300.0
        except Exception:
            size_scale = 1.0
        # Increase default spacing noticeably to create more room in the
        # target tree for family relations. Scale with node size and keep a
        # healthy minimum even for small nodes.
        spacing_scale = max(20.5, 2.0 * size_scale * max(1.0, symbol_node_scale))
        eff_x = target_x_spacing * spacing_scale
        eff_y = target_y_spacing * spacing_scale
        # identify categories
        symbol_nodes = []
        target_symbols = []
        for node, data in graph.nodes(data=True):
            if data.get("type") == context.hgraph.symbol_type_id:
                if align_target_nodes and "depth" in data:
                    target_symbols.append(node)
                else:
                    symbol_nodes.append(node)

        tree_relation_types = {
            context.parent_relation,
        }
        tree_relation_nodes = [
            n for n, d in graph.nodes(data=True) if d.get("type") in tree_relation_types
        ]
        # sibling/cousin relations are treated as regular middle relations
        middle_nodes = [
            n
            for n, d in graph.nodes(data=True)
            if d.get("type")
            not in tree_relation_types | {context.hgraph.symbol_type_id}
        ]

        fixed_positions: dict[str, tuple[float, float]] = {}

        # Determine a common vertical span based on the largest column layer
        def _count_per_depth(nodes: list[str]) -> dict[int, int]:
            depth_map: dict[int, int] = defaultdict(int)
            for name in nodes:
                d = int(graph.nodes[name].get("depth", 0))
                depth_map[d] += 1
            return depth_map

        depth_counts = _count_per_depth(target_symbols) if target_symbols else {}
        max_depth_count = max(depth_counts.values()) if depth_counts else 0
        H_count = max(len(symbol_nodes), len(middle_nodes), max_depth_count, 1)
        y_min, y_max = -0.5 * eff_y * (H_count - 1), 0.5 * eff_y * (H_count - 1)

        def _spread(nodes: list[str]) -> dict[str, float]:
            if not nodes:
                return {}
            nodes_sorted = sorted(nodes)
            if len(nodes_sorted) == 1:
                return {nodes_sorted[0]: 0.0}
            step = (y_max - y_min) / (len(nodes_sorted) - 1)
            return {n: y_min + i * step for i, n in enumerate(nodes_sorted)}

        # left column: object symbols (equidistant across common span)
        for n, y in _spread(symbol_nodes).items():
            fixed_positions[n] = (-2.0 * eff_x, y)

        # middle column: atoms/literals (non-tree relations) across common span
        for n, y in _spread(middle_nodes).items():
            fixed_positions[n] = (0.0, y)

        # right area: target symbols arranged by depth
        if target_symbols:
            depth_to_nodes: dict[int, list[str]] = defaultdict(list)
            for node in target_symbols:
                depth_to_nodes[int(graph.nodes[node].get("depth", 0))].append(node)
            for depth, nodes in sorted(depth_to_nodes.items()):
                if len(nodes) == 1:
                    y_positions = {nodes[0]: 0.0}
                else:
                    nodes_sorted = sorted(
                        nodes,
                        key=lambda name: (
                            graph.nodes[name].get(
                                "target_index",
                                target_index_from_name(
                                    name, context.target_symbol_prefix
                                ),
                            ),
                            name,
                        ),
                    )
                    step = (y_max - y_min) / (len(nodes_sorted) - 1)
                    y_positions = {
                        n: y_min + i * step for i, n in enumerate(nodes_sorted)
                    }
                for node, y in y_positions.items():
                    fixed_positions[node] = (
                        2.0 * eff_x + depth * eff_x,
                        y,
                    )

        # place tree relation nodes near their incident target symbols; for
        # symmetric relations (sibling/cousin) that connect the same two
        # targets twice (a->b and b->a), offset them in opposite directions
        # along the perpendicular to avoid overlap.
        for rel in tree_relation_nodes:
            rel_type = graph.nodes[rel].get("type")
            # collect pos0/pos1 neighbors if available
            pos0 = pos1 = None
            for nbr, edge_dict in graph[rel].items():
                if nbr not in fixed_positions:
                    continue
                for attrs in edge_dict.values():
                    p = attrs.get("position")
                    if p == 0:
                        pos0 = nbr
                    elif p == 1:
                        pos1 = nbr
            if (
                rel_type in {context.sibling_relation, context.cousin_relation}
                and pos0 is not None
                and pos1 is not None
            ):
                # compute canonical perpendicular using min->max target order
                try:
                    i0 = target_index_from_name(pos0, context.target_symbol_prefix)
                    i1 = target_index_from_name(pos1, context.target_symbol_prefix)
                except Exception:
                    i0, i1 = 0, 1
                a_name, b_name = (pos0, pos1) if i0 <= i1 else (pos1, pos0)
                xa, ya = fixed_positions[a_name]
                xb, yb = fixed_positions[b_name]
                mx, my = (xa + xb) / 2.0, (ya + yb) / 2.0
                dx, dy = (xb - xa), (yb - ya)
                dist = (dx * dx + dy * dy) ** 0.5 or 1e-6
                ox, oy = -dy / dist, dx / dist
                # place min->max on +perp and max->min on -perp
                is_min_to_max = i0 <= i1
                sign = 1.0 if is_min_to_max else -1.0
                offset = 0.4 * eff_y
                fixed_positions[rel] = (
                    mx + sign * offset * ox,
                    my + sign * offset * oy,
                )
            else:
                # fallback: average of available neighbors
                neighbors = [
                    nbr for nbr in graph.neighbors(rel) if nbr in fixed_positions
                ]
                if neighbors:
                    xs = [fixed_positions[n][0] for n in neighbors]
                    ys = [fixed_positions[n][1] for n in neighbors]
                    fixed_positions[rel] = (sum(xs) / len(xs), sum(ys) / len(ys))

        # Avoid spring_layout rescaling (which would squash spacing back
        # into a small box). Instead, use the fixed positions directly and
        # place any remaining nodes by averaging their neighbors.
        remaining = [n for n in graph.nodes if n not in fixed_positions]
        if remaining:
            for n in remaining:
                nbrs = [nb for nb in graph.neighbors(n) if nb in fixed_positions]
                if nbrs:
                    xs = [fixed_positions[nb][0] for nb in nbrs]
                    ys = [fixed_positions[nb][1] for nb in nbrs]
                    fixed_positions[n] = (sum(xs) / len(xs), sum(ys) / len(ys))
                else:
                    fixed_positions[n] = (0.0, 0.0)
        layout = fixed_positions

    ax = draw_hgraph(
        graph,
        context=context.hgraph,
        ax=ax,
        with_labels=with_labels,
        edge_labels=edge_labels,
        node_kwargs=node_kwargs,
        edge_kwargs=edge_kwargs,
        layout=layout,
        node_size=node_size,
        node_alpha=node_alpha,
        edge_width=edge_width,
        edge_alpha=edge_alpha,
        label_font_size=label_font_size,
        label_nodes=label_nodes,
        label_node_types=label_node_types,
        label_edges=label_edges,
        symbol_node_scale=symbol_node_scale,
        non_symbol_linestyle=non_symbol_linestyle,
    )

    return ax
