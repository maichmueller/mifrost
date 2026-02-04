from __future__ import annotations

import itertools
import numbers
from collections import defaultdict
from typing import Iterable, Sequence

import networkx as nx
import pymimir
import torch
from torch_geometric.data import Data

from .base_encoder import (
    EncoderFactory,
    GraphEncoderBase,
    check_encoded_by_this,
)
from .pyg_batch_builder import PygDataBatchBuilder
from .pyg_builder import PygBuilder, PygBuilderBase
from .relation_formatter import relation_formatter
from .accessors import (
    atom_objects,
    literal_atom,
    literal_polarity,
    predicate,
    predicate_arity,
)


class ColorGraphEncoder(GraphEncoderBase[Data]):
    """
    A state encoder into an associated colored state-graph for a specified domain.

    Each object will receive its own node, each predicate might receive its own node (if configured to do so),
    each atom will receive multiple nodes which
    """

    def __init__(
        self,
        domain: pymimir.Domain,
        edge_features: bool = False,
        enable_global_predicate_nodes: bool = False,
    ):
        """
        Initialize the color graph encoder

        Parameters
        ----------
        domain: pymimir.Domain, the domain over which instance-states will be encoded
        enable_global_predicate_nodes: bool, whether to add summarising predicate nodes to the graph.
            Predicate nodes will connect with respective pos-0-atom nodes, if applicable.
        edge_features: bool, whether to add features to edges or to encode them in positional atom nodes
            (see `object-graph` in [1]).
        feature_map: FeatureMap,
            the used features for positional, goal, negation, etc. information.

        """
        self.predicate_nodes_enabled = enable_global_predicate_nodes
        self.edge_features = edge_features
        super().__init__(domain, builder=PygBuilder())
        self._ensure_config_hash()
        self._builder.encoder_hash = self.encoder_hash

    def _config_dict(self) -> dict:
        return {
            "edge_features": self.edge_features,
            "predicate_nodes_enabled": self.predicate_nodes_enabled,
        }

    def __eq__(self, other: ColorGraphEncoder):
        return (
            self.predicate_nodes_enabled == other.predicate_nodes_enabled
            and self.domain == other.domain
            and self.edge_features == other.edge_features
        )

    def as_factory(self) -> EncoderFactory:
        return EncoderFactory(
            self.__class__,
            kwargs={
                "enable_global_predicate_nodes": self.predicate_nodes_enabled,
                "edge_features": self.edge_features,
            },
        )

    def _encode(
        self,
        builder: PygBuilderBase,
        facts: Sequence[pymimir.GroundAtom],
        goals: Sequence[pymimir.GroundLiteral],
        actions: Sequence[pymimir.GroundAction],
        **kwargs,
    ):
        goal_level_map = builder.get_graph_attr("goal_level_map")
        color_counter = itertools.count(1)
        colormap: dict[str | None, int] = defaultdict(lambda: next(color_counter))
        colormap[None] = 0  # reserve color 0 for "no feature"

        for atom in facts:
            self._encode_facts_and_goals(
                atom,
                predicate(atom),
                atom_objects(atom),
                builder,
                goal_level=None,
                colormap=colormap,
            )
        for literal in goals:
            atom = literal_atom(literal)
            self._encode_facts_and_goals(
                literal,
                predicate(atom),
                atom_objects(atom),
                builder,
                goal_level=goal_level_map[literal],
                colormap=colormap,
            )
        if actions:
            raise NotImplementedError(
                f"Action encoding not supported in {self.__class__} at the moment. Contribution welcome."
            )

    def _encode_facts_and_goals(
        self,
        item: pymimir.GroundLiteral | pymimir.GroundAtom,
        predicate: pymimir.Predicate,
        objects: Sequence[pymimir.Object],
        builder: PygBuilderBase,
        goal_level: int | None,
        colormap: dict[str | None, int],
    ):
        polarity = (
            literal_polarity(item) if isinstance(item, pymimir.GroundLiteral) else None
        )
        if self.predicate_nodes_enabled:
            predicate_node = relation_formatter(
                predicate,
                goal_level=goal_level,
                polarity=polarity,
            )
            if self.edge_features:
                builder.add_node(predicate_node)
                builder.add_edge(predicate_node, predicate_node)
                builder.set_edge_attr(
                    predicate_node,
                    predicate_node,
                    "type",
                    colormap[
                        relation_formatter(
                            predicate,
                            pos=None,
                            goal_level=goal_level,
                            polarity=polarity,
                        )
                    ],
                )
            else:
                builder.add_node(predicate_node, type=colormap[None])

        if predicate_arity(predicate) == 0:
            item_node = relation_formatter(
                item,
                goal_level=goal_level,
            )
            color = colormap[
                relation_formatter(
                    item,
                    goal_level=goal_level,
                    polarity=polarity,
                )
            ]
            if self.edge_features:
                builder.add_node(item_node)
                builder.add_edge(item_node, item_node)
                builder.set_edge_attr(item_node, item_node, "type", color)
            else:
                builder.add_node(item_node, type=color)
            return

        prev_item_node = None
        for pos, obj in enumerate(objects):
            object_node = relation_formatter(obj)
            color = colormap[
                relation_formatter(
                    item,
                    pos,
                    goal_level=goal_level,
                    polarity=polarity,
                )
            ]
            if self.edge_features:
                item_node = relation_formatter(
                    item,
                    goal_level=goal_level,
                    polarity=polarity,
                )
                builder.add_node(object_node)
                builder.add_node(item_node)
                builder.add_edge(object_node, item_node)
                builder.set_edge_attr(object_node, item_node, "type", color)
                if self.predicate_nodes_enabled and pos == 0:
                    builder.add_edge(predicate_node, item_node)
                    builder.set_edge_attr(
                        predicate_node,
                        item_node,
                        "type",
                        colormap[
                            relation_formatter(
                                item,
                                pos=None,
                                goal_level=goal_level,
                                polarity=polarity,
                            )
                        ],
                    )
            else:
                item_node = relation_formatter(
                    item,
                    pos,
                    goal_level=goal_level,
                    polarity=polarity,
                )
                builder.add_node(object_node, type=colormap[None])
                builder.add_node(item_node, type=color)
                builder.add_edge(object_node, item_node)
                if pos > 0:
                    builder.add_edge(prev_item_node, item_node)
                elif self.predicate_nodes_enabled:
                    builder.add_edge(predicate_node, item_node)
            prev_item_node = item_node

    def _build_data(self, builder: PygBuilderBase) -> Data:
        data = Data()
        node_keys = builder.node_keys[PygBuilder._NODE_TYPE]
        num_nodes = len(node_keys)
        data.node_names = list(node_keys)

        if not self.edge_features:
            node_types = builder.node_attrs[PygBuilder._NODE_TYPE].get("type", [])
            if node_types:
                data.x = torch.as_tensor(node_types).view(-1, 1)

        indices = []
        edge_attr_values = []
        for edge_key, edges in builder.edge_indices.items():
            attr_list = builder.edge_attrs.get(edge_key, {}).get("type", [])
            for idx, (src_idx, dst_idx) in enumerate(edges):
                value = attr_list[idx] if idx < len(attr_list) else None
                indices.append((src_idx, dst_idx))
                if self.edge_features:
                    edge_attr_values.append(value)
                if src_idx != dst_idx:
                    indices.append((dst_idx, src_idx))
                    if self.edge_features:
                        edge_attr_values.append(value)

        if not indices:
            data.edge_index = torch.empty((2, 0), dtype=torch.long)
        else:
            data.edge_index = torch.tensor(indices, dtype=torch.long).t().contiguous()

        if self.edge_features:
            if edge_attr_values:
                data.edge_attr = torch.as_tensor(edge_attr_values).view(-1, 1)
            else:
                data.edge_attr = None
            if data.x is None and data.pos is None:
                data.num_nodes = num_nodes
        else:
            data.edge_attr = None

        return data

    def _new_batch_builder(self) -> PygDataBatchBuilder:
        return PygDataBatchBuilder()

    def to_networkx(self, data: Data) -> nx.Graph:
        """Reconstruct a NetworkX graph from Data produced by this encoder."""
        graph = nx.Graph()
        node_names = getattr(data, "node_names", None)
        if not node_names:
            if hasattr(data, "x") and data.x is not None:
                count = data.x.shape[0]
            else:
                count = getattr(data, "num_nodes", 0) or 0
            node_names = [str(i) for i in range(count)]

        for i, name in enumerate(node_names):
            val = data.x[i] if hasattr(data, "x") and data.x is not None else 0
            attrs = {"type": val.item() if torch.is_tensor(val) else val}
            if hasattr(data, "goal_level") and data.goal_level is not None:
                gval = data.goal_level[i]
                attrs["goal_level"] = gval.item() if torch.is_tensor(gval) else gval
            graph.add_node(name, **attrs)

        if hasattr(data, "edge_index"):
            for i in range(data.edge_index.shape[1]):
                u_idx = data.edge_index[0, i].item()
                v_idx = data.edge_index[1, i].item()
                u_name = node_names[u_idx]
                v_name = node_names[v_idx]
                attrs = {}
                if hasattr(data, "edge_attr") and data.edge_attr is not None:
                    val = data.edge_attr[i]
                    attrs["type"] = (
                        int(val.item()) if torch.is_tensor(val) else int(val)
                    )
                graph.add_edge(u_name, v_name, **attrs)

        graph.graph["encoder_hash"] = getattr(data, "encoder_hash", None)
        return graph

    @check_encoded_by_this
    def draw(
        self,
        data: Data,
        *,
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

        node_kwargs = node_kwargs or {}
        edge_kwargs = edge_kwargs or {}

        if ax is None:
            _, ax = plt.subplots()

        if hasattr(data, "edge_types"):  # HeteroData
            graph = self.to_networkx(data)
        elif hasattr(data, "edge_index") and not hasattr(
            data, "edge_types"
        ):  # Homogeneous Data
            graph = self.to_networkx(data)
        else:
            graph = data

        pos = layout or nx.spring_layout(graph)

        node_types = [graph.nodes[n].get("type", 0) for n in graph.nodes]
        unique_types = list(dict.fromkeys(node_types))
        type_to_color: dict[int, tuple[float, float, float, float]] = {}
        if unique_types:
            cmap = plt.get_cmap("tab20_r")
            type_to_color = {
                t: cmap(i / max(1, len(unique_types) - 1))
                for i, t in enumerate(unique_types)
            }
        default_node_color = "#BBBBBB"

        base_node_kwargs = dict(node_kwargs)
        if node_size is not None:
            base_node_kwargs.setdefault("node_size", node_size)
        if node_alpha is not None:
            base_node_kwargs.setdefault("alpha", node_alpha)

        base_node_size_value = base_node_kwargs.get("node_size")
        if isinstance(base_node_size_value, Sequence) and not isinstance(
            base_node_size_value, (str, bytes)
        ):
            base_node_size_value = (
                base_node_size_value[0] if base_node_size_value else None
            )
        if base_node_size_value is None:
            inferred_size = node_size if node_size is not None else 300
            base_node_kwargs.setdefault("node_size", inferred_size)
            base_node_size_value = inferred_size
        elif not isinstance(base_node_size_value, numbers.Real):
            base_node_size_value = 300

        symbol_type = min(unique_types) if unique_types else None
        symbol_nodes: list[str] = []
        if symbol_type is not None:
            symbol_nodes = [
                node
                for node, data in graph.nodes(data=True)
                if data.get("type") == symbol_type
            ]
        symbol_set = set(symbol_nodes)
        other_nodes = [node for node in graph.nodes if node not in symbol_set]

        other_kwargs = dict(base_node_kwargs)
        other_kwargs.setdefault("edgecolors", "#444444")
        other_kwargs.setdefault("linewidths", 1.2)

        if other_nodes:
            other_colors = [
                type_to_color.get(graph.nodes[node].get("type"), default_node_color)
                for node in other_nodes
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
            symbol_colors = [
                type_to_color.get(symbol_type, default_node_color) for _ in symbol_nodes
            ]
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
            labels_to_draw = {
                node: node for node in graph.nodes if node in explicit_labels
            }
        elif with_labels:
            labels_to_draw = {node: node for node in graph.nodes}

        if labels_to_draw:
            label_kwargs = {}
            if label_font_size is not None:
                label_kwargs["font_size"] = label_font_size
            nx.draw_networkx_labels(
                graph, pos, labels=labels_to_draw, ax=ax, **label_kwargs
            )

        edge_label_set = None
        if label_edges:
            edge_label_set = {tuple(edge) for edge in label_edges}

        edge_attr_name = None
        if any("position" in data for _, _, data in graph.edges(data=True)):
            edge_attr_name = "position"
        elif any("type" in data for _, _, data in graph.edges(data=True)):
            edge_attr_name = "type"

        if edge_attr_name:
            edge_values = [
                data.get(edge_attr_name) for _, _, data in graph.edges(data=True)
            ]
            unique_values = [
                val for val in dict.fromkeys(edge_values) if val is not None
            ]
            cmap = plt.get_cmap("Dark2")
            value_to_color = {
                val: cmap(i / max(1, len(unique_values) - 1))
                for i, val in enumerate(unique_values)
            }
            edge_colors = [value_to_color.get(val, "#666666") for val in edge_values]
        else:
            edge_colors = "#666666"

        if edge_width is not None:
            edge_kwargs.setdefault("width", edge_width)
        if edge_alpha is not None:
            edge_kwargs.setdefault("alpha", edge_alpha)

        nx.draw_networkx_edges(
            graph,
            pos,
            edge_color=edge_colors,
            ax=ax,
            **edge_kwargs,
        )

        draw_edge_labels = edge_labels or label_edges is not None
        if draw_edge_labels and edge_attr_name:
            labels = {}
            for u, v, data in graph.edges(data=True):
                if edge_label_set is not None and (u, v) not in edge_label_set:
                    continue
                labels[(u, v)] = data.get(edge_attr_name)
            if labels:
                label_kwargs = {}
                if label_font_size is not None:
                    label_kwargs["font_size"] = label_font_size
                nx.draw_networkx_edge_labels(
                    graph,
                    pos,
                    edge_labels=labels,
                    ax=ax,
                    **label_kwargs,
                )

        ax.set_axis_off()
        return ax
