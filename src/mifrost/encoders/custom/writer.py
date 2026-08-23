"""Graph assembly facade over BatchBuilder for custom encoders."""

from __future__ import annotations

from collections.abc import Hashable, Sequence
from typing import Any

from ..._core import BatchBuilder, BatchEncoding
from .state_view import StateView
from .tables import EdgeSink, NodeTable, Vocabulary


class GraphWriter:
    """One-graph writer that owns a BatchBuilder, NodeTable and EdgeSink.

    A fresh writer is created per encoded graph; `finish` emits the graph and
    releases all accumulated content, after which the writer must not be
    reused.
    """

    def __init__(
        self,
        view: StateView,
        *,
        graph_kind: str = "homo",
        export_node_names: bool = True,
    ) -> None:
        """Start one empty graph against ``view``'s problem schema."""

        self.view = view
        self.graph_kind = graph_kind
        self.export_node_names = export_node_names
        self._builder = BatchBuilder()
        self._builder.set_graph_kind(graph_kind)
        self.nodes = NodeTable()
        self.edges = EdgeSink()
        self._vocabularies: dict[str, Vocabulary] = {}

    def add_node(
        self,
        key: Hashable,
        *,
        role: str,
        channels: Sequence[int] = (),
        name: str | None = None,
    ) -> int:
        """Intern one node in the writer's `NodeTable` and return its id."""

        return self.nodes.id_for(key, role=role, channels=channels, name=name)

    def add_edge(
        self,
        src: int,
        dst: int,
        kind: str,
        pos_a: int = 0,
        pos_b: int = 0,
    ) -> None:
        """Append one directed edge to the writer's `EdgeSink`."""

        self.edges.add(src, dst, kind, pos_a, pos_b)

    def add_both(
        self,
        src: int,
        dst: int,
        kind_fwd: str,
        kind_bwd: str,
        pos_a: int = 0,
        pos_b: int = 0,
    ) -> None:
        """Append an edge pair (forward plus reverse) to the edge sink."""

        self.edges.add_both(src, dst, kind_fwd, kind_bwd, pos_a, pos_b)

    def vocabulary(self, name: str) -> Vocabulary:
        """Return the named vocabulary (e.g. ``"predicates"``), creating it."""

        found = self._vocabularies.get(name)
        if found is None:
            found = Vocabulary()
            self._vocabularies[name] = found
        return found

    def set_vocab_attr(self, name: str) -> None:
        """Export the named vocabulary as graph attr ``vocab_<name>``."""

        vocab = self.vocabulary(name)
        self._builder.set_graph_attr(f"vocab_{name}", vocab.names())

    def set_attr(self, key: str, value: Any) -> None:
        """Set one graph-level attribute on the underlying builder."""

        self._builder.set_graph_attr(key, value)

    def set_flag(self, key: str, value: bool) -> None:
        """Set one schema flag on the underlying builder."""

        self._builder.set_schema_flag(key, bool(value))

    def register_field(self, key: str, spec: dict) -> None:
        """Register a graph-field spec on the underlying builder."""

        self._builder.register_field(key, spec)

    def set_field(self, key: str, values: Any) -> None:
        """Set a graph-field value on the underlying builder."""

        self._builder.set_field(key, values)

    def finish(self) -> BatchEncoding:
        """Emit the accumulated graph and release all retained content.

        Nodes are exported under type ``node`` with feature attr ``x_ids``
        and optional node names; edges under ``("node", "edge", "node")``
        with feature attr ``edge_attr`` of fixed width 3. Object names and
        the ``custom_encoder`` schema flag are always attached.
        """

        builder = self._builder
        nodes = self.nodes
        count = nodes.count
        if count > 0:
            builder.add_nodes("node", count)
            builder.add_node_features("node", "x_ids", nodes.to_float_array())
            names = nodes.names()
            if self.export_node_names and names is not None:
                builder.set_node_names("node", names)
        edge_index, edge_attr = self.edges.to_arrays()
        builder.ensure_edge_type("node", "edge", "node")
        builder.add_edge_features("node", "edge", "node", "edge_attr", edge_attr)
        if len(edge_attr) > 0:
            builder.add_edges("node", "edge", "node", edge_index[0], edge_index[1])
        builder.set_object_names(self.view.objects)
        builder.set_schema_flag("custom_encoder", True)
        builder.set_graph_attr("vocab_edge_kinds", self.edges.kinds.names())
        encoding = builder.build()
        self._builder = BatchBuilder()
        self._builder.set_graph_kind(self.graph_kind)
        self.nodes = NodeTable()
        self.edges = EdgeSink()
        self._vocabularies = {}
        return encoding


__all__ = ["GraphWriter"]
