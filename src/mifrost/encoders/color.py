from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Iterable, Mapping

import torch
from torch_geometric.data import Data

if TYPE_CHECKING:
    import networkx as nx

from .. import _neutral_core
from ..backends._color_runtime import ColorBackendName, create_color_runtime
from .base import (
    ActionBatchInput,
    ActionBatchParam,
    CollateSpecParam,
    EncoderBase,
    GoalBatchInput,
    GoalBatchParam,
    StateBatchInput,
    StreamEncoderBase,
    SubgoalLayersInput,
    SubgoalLayersBatchParam,
)
from .types import (
    DomainInput,
    GoalLiteralInput,
    HomoEncoding,
    StateInput,
)


@dataclass
class ColorEncoderStream(StreamEncoderBase[Data]):
    """Streaming wrapper for ``ColorEncoder``."""

    _encoder: "ColorEncoder"

    def __post_init__(self) -> None:
        """Initialize an empty homo builder for streaming."""
        self._stream = self._encoder._runtime.make_stream()
        self._reset_builder()

    def append(
        self,
        state: StateInput,
        *,
        goals: Iterable[GoalLiteralInput] | None = None,
        subgoal_layers: Iterable[Iterable[GoalLiteralInput]] | None = None,
    ) -> int:
        """Append one state encoding to the color stream."""
        return self._coerce_stream_id(
            self._stream.append(state, goals=goals, subgoal_layers=subgoal_layers)
        )

    def remove(self, stream_id: int) -> None:
        self._stream.remove(stream_id)

    def update(
        self,
        stream_id: int,
        state: StateInput,
        *,
        goals: Iterable[GoalLiteralInput] | None = None,
        subgoal_layers: Iterable[Iterable[GoalLiteralInput]] | None = None,
    ) -> None:
        self._stream.update(
            stream_id, state, goals=goals, subgoal_layers=subgoal_layers
        )

    def _reset_builder(self) -> None:
        """Reset stream accumulation state."""
        self._stream.reset()


class ColorEncoder(EncoderBase[Data]):
    """
    Homogeneous color encoder backed by ``ColorEncoderEngine``.

    Produces compact ``Data``/``Batch`` outputs with integer-like node/edge
    attributes suitable for color-based graph models.
    """

    def __init__(
        self,
        domain: DomainInput,
        *,
        backend: ColorBackendName | str | None = None,
        edge_features: bool = False,
        enable_global_predicate_nodes: bool = False,
        export_node_names: bool = True,
    ) -> None:
        """Create a color encoder for one domain."""
        config = _neutral_core.SemanticColorEncoderConfig(
            edge_features=edge_features,
            enable_global_predicate_nodes=enable_global_predicate_nodes,
            export_node_names=export_node_names,
        )
        self._runtime = create_color_runtime(domain, config, backend=backend)
        self._engine = self._runtime.engine
        self._config = config
        self.backend = self._runtime.backend_name
        self.edge_features = edge_features
        self.predicate_nodes_enabled = enable_global_predicate_nodes
        self.export_node_names = export_node_names

    @property
    def engine(self) -> Any:
        """Expose the underlying C++ color engine."""
        return self._engine

    @property
    def config(self) -> Any:
        """Expose the backend-neutral resolved Color config."""
        return self._config

    def _encode(
        self,
        state: StateInput,
        *,
        goals: GoalBatchInput = None,
        actions: ActionBatchInput = None,
        subgoal_layers: SubgoalLayersInput = None,
    ) -> HomoEncoding:
        """Encode one state into homogeneous encoding dictionary."""
        return self._runtime.encode_one(
            state,
            goals=goals,
            actions=actions,
            subgoal_layers=subgoal_layers,
        )

    def encode(
        self,
        state: StateInput,
        *,
        goals: GoalBatchInput = None,
        actions: ActionBatchInput = None,
        subgoal_layers: SubgoalLayersInput = None,
        include_metadata: bool = True,
        **kwargs,
    ) -> HomoEncoding:
        """Encode one state into native ``BatchEncoding``."""
        return super().encode(
            state,
            goals=goals,
            actions=actions,
            subgoal_layers=subgoal_layers,
            include_metadata=include_metadata,
            **kwargs,
        )

    def _encode_batch(
        self,
        states: StateBatchInput,
        *,
        goals: GoalBatchParam = None,
        actions: ActionBatchParam = None,
        subgoal_layers: SubgoalLayersBatchParam = None,
    ) -> HomoEncoding:
        """Encode one or many states into homogeneous batch encoding_dict."""
        return self._runtime.encode_batch(
            states,
            goals=goals,
            actions=actions,
            subgoal_layers=subgoal_layers,
        )

    def encode_batch(
        self,
        states: StateBatchInput,
        *,
        goals: GoalBatchParam = None,
        actions: ActionBatchParam = None,
        subgoal_layers: SubgoalLayersBatchParam = None,
        batch_attrs: Mapping[str, Any] | None = None,
        collate_spec: CollateSpecParam = None,
        include_metadata: bool = True,
        **kwargs,
    ) -> HomoEncoding:
        """Encode one or many states into native ``BatchEncoding``."""
        return super().encode_batch(
            states,
            goals=goals,
            actions=actions,
            subgoal_layers=subgoal_layers,
            batch_attrs=batch_attrs,
            collate_spec=collate_spec,
            include_metadata=include_metadata,
            **kwargs,
        )

    def stream(self) -> ColorEncoderStream:
        """Create a streaming encoder sharing this encoder's C++ engine."""
        return ColorEncoderStream(self)

    def to_networkx(self, data: Data) -> nx.Graph:
        """Convert a color-encoded PyG graph into a NetworkX graph."""
        import networkx as nx

        graph = nx.Graph()
        node_names = getattr(data, "node_names", None)
        if not node_names:
            if hasattr(data, "x") and data.x is not None:
                count = data.x.shape[0]
            else:
                count = data.num_nodes
            node_names = [str(i) for i in range(count)]

        has_scalar_x = (
            hasattr(data, "x")
            and data.x is not None
            and torch.is_tensor(data.x)
            and data.x.dim() == 2
            and data.x.size(1) > 0
        )
        for i, name in enumerate(node_names):
            val = data.x[i] if has_scalar_x else 0
            attrs = {"type": val.item() if torch.is_tensor(val) else val}
            if hasattr(data, "goal_level") and data.goal_level is not None:
                gval = data.goal_level[i]
                attrs["goal_level"] = gval.item() if torch.is_tensor(gval) else gval
            graph.add_node(name, **attrs)

        for i in range(data.edge_index.shape[1]):
            u_idx = data.edge_index[0, i].item()
            v_idx = data.edge_index[1, i].item()
            u_name = node_names[u_idx]
            v_name = node_names[v_idx]
            attrs = {}
            if hasattr(data, "edge_attr") and data.edge_attr is not None:
                val = data.edge_attr[i]
                attrs["type"] = int(val.item()) if torch.is_tensor(val) else int(val)
            graph.add_edge(u_name, v_name, **attrs)

        graph.graph["encoder_hash"] = getattr(data, "encoder_hash", None)
        return graph

    def draw(
        self,
        data: Data,
        *,
        with_labels: bool = True,
        edge_labels: bool = False,
        ax: Any | None = None,
        node_size: int = 300,
        font_size: int = 8,
    ) -> Any:
        """Render a color-encoded graph with matplotlib and return the axis."""
        import networkx as nx

        try:
            import matplotlib.pyplot as plt
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "ColorEncoder.draw requires matplotlib to be installed"
            ) from exc

        graph = self.to_networkx(data)
        if ax is None:
            _, ax = plt.subplots()

        positions = nx.spring_layout(graph, seed=0)
        nx.draw_networkx(
            graph,
            pos=positions,
            ax=ax,
            with_labels=with_labels,
            node_size=node_size,
            font_size=font_size,
        )

        if edge_labels:
            labels = {}
            for src, dst, attrs in graph.edges(data=True):
                if "type" in attrs:
                    labels[(src, dst)] = attrs["type"]
            if labels:
                nx.draw_networkx_edge_labels(
                    graph,
                    pos=positions,
                    edge_labels=labels,
                    ax=ax,
                    font_size=max(6, font_size - 1),
                )

        return ax


__all__ = ["ColorEncoder", "ColorEncoderStream"]
