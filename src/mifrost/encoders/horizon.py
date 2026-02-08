from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import networkx as nx
from torch_geometric.data import HeteroData

from .._core import (
    BatchBuilder,
    DEFAULT_HISTORY_LINK_RELATION,
    DEFAULT_LGAN_NN_EDGE_POS,
    DEFAULT_SYMBOL_TYPE_ID,
    GoalInputs,
    HorizonEncoderConfig,
    HorizonHGraphEncoderEngine,
    HorizonStreamEncoder as _HorizonStreamEncoder,
    TransitionDAG,
)
from .base import (
    ActionBatchInput,
    GoalBatchInput,
    StateBatchInput,
    StreamEncoderBase,
    SubgoalLayersInput,
)
from .common import _advanced_state, _parts_to_pyg, _split_goals
from .hgraph import HGraphEncoder
from .types import (
    DomainInput,
    GoalLiteralInput,
    StateInput,
    default_goals_from_state,
    is_goal_literal_input,
    is_state_input,
)


def _ensure_dag(root: StateInput, dag: TransitionDAG | None) -> TransitionDAG:
    """Return an explicit DAG or create a default single-root DAG."""
    if dag is not None:
        return dag
    adv_root = _advanced_state(root)
    return TransitionDAG(adv_root)


def _prepare_horizon_goals(
    root: StateInput,
    goals: GoalBatchInput,
    subgoal_layers: SubgoalLayersInput,
) -> GoalInputs:
    """Resolve user-provided or problem-default goals into ``GoalInputs``."""
    if goals is None:
        goals = default_goals_from_state(root)
    inputs = _split_goals(goals, subgoal_layers)
    return inputs


def _is_literal(value: object) -> bool:
    """Best-effort literal type probe for per-state goal detection."""
    return is_goal_literal_input(value)


@dataclass
class HorizonEncoderStream(StreamEncoderBase[HeteroData]):
    """Streaming wrapper for ``HorizonHGraphEncoderEngine``."""

    _engine: HorizonHGraphEncoderEngine

    def __post_init__(self) -> None:
        """Initialize an empty hetero builder for streaming."""
        self._stream = _HorizonStreamEncoder(self._engine)
        self._reset_builder()

    def append(
        self,
        root: StateInput,
        dag: TransitionDAG | None = None,
        *,
        goals: Iterable[GoalLiteralInput] | None = None,
        subgoal_layers: Iterable[Iterable[GoalLiteralInput]] | None = None,
    ) -> int:
        """Append one root/DAG encoding to the stream."""
        adv_root = _advanced_state(root)
        dag = _ensure_dag(root, dag)
        inputs = _prepare_horizon_goals(root, goals, subgoal_layers)
        return self._coerce_stream_id(self._stream.append(adv_root, dag, inputs))

    def remove(self, stream_id: int) -> None:
        self._stream.remove(stream_id)

    def update(
        self,
        stream_id: int,
        root: StateInput,
        dag: TransitionDAG | None = None,
        *,
        goals: Iterable[GoalLiteralInput] | None = None,
        subgoal_layers: Iterable[Iterable[GoalLiteralInput]] | None = None,
    ) -> None:
        adv_root = _advanced_state(root)
        dag = _ensure_dag(root, dag)
        inputs = _prepare_horizon_goals(root, goals, subgoal_layers)
        self._stream.update(stream_id, adv_root, dag, inputs)

    def _reset_builder(self) -> None:
        """Reset stream accumulation state."""
        self._stream.reset()

    def _flush_batch_encoding_py_impl(self) -> Mapping[str, object]:
        return self._stream.flush_batch_encoding_py()

    def _parts_to_pyg(
        self,
        parts: Mapping[str, object],
        *,
        as_batch: bool,
        include_metadata: bool = True,
    ) -> HeteroData:
        return _parts_to_pyg(
            parts, as_batch=as_batch, include_metadata=include_metadata
        )


class HorizonEncoder(HGraphEncoder):
    """
    Horizon lookahead encoder backed by ``HorizonHGraphEncoderEngine``.

    This encoder combines a root state, a ``TransitionDAG`` and goals into one
    hetero graph representation.
    """

    def __init__(
        self,
        domain: DomainInput,
        *,
        transition_mode: object | None = None,
        target_symbol_prefix: str | None = None,
        parent_relation: str | None = None,
        sibling_relation: str | None = None,
        cousin_relation: str | None = None,
        enable_parent_relation: bool | None = None,
        enable_sibling_relation: bool | None = None,
        enable_cousin_relation: bool | None = None,
        exclude_root_candidate: bool | None = None,
        max_goal_level: int | None = None,
        symbol_type_id: str | None = DEFAULT_SYMBOL_TYPE_ID,
        ignore_actions: bool | None = None,
        add_nullary_predicates: bool | None = None,
        include_lgan_edges: bool | None = None,
        include_static: bool | None = None,
        include_empty_edge_types: bool | None = None,
        support_literals: bool | None = None,
        nullary_object_name: str | None = None,
        lgan_nn_edge_pos: str | None = DEFAULT_LGAN_NN_EDGE_POS,
        history_link_relation: str | None = DEFAULT_HISTORY_LINK_RELATION,
    ) -> None:
        """Create a horizon encoder for one domain."""
        super().__init__(
            domain,
            symbol_type_id=symbol_type_id,
            ignore_actions=ignore_actions,
            add_nullary_predicates=add_nullary_predicates,
            include_lgan_edges=include_lgan_edges,
            include_static=include_static,
            include_empty_edge_types=include_empty_edge_types,
            max_goal_level=max_goal_level,
            support_literals=support_literals,
            nullary_object_name=nullary_object_name,
            lgan_nn_edge_pos=lgan_nn_edge_pos,
            history_link_relation=history_link_relation,
            _config_cls=HorizonEncoderConfig,
            _engine_cls=HorizonHGraphEncoderEngine,
            transition_mode=transition_mode,
            target_symbol_prefix=target_symbol_prefix,
            parent_relation=parent_relation,
            sibling_relation=sibling_relation,
            cousin_relation=cousin_relation,
            enable_parent_relation=enable_parent_relation,
            enable_sibling_relation=enable_sibling_relation,
            enable_cousin_relation=enable_cousin_relation,
            exclude_root_candidate=exclude_root_candidate,
        )
        config = self.config
        self.target_symbol_prefix = config.target_symbol_prefix
        self.parent_relation = config.parent_relation
        self.sibling_relation = config.sibling_relation
        self.cousin_relation = config.cousin_relation

    @property
    def engine(self) -> HorizonHGraphEncoderEngine:
        """Expose the underlying C++ horizon engine."""
        return self._engine

    def encode_parts(
        self,
        root: StateInput,
        dag: TransitionDAG | None = None,
        *,
        goals: GoalBatchInput = None,
        actions: ActionBatchInput = None,
        subgoal_layers: SubgoalLayersInput = None,
        **_: object,
    ) -> Mapping[str, object]:
        """Encode one root/DAG pair into parts."""
        if actions is not None:
            # Horizon encoding does not consume actions directly.
            _ = actions
        adv_root = _advanced_state(root)
        dag = _ensure_dag(root, dag)
        inputs = _prepare_horizon_goals(root, goals, subgoal_layers)
        return self._engine.encode(adv_root, dag, inputs)

    def encode(
        self,
        root: StateInput,
        dag: TransitionDAG | None = None,
        *,
        goals: GoalBatchInput = None,
        subgoal_layers: SubgoalLayersInput = None,
        include_metadata: bool = True,
        **kwargs: object,
    ) -> HeteroData:
        """Encode one root/DAG pair into ``HeteroData``."""
        return super().encode(
            root,
            goals=goals,
            subgoal_layers=subgoal_layers,
            dag=dag,
            include_metadata=include_metadata,
            **kwargs,
        )

    def _encode_batch_parts(
        self,
        roots: StateBatchInput,
        dags: Iterable[TransitionDAG] | TransitionDAG | None = None,
        *,
        goals: GoalBatchInput | Sequence[Iterable[GoalLiteralInput]] = None,
        subgoal_layers: SubgoalLayersInput = None,
    ) -> Mapping[str, object]:
        """Internal batch implementation shared by public batch APIs."""
        if is_state_input(roots):
            root_list = [roots]
        else:
            if isinstance(roots, (str, bytes)):
                raise TypeError("encode_batch expects a state or an iterable of states")
            root_list = list(roots)

        if dags is None:
            dag_list = [None] * len(root_list)
        elif isinstance(dags, TransitionDAG):
            dag_list = [dags]
        else:
            dag_list = list(dags)
        if len(dag_list) != len(root_list):
            raise ValueError("dags length must match roots length")

        goals_per_state: list[Iterable[GoalLiteralInput]] | None = None
        if (
            goals is not None
            and isinstance(goals, Sequence)
            and len(goals) == len(root_list)
        ):
            first = goals[0] if goals else None
            if first is not None and not _is_literal(first):
                goals_per_state = list(goals)

        shared_inputs: GoalInputs | None = None
        if goals is not None and goals_per_state is None:
            shared_inputs = _split_goals(goals, subgoal_layers)

        builder = BatchBuilder()
        builder.set_graph_kind("hetero")
        for idx, root in enumerate(root_list):
            adv_root = _advanced_state(root)
            dag = _ensure_dag(root, dag_list[idx])
            if goals_per_state is not None:
                inputs = _split_goals(goals_per_state[idx], subgoal_layers)
            else:
                inputs = (
                    shared_inputs
                    if shared_inputs is not None
                    else _prepare_horizon_goals(root, None, subgoal_layers)
                )
            self._engine.encode(adv_root, dag, inputs, builder)
            builder.next_graph()
        return builder.build_batch_encoding_py()

    def encode_batch(
        self,
        roots: StateBatchInput,
        dags: Iterable[TransitionDAG] | TransitionDAG | None = None,
        *,
        goals: GoalBatchInput | Sequence[Iterable[GoalLiteralInput]] = None,
        subgoal_layers: SubgoalLayersInput = None,
        include_metadata: bool = True,
        **kwargs: object,
    ) -> HeteroData:
        """Encode one or many root/DAG pairs into batched ``HeteroData``."""
        return super().encode_batch(
            roots,
            goals=goals,
            subgoal_layers=subgoal_layers,
            dags=dags,
            include_metadata=include_metadata,
            **kwargs,
        )

    def _accepted_kwargs(self) -> set[str]:
        """Accept transition DAG kwargs in the generic base API."""
        return {"dag", "dags"}

    def _parts_to_pyg(
        self,
        parts: Mapping[str, object],
        *,
        as_batch: bool,
        include_metadata: bool = True,
    ) -> HeteroData:
        return _parts_to_pyg(
            parts, as_batch=as_batch, include_metadata=include_metadata
        )

    def encode_batch_parts(
        self,
        roots: StateBatchInput,
        dags: Iterable[TransitionDAG] | TransitionDAG | None = None,
        *,
        goals: GoalBatchInput | Sequence[Iterable[GoalLiteralInput]] = None,
        subgoal_layers: SubgoalLayersInput = None,
    ) -> Mapping[str, object]:
        """Encode one or many root/DAG pairs into batch parts."""
        return self._encode_batch_parts(
            roots, dags, goals=goals, subgoal_layers=subgoal_layers
        )

    def stream(self) -> HorizonEncoderStream:
        """Create a streaming encoder sharing this encoder's C++ engine."""
        return HorizonEncoderStream(self._engine)

    def to_networkx(self, data: HeteroData) -> nx.MultiGraph:
        """Convert encoded horizon ``HeteroData`` into a named multigraph."""
        graph = nx.MultiGraph()
        symbol_type = self.symbol_type_id
        parent_type = getattr(data, "parent_relation", self.parent_relation)

        node_names_by_type: dict[str, list[str]] = {}
        for node_type in data.node_types:
            storage = data[node_type]
            names = list(getattr(storage, "node_names", []))
            node_names_by_type[node_type] = names

        symbol_nodes = node_names_by_type.get(symbol_type, [])
        target_names = list(getattr(data, "target_names", []))
        target_depths = list(getattr(data, "target_depths", []))
        target_indices = list(getattr(data, "target_indices", []))
        target_positions = list(getattr(data, "target_positions", []))
        object_names = list(getattr(data, "object_names", []))

        target_info: dict[int, tuple[str, int | None, int | None]] = {}
        for pos, sym_idx in enumerate(target_positions):
            if sym_idx < 0 or sym_idx >= len(symbol_nodes):
                continue
            target_name = (
                target_names[pos] if pos < len(target_names) else symbol_nodes[sym_idx]
            )
            depth = target_depths[pos] if pos < len(target_depths) else None
            index = target_indices[pos] if pos < len(target_indices) else None
            target_info[sym_idx] = (target_name, depth, index)

        object_iter = iter(object_names)
        for idx, node_key in enumerate(symbol_nodes):
            if idx in target_info:
                target_name, depth, index = target_info[idx]
                graph.add_node(
                    node_key,
                    type=symbol_type,
                    name=target_name,
                    depth=depth,
                    target_index=index,
                )
            else:
                object_name = next(object_iter, node_key)
                graph.add_node(node_key, type=symbol_type, name=object_name)

        for other_type, names in node_names_by_type.items():
            if other_type == symbol_type:
                continue
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
                if src_idx < 0 or src_idx >= len(src_names):
                    continue
                if dst_idx < 0 or dst_idx >= len(dst_names):
                    continue
                src_name = src_names[src_idx]
                dst_name = dst_names[dst_idx]
                graph.add_edge(src_name, dst_name, position=position)

        if parent_type in node_names_by_type:
            target_name_to_idx = {
                name: idx for idx, name in enumerate(target_names or symbol_nodes)
            }
            for transition_name in node_names_by_type[parent_type]:
                parent_idx = None
                child_idx = None
                for neighbor, edge_dict in graph[transition_name].items():
                    for edge_data in edge_dict.values():
                        position = edge_data.get("position")
                        if position == 0:
                            parent_idx = target_name_to_idx.get(neighbor)
                        elif position == 1:
                            child_idx = target_name_to_idx.get(neighbor)
                if parent_idx is not None:
                    graph.nodes[transition_name]["parent"] = parent_idx
                if child_idx is not None:
                    graph.nodes[transition_name]["child"] = child_idx
        return graph

    def _target_index_from_name(self, name: str) -> int:
        """Extract the integer index from target node names like ``target:3``."""
        if not name.startswith(self.target_symbol_prefix):
            return -1
        remainder = name[len(self.target_symbol_prefix) :]
        if not remainder.isdigit():
            return -1
        try:
            return int(remainder)
        except ValueError:
            return -1

    def draw(
        self,
        graph: nx.MultiGraph | HeteroData,
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
        align_target_nodes: bool = True,
        target_x_spacing: float = 4.0,
        target_y_spacing: float = 2.0,
        layout_seed: int | None = 7,
        symbol_node_scale: float = 1.5,
        non_symbol_linestyle: str | None = "--",
    ):
        if hasattr(graph, "edge_types"):  # HeteroData
            graph = self.to_networkx(graph)
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
                if data.get("type") == self.symbol_type_id:
                    if align_target_nodes and "depth" in data:
                        target_symbols.append(node)
                    else:
                        symbol_nodes.append(node)

            tree_relation_types = {
                self.parent_relation,
            }
            tree_relation_nodes = [
                n
                for n, d in graph.nodes(data=True)
                if d.get("type") in tree_relation_types
            ]
            # sibling/cousin relations are treated as regular middle relations
            middle_nodes = [
                n
                for n, d in graph.nodes(data=True)
                if d.get("type") not in tree_relation_types | {self.symbol_type_id}
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
                                    "target_index", self._target_index_from_name(name)
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
                    rel_type in {self.sibling_relation, self.cousin_relation}
                    and pos0 is not None
                    and pos1 is not None
                ):
                    # compute canonical perpendicular using min->max target order
                    try:
                        i0 = self._target_index_from_name(pos0)
                        i1 = self._target_index_from_name(pos1)
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

        ax = super().draw(
            graph,
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


__all__ = ["HorizonEncoder", "HorizonEncoderStream"]
