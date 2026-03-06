from __future__ import annotations

import numbers
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import networkx as nx
from torch_geometric.data import HeteroData
from torch_geometric.utils import to_networkx

from .._core import BatchEncoding
from .. import _core
from .._core import (
    BatchBuilder,
    DEFAULT_HISTORY_LINK_RELATION,
    DEFAULT_LGAN_RR_EDGE_POS,
    DEFAULT_LGAN_TN_EDGE_POS,
    DEFAULT_LGAN_NN_EDGE_POS,
    DEFAULT_SYMBOL_TYPE_ID,
    HGraphEncoderConfig,
    HGraphEncoderEngine,
    HGraphStreamEncoder as _HGraphStreamEncoder,
)
from .base import (
    ActionBatchInput,
    ActionBatchParam,
    CollateSpecParam,
    EncoderBase,
    GoalBatchInput,
    GoalBatchParam,
    HistorySubgoalsBatchParam,
    StateBatchInput,
    StreamEncoderBase,
    SubgoalLayersInput,
    SubgoalLayersBatchParam,
)
from .common import (
    _advanced_domain,
    _advanced_state,
    _convert_batch_payload,
    _prepare_actions,
    _prepare_history_subgoals,
    _split_goals,
)
from ._action_contract import (
    parse_flat_actions,
)
from ._target_sources import TargetSource, normalize_target_sources
from .types import (
    DomainInput,
    GoalLiteralInput,
    GroundActionInput,
    HeteroEncoding,
    HistorySubgoalInput,
    StateInput,
    default_goals_from_state,
    is_action_input,
    is_goal_literal_input,
    is_state_input,
    to_advanced_action,
    to_advanced_literal,
    to_advanced_state,
)

_HGraphMutableStreamEncoder = getattr(
    _core, "HGraphMutableStreamEncoder", _HGraphStreamEncoder
)


def _build_config(config_cls, **kwargs: object):
    """Build a nanobind config object from non-None keyword values."""
    clean_kwargs = {key: value for key, value in kwargs.items() if value is not None}
    return config_cls(**clean_kwargs)


def _draw_hgraph_graph(
    graph: nx.Graph,
    *,
    symbol_type_id: str,
    lgan_tn_edge_pos: str,
    lgan_nn_edge_pos: str,
    lgan_rr_edge_pos: str,
    include_lgan_edges: bool,
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
        node for node in graph.nodes if graph.nodes[node]["type"] == symbol_type_id
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
        symbol_colors = [type_to_color[symbol_type_id] for _ in symbol_nodes]
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

    lgan_edge_positions = {lgan_tn_edge_pos, lgan_nn_edge_pos, lgan_rr_edge_pos}

    edge_attr_name = "position"
    standard_edges = []
    standard_colors = []
    lgan_edges = []
    lgan_colors = []

    if graph.is_multigraph():
        all_edge_iter = graph.edges(keys=True, data=True)
    else:
        all_edge_iter = graph.edges(data=True)

    all_positions = []
    for edge_info in all_edge_iter:
        data_dict = edge_info[-1]
        pos_value = data_dict.get(edge_attr_name)
        if pos_value is not None:
            all_positions.append(pos_value)

    unique_positions = list(dict.fromkeys(p for p in all_positions))
    edge_pos_to_color = {}
    if unique_positions:
        cmap = plt.get_cmap("Dark2")
        edge_pos_to_color = {
            value: cmap(i / max(1, len(unique_positions) - 1))
            for i, value in enumerate(unique_positions)
        }

    if graph.is_multigraph():
        all_edge_iter = graph.edges(keys=True, data=True)
    else:
        all_edge_iter = graph.edges(data=True)

    for edge_info in all_edge_iter:
        src, dst = edge_info[0], edge_info[1]
        data_dict = edge_info[-1]
        pos_value = data_dict.get(edge_attr_name)
        color = edge_pos_to_color.get(pos_value, "#666666")

        if pos_value in lgan_edge_positions:
            lgan_edges.append((src, dst))
            lgan_colors.append(color)
        else:
            standard_edges.append((src, dst))
            standard_colors.append(color)

    if edge_width is not None:
        edge_kwargs.setdefault("width", edge_width)
    if edge_alpha is not None:
        edge_kwargs.setdefault("alpha", edge_alpha)

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

    if lgan_edges:
        lgan_kwargs = dict(edge_kwargs)
        lgan_kwargs["style"] = "dashed"
        nx.draw_networkx_edges(
            graph,
            pos,
            edgelist=lgan_edges,
            edge_color=lgan_colors,
            arrows=graph.is_directed(),
            ax=ax,
            **lgan_kwargs,
        )

    if unique_types:
        from matplotlib.patches import Patch

        legend_handles = [
            Patch(facecolor=type_to_color[ntype], edgecolor="none", label=str(ntype))
            for ntype in unique_types
        ]
        node_title = "Node Types"
        if include_lgan_edges:
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
                color=edge_pos_to_color[pos_value],
                linestyle="dashed" if pos_value in lgan_edge_positions else "solid",
                label=f"LGAN ({pos_value})"
                if pos_value in lgan_edge_positions
                else f"pos: {pos_value}",
            )
            for pos_value in unique_positions
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
        if graph.is_multigraph():
            labels = {}
            for src, dst, key, data in graph.edges(keys=True, data=True):
                if (
                    label_edge_set is not None
                    and (src, dst, key) not in label_edge_set
                    and (src, dst) not in label_edge_set
                ):
                    continue
                labels[(src, dst, key)] = data.get(edge_attr_name)
        else:
            labels = {}
            for src, dst, data in graph.edges(data=True):
                if label_edge_set is not None and (src, dst) not in label_edge_set:
                    continue
                labels[(src, dst)] = data.get(edge_attr_name)
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


@dataclass
class HGraphMutableEncoderStream(StreamEncoderBase[HeteroData]):
    """Mutable streaming wrapper (append/update/remove) for ``HGraphEncoderEngine``."""

    _engine: HGraphEncoderEngine

    def __post_init__(self) -> None:
        self._stream = _HGraphMutableStreamEncoder(self._engine)
        self._reset_builder()

    def append(
        self,
        state: StateInput,
        *,
        goals: Iterable[GoalLiteralInput] | None = None,
        actions: Iterable[GroundActionInput] | None = None,
        subgoal_layers: Iterable[Iterable[GoalLiteralInput]] | None = None,
        history_subgoals: HistorySubgoalInput | None = None,
        history_max_steps: int | None = None,
    ) -> int:
        """
        Append one state encoding to the current stream.

        If goals/actions are omitted, the engine uses the state's problem goals.
        """
        adv_state = _advanced_state(state)
        action_inputs = parse_flat_actions(actions)
        action_list = _prepare_actions(action_inputs)
        history_list = _prepare_history_subgoals(history_subgoals)
        if (
            goals is None
            and subgoal_layers is None
            and not action_list
            and not history_list
        ):
            # Fast path: let the engine derive goals from the state/problem.
            return self._coerce_stream_id(self._stream.append(adv_state))
        else:
            if goals is None:
                goals = default_goals_from_state(state)
            inputs = _split_goals(goals, subgoal_layers)
            if history_list:
                return self._coerce_stream_id(
                    self._stream.append(
                        adv_state,
                        inputs,
                        action_list,
                        history_list,
                        history_max_steps,
                    )
                )
            return self._coerce_stream_id(
                self._stream.append(adv_state, inputs, action_list)
            )

    def remove(self, stream_id: int) -> None:
        self._stream.remove(stream_id)

    def update(
        self,
        stream_id: int,
        state: StateInput,
        *,
        goals: Iterable[GoalLiteralInput] | None = None,
        actions: Iterable[GroundActionInput] | None = None,
        subgoal_layers: Iterable[Iterable[GoalLiteralInput]] | None = None,
        history_subgoals: HistorySubgoalInput | None = None,
        history_max_steps: int | None = None,
    ) -> None:
        adv_state = _advanced_state(state)
        action_inputs = parse_flat_actions(actions)
        action_list = _prepare_actions(action_inputs)
        history_list = _prepare_history_subgoals(history_subgoals)
        if (
            goals is None
            and subgoal_layers is None
            and not action_list
            and not history_list
        ):
            self._stream.update(stream_id, adv_state)
            return
        if goals is None:
            goals = default_goals_from_state(state)
        inputs = _split_goals(goals, subgoal_layers)
        if history_list:
            self._stream.update(
                stream_id,
                adv_state,
                inputs,
                action_list,
                history_list,
                history_max_steps,
            )
        else:
            self._stream.update(
                stream_id,
                adv_state,
                inputs,
                action_list,
            )

    def _reset_builder(self) -> None:
        """Reset stream accumulation state."""
        self._stream.reset()


@dataclass
class HGraphEncoderStream(StreamEncoderBase[HeteroData]):
    """Append-only streaming wrapper for ``HGraphEncoderEngine``."""

    _engine: HGraphEncoderEngine

    def __post_init__(self) -> None:
        self._stream = _HGraphStreamEncoder(self._engine)
        self._reset_builder()

    def append(
        self,
        state: StateInput,
        *,
        goals: Iterable[GoalLiteralInput] | None = None,
        actions: Iterable[GroundActionInput] | None = None,
        subgoal_layers: Iterable[Iterable[GoalLiteralInput]] | None = None,
        history_subgoals: HistorySubgoalInput | None = None,
        history_max_steps: int | None = None,
    ) -> int:
        adv_state = _advanced_state(state)
        action_inputs = parse_flat_actions(actions)
        action_list = _prepare_actions(action_inputs)
        history_list = _prepare_history_subgoals(history_subgoals)
        if (
            goals is None
            and subgoal_layers is None
            and not action_list
            and not history_list
        ):
            return self._coerce_stream_id(self._stream.append(adv_state))
        if goals is None:
            goals = default_goals_from_state(state)
        inputs = _split_goals(goals, subgoal_layers)
        if history_list:
            return self._coerce_stream_id(
                self._stream.append(
                    adv_state,
                    inputs,
                    action_list,
                    history_list,
                    history_max_steps,
                )
            )
        return self._coerce_stream_id(
            self._stream.append(adv_state, inputs, action_list)
        )

    def _reset_builder(self) -> None:
        self._stream.reset()


class HGraphEncoder(EncoderBase[HeteroData]):
    """
    General heterogeneous graph encoder backed by ``HGraphEncoderEngine``.

    Use this encoder when you need state-based atom/object/action graphs in
    hetero PyG format.
    """

    @staticmethod
    def _make_config(config_cls, **kwargs: object):
        """Create a config object with optional-field filtering."""
        return _build_config(config_cls, **kwargs)

    def _init_engine_from_config(
        self,
        domain: DomainInput,
        config: object,
        *,
        engine_cls=HGraphEncoderEngine,
    ) -> None:
        """Initialize encoder runtime state from a prepared config object."""
        self._engine = engine_cls(_advanced_domain(domain), config)
        self._config = config
        self.symbol_type_id = config.symbol_type_id
        self.lgan_tn_edge_pos = getattr(
            config, "lgan_tn_edge_pos", DEFAULT_LGAN_TN_EDGE_POS
        )
        self.lgan_nn_edge_pos = getattr(
            config, "lgan_nn_edge_pos", DEFAULT_LGAN_NN_EDGE_POS
        )
        self.lgan_rr_edge_pos = getattr(
            config, "lgan_rr_edge_pos", DEFAULT_LGAN_RR_EDGE_POS
        )
        self.include_lgan_edges = getattr(config, "include_lgan_edges", False)
        self._lgan_edge_positions = {
            self.lgan_tn_edge_pos,
            self.lgan_nn_edge_pos,
            self.lgan_rr_edge_pos,
        }

    def __init__(
        self,
        domain: DomainInput,
        *,
        symbol_type_id: str = DEFAULT_SYMBOL_TYPE_ID,
        target_symbol_prefix: str = "target:",
        ignore_actions: bool = True,
        add_nullary_predicates: bool = False,
        include_lgan_edges: bool = False,
        include_static: bool = True,
        include_empty_edge_types: bool = True,
        export_node_names: bool = True,
        target_sources: Iterable[TargetSource | str] | None = None,
        max_goal_level: int = 0,
        support_literals: bool = False,
        nullary_object_name: str = "![nullary_symbol]!",
        lgan_tn_edge_pos: str = DEFAULT_LGAN_TN_EDGE_POS,
        lgan_nn_edge_pos: str = DEFAULT_LGAN_NN_EDGE_POS,
        lgan_rr_edge_pos: str = DEFAULT_LGAN_RR_EDGE_POS,
        history_link_relation: str = DEFAULT_HISTORY_LINK_RELATION,
        _config_cls=HGraphEncoderConfig,
        _engine_cls=HGraphEncoderEngine,
        **extra_config_kwargs: object,
    ) -> None:
        """Create an HGraph encoder for one domain."""
        config = self._make_config(
            _config_cls,
            symbol_type_id=symbol_type_id,
            target_symbol_prefix=target_symbol_prefix,
            ignore_actions=ignore_actions,
            add_nullary_predicates=add_nullary_predicates,
            include_lgan_edges=include_lgan_edges,
            include_static=include_static,
            include_empty_edge_types=include_empty_edge_types,
            export_node_names=export_node_names,
            target_sources=normalize_target_sources(target_sources),
            max_goal_level=max_goal_level,
            support_literals=support_literals,
            nullary_object_name=nullary_object_name,
            lgan_tn_edge_pos=lgan_tn_edge_pos,
            lgan_nn_edge_pos=lgan_nn_edge_pos,
            lgan_rr_edge_pos=lgan_rr_edge_pos,
            history_link_relation=history_link_relation,
            **extra_config_kwargs,
        )
        self._init_engine_from_config(domain, config, engine_cls=_engine_cls)

    def _encode_one_into_builder(
        self,
        state: StateInput,
        builder: BatchBuilder,
        *,
        goals: GoalBatchInput = None,
        actions: Iterable[GroundActionInput] | None = None,
        subgoal_layers: SubgoalLayersInput = None,
        history_subgoals: HistorySubgoalInput | None = None,
        history_max_steps: int | None = None,
    ) -> None:
        adv_state = _advanced_state(state)
        action_inputs = parse_flat_actions(actions)
        action_list = _prepare_actions(action_inputs)
        history_list = _prepare_history_subgoals(history_subgoals)
        if (
            goals is None
            and subgoal_layers is None
            and not action_list
            and not history_list
        ):
            self._engine.encode(adv_state, builder)
            return

        goals_input = goals if goals is not None else default_goals_from_state(state)
        inputs = _split_goals(goals_input, subgoal_layers)
        if history_list:
            self._engine.encode(
                adv_state, inputs, action_list, history_list, history_max_steps, builder
            )
            return
        self._engine.encode(adv_state, inputs, action_list, builder)

    @property
    def engine(self) -> HGraphEncoderEngine:
        """Expose the underlying C++ engine for advanced usage."""
        return self._engine

    @property
    def config(self):
        """Expose the effective encoder config object."""
        return self._config

    @property
    def relation_dict(self) -> Any:
        """Expose the effective built relation dictionary from the C++ engine."""
        return self._engine.relation_dict

    def update_relations(self, relation_dict: Any) -> None:
        """Replace relation dictionary used by the underlying C++ engine."""
        if isinstance(relation_dict, _core.RelationDict):
            core_relation_dict = relation_dict
        elif isinstance(relation_dict, Mapping):
            core_relation_dict = _core.RelationDict(dict(relation_dict))
        else:
            raise TypeError(
                "update_relations expects mifrost.RelationDict or a mapping[str, int]"
            )
        self._engine.update_relations(core_relation_dict)

    def _accepted_kwargs(self) -> set[str]:
        return {"history_subgoals", "history_max_steps"}

    def _encode(
        self,
        state: StateInput,
        *,
        goals: GoalBatchInput = None,
        actions: Iterable[GroundActionInput] | None = None,
        subgoal_layers: SubgoalLayersInput = None,
        history_subgoals: HistorySubgoalInput | None = None,
        history_max_steps: int | None = None,
    ) -> BatchEncoding:
        """Encode one state to normalized batch encoding."""
        builder = BatchBuilder()
        builder.set_graph_kind("hetero")
        self._encode_one_into_builder(
            state,
            builder,
            goals=goals,
            actions=actions,
            subgoal_layers=subgoal_layers,
            history_subgoals=history_subgoals,
            history_max_steps=history_max_steps,
        )
        builder.next_graph()
        return builder.build()

    def encode(
        self,
        state: StateInput,
        *,
        goals: GoalBatchInput = None,
        actions: Iterable[GroundActionInput] | None = None,
        subgoal_layers: SubgoalLayersInput = None,
        history_subgoals: HistorySubgoalInput | None = None,
        history_max_steps: int | None = None,
        include_metadata: bool = True,
        **kwargs: object,
    ) -> BatchEncoding:
        """Encode one state into native ``BatchEncoding``."""
        return super().encode(
            state,
            goals=goals,
            actions=actions,
            subgoal_layers=subgoal_layers,
            history_subgoals=history_subgoals,
            history_max_steps=history_max_steps,
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
        history_subgoals: HistorySubgoalsBatchParam = None,
        history_max_steps: int | None = None,
    ) -> BatchEncoding:
        """Encode one or many states to one native batch encoding."""
        states_for_core = _convert_batch_payload(
            states,
            is_leaf=is_state_input,
            convert_leaf=to_advanced_state,
        )
        goals_for_core = _convert_batch_payload(
            goals,
            is_leaf=is_goal_literal_input,
            convert_leaf=to_advanced_literal,
        )
        actions_for_core = _convert_batch_payload(
            actions,
            is_leaf=is_action_input,
            convert_leaf=to_advanced_action,
        )
        subgoal_layers_for_core = _convert_batch_payload(
            subgoal_layers,
            is_leaf=is_goal_literal_input,
            convert_leaf=to_advanced_literal,
        )
        history_subgoals_for_core = _convert_batch_payload(
            history_subgoals,
            is_leaf=is_goal_literal_input,
            convert_leaf=to_advanced_literal,
        )
        return self._engine.encode_batch(
            states_for_core,
            goals=goals_for_core,
            actions=actions_for_core,
            subgoal_layers=subgoal_layers_for_core,
            history_subgoals=history_subgoals_for_core,
            history_max_steps=history_max_steps,
        )

    def encode_batch(
        self,
        states: StateBatchInput,
        *,
        goals: GoalBatchParam = None,
        actions: ActionBatchParam = None,
        subgoal_layers: SubgoalLayersBatchParam = None,
        history_subgoals: HistorySubgoalsBatchParam = None,
        history_max_steps: int | None = None,
        batch_attrs: Mapping[str, Any] | None = None,
        collate_spec: CollateSpecParam = None,
        include_metadata: bool = True,
        **kwargs: object,
    ) -> BatchEncoding:
        """Encode one or many states into native ``BatchEncoding``."""
        return super().encode_batch(
            states,
            goals=goals,
            actions=actions,
            subgoal_layers=subgoal_layers,
            history_subgoals=history_subgoals,
            history_max_steps=history_max_steps,
            batch_attrs=batch_attrs,
            collate_spec=collate_spec,
            include_metadata=include_metadata,
            **kwargs,
        )

    def stream(self) -> HGraphEncoderStream:
        """Create an append-only streaming encoder sharing this encoder's C++ engine."""
        return HGraphEncoderStream(self._engine)

    def mutable_stream(self) -> HGraphMutableEncoderStream:
        """Create a mutable streaming encoder supporting update/remove."""
        return HGraphMutableEncoderStream(self._engine)

    def to_networkx(self, data: HeteroData) -> nx.MultiDiGraph:
        """Convert ``HeteroData`` to named NetworkX graph for plotting."""
        graph = to_networkx(
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
            if suffix > 0:
                display_name = f"{name}#{suffix}"
            else:
                display_name = name
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

    def draw(
        self,
        data: HeteroData,
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

        graph = data if isinstance(data, nx.Graph) else self.to_networkx(data)
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
            base_node_size_value = (
                base_node_size_value[0] if base_node_size_value else None
            )
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
            if graph.nodes[node]["type"] == self.symbol_type_id
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
            symbol_colors = [type_to_color[self.symbol_type_id] for _ in symbol_nodes]
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

            if p_val in self._lgan_edge_positions:
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
                Patch(
                    facecolor=type_to_color[ntype], edgecolor="none", label=str(ntype)
                )
                for ntype in unique_types
            ]
            node_title = "Node Types"
            if self.include_lgan_edges:
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
                    linestyle="dashed"
                    if p_val in self._lgan_edge_positions
                    else "solid",
                    label=f"LGAN ({p_val})"
                    if p_val in self._lgan_edge_positions
                    else f"pos: {p_val}",
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
            if graph.is_multigraph():
                labels = {}
                for u, v, k, data in graph.edges(keys=True, data=True):
                    if (
                        label_edge_set is not None
                        and (u, v, k) not in label_edge_set
                        and (u, v) not in label_edge_set
                    ):
                        continue
                    labels[(u, v, k)] = data.get(edge_attr_name)
            else:
                labels = {}
                for u, v, data in graph.edges(data=True):
                    if label_edge_set is not None and (u, v) not in label_edge_set:
                        continue
                    labels[(u, v)] = data.get(edge_attr_name)
            labels = {
                edge: value for edge, value in labels.items() if value is not None
            }
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


__all__ = ["HGraphEncoder", "HGraphEncoderStream", "HGraphMutableEncoderStream"]
