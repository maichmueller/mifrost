from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Iterable, Mapping

from torch_geometric.data import HeteroData

from .. import _neutral_core
from .._core import (
    DEFAULT_HISTORY_LINK_RELATION,
    DEFAULT_LGAN_RR_EDGE_POS,
    DEFAULT_LGAN_TN_EDGE_POS,
    DEFAULT_LGAN_NN_EDGE_POS,
    DEFAULT_SYMBOL_TYPE_ID,
    BatchEncoding,
    HorizonEncoderMode,
)
from ..backends._horizon_runtime import (
    HorizonBackendName,
    create_horizon_runtime,
)
from .base import (
    ActionBatchInput,
    ActionBatchParam,
    CollateSpecParam,
    DagBatchParam,
    GoalBatchInput,
    GoalBatchParam,
    HistorySubgoalsBatchParam,
    StateBatchInput,
    StreamEncoderBase,
    SubgoalLayersInput,
    SubgoalLayersBatchParam,
)
from ._horizon_validation import (
    validate_batch_unsupported_lanes,
    validate_single_unsupported_lanes,
)
from ._root_policy import RootPolicy, normalize_root_policy
from ._visualization import (
    HorizonVisualizationContext,
    draw_horizon,
    horizon_to_networkx,
)
from ._rustworkx_dag import RXStateDAG
from .hgraph import HGraphEncoder, TargetSource
from .types import (
    DomainInput,
    GoalLiteralInput,
    HistorySubgoalInput,
    StateInput,
)

if TYPE_CHECKING:
    import networkx as nx

    from .._core import TransitionDAG
else:
    TransitionDAG = Any


_HORIZON_CONFIG_CLS = _neutral_core.SemanticHorizonHGraphEncoderConfig
_HORIZON_MODE = _neutral_core.SemanticHorizonEncoderMode


def _normalize_horizon_mode(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, str):
        name = value.strip().lower()
    else:
        name = str(value).rsplit(".", 1)[-1].lower()
    if name not in {"full", "delta", "action"}:
        raise ValueError("transition_mode must be 'full', 'delta', or 'action'")
    return getattr(_HORIZON_MODE, name)


@dataclass
class HorizonEncoderStream(StreamEncoderBase[HeteroData]):
    """Streaming wrapper for ``HorizonHGraphEncoderEngine``."""

    _owner: Any

    def __post_init__(self) -> None:
        """Initialize an empty hetero builder for streaming."""
        runtime = getattr(self._owner, "_runtime", None)
        self._legacy_engine = runtime is None
        if self._legacy_engine:
            from .._core import HorizonStreamEncoder as NativeHorizonStreamEncoder

            self._stream = NativeHorizonStreamEncoder(self._owner)
        else:
            self._stream = runtime.make_stream()
        self._reset_builder()

    def append(
        self,
        root: StateInput,
        dag: TransitionDAG | RXStateDAG | None = None,
        *,
        goals: Iterable[GoalLiteralInput] | None = None,
        subgoal_layers: Iterable[Iterable[GoalLiteralInput]] | None = None,
    ) -> int:
        """Append one root/DAG encoding to the stream."""
        if self._legacy_engine:
            from ..backends.pymimir_common import _advanced_state
            from ..backends.pymimir_lane_specs import (
                ensure_transition_dag,
                prepare_goal_inputs,
            )

            return self._coerce_stream_id(
                self._stream.append(
                    _advanced_state(root),
                    ensure_transition_dag(root, dag),
                    prepare_goal_inputs(root, goals, subgoal_layers),
                )
            )
        return self._coerce_stream_id(
            self._stream.append(root, dag, goals=goals, subgoal_layers=subgoal_layers)
        )

    def remove(self, stream_id: int) -> None:
        self._stream.remove(stream_id)

    def update(
        self,
        stream_id: int,
        root: StateInput,
        dag: TransitionDAG | RXStateDAG | None = None,
        *,
        goals: Iterable[GoalLiteralInput] | None = None,
        subgoal_layers: Iterable[Iterable[GoalLiteralInput]] | None = None,
    ) -> None:
        if self._legacy_engine:
            from ..backends.pymimir_common import _advanced_state
            from ..backends.pymimir_lane_specs import (
                ensure_transition_dag,
                prepare_goal_inputs,
            )

            self._stream.update(
                stream_id,
                _advanced_state(root),
                ensure_transition_dag(root, dag),
                prepare_goal_inputs(root, goals, subgoal_layers),
            )
            return
        self._stream.update(
            stream_id, root, dag, goals=goals, subgoal_layers=subgoal_layers
        )

    def _reset_builder(self) -> None:
        """Reset stream accumulation state."""
        self._stream.reset()


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
        backend: HorizonBackendName | str | None = None,
        transition_mode: HorizonEncoderMode | str | None = None,
        target_symbol_prefix: str | None = None,
        target_sources: Iterable[TargetSource | str] | None = None,
        parent_relation: str | None = None,
        sibling_relation: str | None = None,
        cousin_relation: str | None = None,
        enable_parent_relation: bool | None = None,
        enable_sibling_relation: bool | None = None,
        enable_cousin_relation: bool | None = None,
        root_policy: RootPolicy | str | None = None,
        max_goal_level: int | None = None,
        symbol_type_id: str | None = DEFAULT_SYMBOL_TYPE_ID,
        ignore_actions: bool | None = None,
        add_nullary_predicates: bool | None = None,
        include_lgan_edges: bool | None = None,
        include_static: bool | None = None,
        include_empty_edge_types: bool | None = None,
        export_node_names: bool | None = None,
        support_literals: bool | None = None,
        goal_derivations: Iterable[Any] | None = None,
        nullary_object_name: str | None = None,
        lgan_tn_edge_pos: str | None = DEFAULT_LGAN_TN_EDGE_POS,
        lgan_nn_edge_pos: str | None = DEFAULT_LGAN_NN_EDGE_POS,
        lgan_rr_edge_pos: str | None = DEFAULT_LGAN_RR_EDGE_POS,
        history_link_relation: str | None = DEFAULT_HISTORY_LINK_RELATION,
    ) -> None:
        """Create a hetero horizon encoder.

        This lane reads a root state plus a `TransitionDAG` and creates
        candidate state rows.

        `target_sources` is therefore simple on this lane:

        - `state`: successor or candidate states from the DAG

        `root_policy` controls how the root state is treated:

        - `include`: encode the root and expose it as a target
        - `encode_only`: encode the root, but omit it from targets and metadata
        - `exclude`: omit the root from targets and keep root facts as base facts

        The main-lane sources `action`, `goal`, `subgoal`, and `history` do
        not create separate targets here. When `include_lgan_edges=True`, LGAN
        anchors are those candidate state rows. There is no
        `lgan_anchor_sources` switch here.
        """
        normalized_root_policy = normalize_root_policy(root_policy)
        config = self._make_config(
            _HORIZON_CONFIG_CLS,
            symbol_type_id=symbol_type_id,
            target_symbol_prefix=target_symbol_prefix,
            ignore_actions=ignore_actions,
            add_nullary_predicates=add_nullary_predicates,
            include_lgan_edges=include_lgan_edges,
            include_static=include_static,
            include_empty_edge_types=include_empty_edge_types,
            export_node_names=export_node_names,
            max_goal_level=max_goal_level,
            support_literals=support_literals,
            goal_derivations=goal_derivations,
            nullary_object_name=nullary_object_name,
            lgan_tn_edge_pos=lgan_tn_edge_pos,
            lgan_nn_edge_pos=lgan_nn_edge_pos,
            lgan_rr_edge_pos=lgan_rr_edge_pos,
            history_link_relation=history_link_relation,
            transition_mode=_normalize_horizon_mode(transition_mode),
            target_sources=(
                None
                if target_sources is None
                else self._normalize_horizon_target_sources(target_sources)
            ),
            parent_relation=parent_relation,
            sibling_relation=sibling_relation,
            cousin_relation=cousin_relation,
            enable_parent_relation=enable_parent_relation,
            enable_sibling_relation=enable_sibling_relation,
            enable_cousin_relation=enable_cousin_relation,
            root_policy=normalized_root_policy,
        )
        self._runtime = create_horizon_runtime(domain, config, backend=backend)
        self._engine = self._runtime.engine
        self._config = self._engine.config
        self.backend = self._runtime.backend_name
        self.symbol_type_id = self._config.symbol_type_id
        self.target_symbol_prefix = self._config.target_symbol_prefix
        self.parent_relation = self._config.parent_relation
        self.sibling_relation = self._config.sibling_relation
        self.cousin_relation = self._config.cousin_relation
        self.lgan_tn_edge_pos = self._config.lgan_tn_edge_pos
        self.lgan_nn_edge_pos = self._config.lgan_nn_edge_pos
        self.lgan_rr_edge_pos = self._config.lgan_rr_edge_pos
        self.include_lgan_edges = self._config.include_lgan_edges
        self.lgan_anchor_sources = set(self._config.lgan_anchor_sources)
        self._lgan_edge_positions = {
            self.lgan_tn_edge_pos,
            self.lgan_nn_edge_pos,
            self.lgan_rr_edge_pos,
        }

    @staticmethod
    def _normalize_horizon_target_sources(
        target_sources: Iterable[TargetSource | str],
    ) -> set[TargetSource]:
        from ._target_sources import normalize_target_sources

        normalized = normalize_target_sources(target_sources)
        assert normalized is not None
        return normalized

    @property
    def engine(self) -> Any:
        """Expose the underlying C++ horizon engine."""
        return self._engine

    def _encode(
        self,
        root: StateInput,
        dag: TransitionDAG | RXStateDAG | None = None,
        *,
        goals: GoalBatchInput = None,
        actions: ActionBatchInput = None,
        subgoal_layers: SubgoalLayersInput = None,
        history_subgoals: HistorySubgoalInput | None = None,
        history_max_steps: int | None = None,
        **_,
    ) -> BatchEncoding:
        """Encode one root/DAG pair."""
        validate_single_unsupported_lanes(
            actions=actions,
            history_subgoals=history_subgoals,
            history_max_steps=history_max_steps,
        )
        return self._runtime.encode_one(
            root, dag, goals=goals, subgoal_layers=subgoal_layers
        )

    def _encode_one_into_builder(
        self,
        root: StateInput,
        builder: Any,
        *,
        dag: TransitionDAG | RXStateDAG | None = None,
        goals: GoalBatchInput = None,
        actions: ActionBatchInput = None,
        subgoal_layers: SubgoalLayersInput = None,
        history_subgoals: HistorySubgoalInput | None = None,
        history_max_steps: int | None = None,
    ) -> None:
        """Append one validated root/DAG graph into a caller-owned builder."""
        validate_single_unsupported_lanes(
            actions=actions,
            history_subgoals=history_subgoals,
            history_max_steps=history_max_steps,
        )
        self._runtime.append_into_builder(
            root,
            builder,
            dag=dag,
            goals=goals,
            subgoal_layers=subgoal_layers,
        )

    def encode(
        self,
        root: StateInput,
        dag: TransitionDAG | RXStateDAG | None = None,
        *,
        goals: GoalBatchInput = None,
        actions: ActionBatchInput = None,
        subgoal_layers: SubgoalLayersInput = None,
        history_subgoals: HistorySubgoalInput | None = None,
        history_max_steps: int | None = None,
        include_metadata: bool = True,
        **kwargs,
    ) -> BatchEncoding:
        """Encode one root/DAG pair into native ``BatchEncoding``."""
        return super().encode(
            root,
            goals=goals,
            actions=actions,
            subgoal_layers=subgoal_layers,
            history_subgoals=history_subgoals,
            history_max_steps=history_max_steps,
            dag=dag,
            include_metadata=include_metadata,
            **kwargs,
        )

    def encode_batch(
        self,
        roots: StateBatchInput,
        dags: DagBatchParam = None,
        *,
        goals: GoalBatchParam = None,
        actions: ActionBatchParam = None,
        subgoal_layers: SubgoalLayersBatchParam = None,
        history_subgoals: HistorySubgoalsBatchParam = None,
        history_max_steps: int | None = None,
        batch_attrs: Mapping[str, Any] | None = None,
        collate_spec: CollateSpecParam = None,
        include_metadata: bool = True,
        **kwargs,
    ) -> BatchEncoding:
        """Encode one or many root/DAG pairs into native ``BatchEncoding``."""
        return super().encode_batch(
            roots,
            goals=goals,
            actions=actions,
            subgoal_layers=subgoal_layers,
            history_subgoals=history_subgoals,
            history_max_steps=history_max_steps,
            batch_attrs=batch_attrs,
            collate_spec=collate_spec,
            dags=dags,
            include_metadata=include_metadata,
            **kwargs,
        )

    def _accepted_kwargs(self) -> set[str]:
        """Accept transition DAG kwargs in the generic base API."""
        return super()._accepted_kwargs() | {"dag", "dags"}

    def _encode_batch(
        self,
        roots: StateBatchInput,
        dags: DagBatchParam = None,
        *,
        goals: GoalBatchParam = None,
        subgoal_layers: SubgoalLayersBatchParam = None,
        actions: ActionBatchParam = None,
        history_subgoals: HistorySubgoalsBatchParam = None,
        history_max_steps: int | None = None,
    ) -> BatchEncoding:
        """Encode one or many root/DAG pairs into one batch encoding."""
        validate_batch_unsupported_lanes(
            actions=actions,
            history_subgoals=history_subgoals,
            history_max_steps=history_max_steps,
        )
        return self._runtime.encode_batch(
            roots,
            dags=dags,
            goals=goals,
            subgoal_layers=subgoal_layers,
        )

    def stream(self) -> HorizonEncoderStream:
        """Create a streaming encoder sharing this encoder's C++ engine."""
        return HorizonEncoderStream(self)

    def _horizon_visualization_context(self) -> HorizonVisualizationContext:
        return HorizonVisualizationContext(
            hgraph=self._visualization_context(),
            parent_relation=self.parent_relation,
            sibling_relation=self.sibling_relation,
            cousin_relation=self.cousin_relation,
            target_symbol_prefix=self.target_symbol_prefix,
        )

    def to_networkx(self, data: HeteroData) -> nx.MultiGraph:
        """Convert encoded horizon ``HeteroData`` into a named multigraph."""
        return horizon_to_networkx(data, self._horizon_visualization_context())

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
        if hasattr(graph, "edge_types"):
            graph = self.to_networkx(graph)
        return draw_horizon(
            graph,
            context=self._horizon_visualization_context(),
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
            align_target_nodes=align_target_nodes,
            target_x_spacing=target_x_spacing,
            target_y_spacing=target_y_spacing,
            layout_seed=layout_seed,
            symbol_node_scale=symbol_node_scale,
            non_symbol_linestyle=non_symbol_linestyle,
        )


__all__ = ["HorizonEncoder", "HorizonEncoderStream"]
