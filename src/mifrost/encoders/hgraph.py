from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import networkx as nx
from torch_geometric.data import HeteroData

from .. import _core, _neutral_core
from .._core import (
    BatchEncoding,
    BatchBuilder,
    DEFAULT_HISTORY_LINK_RELATION,
    DEFAULT_LGAN_RR_EDGE_POS,
    DEFAULT_LGAN_TN_EDGE_POS,
    DEFAULT_LGAN_NN_EDGE_POS,
    DEFAULT_SYMBOL_TYPE_ID,
)
from ..backends._hgraph_runtime import HGraphBackendName, create_hgraph_runtime
from .base import (
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
from ._target_sources import TargetSource, normalize_target_sources
from ._visualization import (
    HGraphVisualizationContext,
    draw_hgraph,
    hgraph_to_networkx,
)
from .types import (
    DomainInput,
    GoalLiteralInput,
    GroundActionInput,
    HistorySubgoalInput,
    StateInput,
    default_goals_from_state,
)

_BASE_HGRAPH_CONFIG_CLS = _neutral_core.SemanticHGraphEncoderConfig
_BASE_HGRAPH_ENGINE_CLS = None


def _build_config(config_cls, **kwargs: Any):
    """Build a nanobind config object from non-None keyword values."""
    clean_kwargs = {key: value for key, value in kwargs.items() if value is not None}
    return config_cls(**clean_kwargs)


@dataclass
class HGraphMutableEncoderStream(StreamEncoderBase[HeteroData]):
    """Mutable streaming wrapper (append/update/remove) for ``HGraphEncoderEngine``."""

    _owner: Any

    def __post_init__(self) -> None:
        self._stream: Any
        runtime = getattr(self._owner, "_runtime", None)
        if runtime is None:
            stream_type = getattr(
                _core,
                "HGraphMutableStreamEncoder",
                getattr(_core, "HGraphStreamEncoder"),
            )
            self._stream = stream_type(self._owner)
        else:
            self._stream = runtime.make_stream(mutable=True)
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
        if getattr(self._owner, "_runtime", None) is not None:
            return self._coerce_stream_id(
                self._stream.append(
                    state,
                    goals=goals,
                    actions=actions,
                    subgoal_layers=subgoal_layers,
                    history_subgoals=history_subgoals,
                    history_max_steps=history_max_steps,
                )
            )
        from ..backends.pymimir_common import _advanced_state, _split_goals
        from ..backends.pymimir_lane_specs import prepare_optional_payloads

        adv_state = _advanced_state(state)
        payloads = prepare_optional_payloads(
            actions=actions,
            history_subgoals=history_subgoals,
        )
        action_list = payloads.actions
        history_list = payloads.history_subgoals
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
        if getattr(self._owner, "_runtime", None) is not None:
            self._stream.update(
                stream_id,
                state,
                goals=goals,
                actions=actions,
                subgoal_layers=subgoal_layers,
                history_subgoals=history_subgoals,
                history_max_steps=history_max_steps,
            )
            return
        from ..backends.pymimir_common import _advanced_state, _split_goals
        from ..backends.pymimir_lane_specs import prepare_optional_payloads

        adv_state = _advanced_state(state)
        payloads = prepare_optional_payloads(
            actions=actions,
            history_subgoals=history_subgoals,
        )
        action_list = payloads.actions
        history_list = payloads.history_subgoals
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

    _owner: Any

    def __post_init__(self) -> None:
        self._stream: Any
        runtime = getattr(self._owner, "_runtime", None)
        if runtime is None:
            self._stream = _core.HGraphStreamEncoder(self._owner)
        else:
            self._stream = runtime.make_stream(mutable=False)
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
        if getattr(self._owner, "_runtime", None) is not None:
            return self._coerce_stream_id(
                self._stream.append(
                    state,
                    goals=goals,
                    actions=actions,
                    subgoal_layers=subgoal_layers,
                    history_subgoals=history_subgoals,
                    history_max_steps=history_max_steps,
                )
            )
        from ..backends.pymimir_common import _advanced_state, _split_goals
        from ..backends.pymimir_lane_specs import prepare_optional_payloads

        adv_state = _advanced_state(state)
        payloads = prepare_optional_payloads(
            actions=actions,
            history_subgoals=history_subgoals,
        )
        action_list = payloads.actions
        history_list = payloads.history_subgoals
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
    def _make_config(config_cls, **kwargs: Any):
        """Create a config object with optional-field filtering."""
        return _build_config(config_cls, **kwargs)

    def _init_engine_from_config(
        self,
        domain: DomainInput,
        config: Any,
        *,
        engine_cls: Any,
    ) -> None:
        """Initialize encoder runtime state from a prepared config object."""
        from ..backends.pymimir_common import _advanced_domain

        self._runtime = None
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
        self.lgan_anchor_sources = set(getattr(config, "lgan_anchor_sources", set()))
        self._lgan_edge_positions = {
            self.lgan_tn_edge_pos,
            self.lgan_nn_edge_pos,
            self.lgan_rr_edge_pos,
        }

    def __init__(
        self,
        domain: DomainInput,
        *,
        backend: HGraphBackendName | str | None = None,
        symbol_type_id: str = DEFAULT_SYMBOL_TYPE_ID,
        target_symbol_prefix: str = "target:",
        ignore_actions: bool = True,
        add_nullary_predicates: bool = False,
        include_lgan_edges: bool = False,
        lgan_anchor_sources: Iterable[TargetSource | str] | None = None,
        include_static: bool = True,
        include_empty_edge_types: bool = True,
        export_node_names: bool = True,
        target_sources: Iterable[TargetSource | str] | None = None,
        max_goal_level: int = 0,
        support_literals: bool = False,
        goal_derivations: Iterable[Any] | None = None,
        nullary_object_name: str = "![nullary_symbol]!",
        lgan_tn_edge_pos: str = DEFAULT_LGAN_TN_EDGE_POS,
        lgan_nn_edge_pos: str = DEFAULT_LGAN_NN_EDGE_POS,
        lgan_rr_edge_pos: str = DEFAULT_LGAN_RR_EDGE_POS,
        history_link_relation: str = DEFAULT_HISTORY_LINK_RELATION,
        _config_cls=_BASE_HGRAPH_CONFIG_CLS,
        _engine_cls=_BASE_HGRAPH_ENGINE_CLS,
        **extra_config_kwargs,
    ) -> None:
        """Create an HGraph encoder for one domain.

        `target_sources` answers "what should count as a selectable target?"
        on the main hetero state lane:

        - `action`: explicit grounded actions from `actions=...`
        - `goal`: literals from the root `goals=...` input
        - `subgoal`: literals from `subgoal_layers=...`
        - `history`: literals from `history_subgoals=...`
        - `state`: not used here; state targets belong to `HorizonEncoder`

        When `include_lgan_edges=True`, `lgan_anchor_sources` can additionally
        create LGAN-only anchor symbols for `goal`, `subgoal`, and `history`
        without turning them into prediction targets.
        """
        self._runtime: Any = None
        normalized_lgan_anchor_sources = normalize_target_sources(lgan_anchor_sources)
        if (
            normalized_lgan_anchor_sources is not None
            and TargetSource.states in normalized_lgan_anchor_sources
        ):
            raise ValueError(
                "HGraphEncoder currently supports lgan_anchor_sources="
                "{'action', 'goal', 'subgoal', 'history'} only; 'state' "
                "belongs to HorizonEncoder candidate targets"
            )
        uses_public_base_runtime = (
            _config_cls is _BASE_HGRAPH_CONFIG_CLS
            and _engine_cls is _BASE_HGRAPH_ENGINE_CLS
        )
        if not uses_public_base_runtime and backend is not None:
            raise ValueError(
                "backend selection is supported only by the base HGraphEncoder; "
                "private custom-engine compatibility constructors do not select "
                "a backend; use the public Horizon or Transition encoder"
            )
        config = self._make_config(
            _config_cls,
            symbol_type_id=symbol_type_id,
            target_symbol_prefix=target_symbol_prefix,
            ignore_actions=ignore_actions,
            add_nullary_predicates=add_nullary_predicates,
            include_lgan_edges=include_lgan_edges,
            lgan_anchor_sources=normalized_lgan_anchor_sources,
            include_static=include_static,
            include_empty_edge_types=include_empty_edge_types,
            export_node_names=export_node_names,
            target_sources=normalize_target_sources(target_sources),
            max_goal_level=max_goal_level,
            support_literals=support_literals,
            goal_derivations=goal_derivations,
            nullary_object_name=nullary_object_name,
            lgan_tn_edge_pos=lgan_tn_edge_pos,
            lgan_nn_edge_pos=lgan_nn_edge_pos,
            lgan_rr_edge_pos=lgan_rr_edge_pos,
            history_link_relation=history_link_relation,
            **extra_config_kwargs,
        )
        if uses_public_base_runtime:
            self._runtime = create_hgraph_runtime(domain, config, backend=backend)
            self._engine = self._runtime.engine
            self._config = config
            self.backend = self._runtime.backend_name
            self.symbol_type_id = config.symbol_type_id
            self.lgan_tn_edge_pos = config.lgan_tn_edge_pos
            self.lgan_nn_edge_pos = config.lgan_nn_edge_pos
            self.lgan_rr_edge_pos = config.lgan_rr_edge_pos
            self.include_lgan_edges = config.include_lgan_edges
            self.lgan_anchor_sources = set(config.lgan_anchor_sources)
            self._lgan_edge_positions = {
                self.lgan_tn_edge_pos,
                self.lgan_nn_edge_pos,
                self.lgan_rr_edge_pos,
            }
        else:
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
        if self._runtime is not None:
            self._runtime.append_into_builder(
                state,
                builder,
                goals=goals,
                actions=actions,
                subgoal_layers=subgoal_layers,
                history_subgoals=history_subgoals,
                history_max_steps=history_max_steps,
            )
            return
        from ..backends.pymimir_common import _advanced_state, _split_goals
        from ..backends.pymimir_lane_specs import prepare_optional_payloads

        adv_state = _advanced_state(state)
        payloads = prepare_optional_payloads(
            actions=actions,
            history_subgoals=history_subgoals,
        )
        action_list = payloads.actions
        history_list = payloads.history_subgoals
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
            if history_max_steps is None:
                self._engine.encode(
                    adv_state, inputs, action_list, history_list, builder
                )
                return
            self._engine.encode(
                adv_state,
                inputs,
                action_list,
                history_list,
                history_max_steps,
                builder,
            )
            return
        self._engine.encode(adv_state, inputs, action_list, builder)

    @property
    def engine(self) -> Any:
        """Expose the underlying C++ engine for advanced usage."""
        return self._engine

    @property
    def config(self):
        """Expose the effective encoder config object."""
        return self._config

    @property
    def relation_dict(self) -> Any:
        """Expose the effective built relation dictionary from the C++ engine."""
        if self._runtime is not None:
            return self._runtime.relation_dict
        return self._engine.relation_dict

    def update_relations(self, relation_dict: Any) -> None:
        """Replace relation dictionary used by the underlying C++ engine."""
        if self._runtime is not None:
            self._runtime.update_relations(relation_dict)
            return
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
        return super()._accepted_kwargs() | {"history_subgoals", "history_max_steps"}

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
        **kwargs,
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
        if self._runtime is not None:
            return self._runtime.encode_batch(
                states,
                goals=goals,
                actions=actions,
                subgoal_layers=subgoal_layers,
                history_subgoals=history_subgoals,
                history_max_steps=history_max_steps,
            )
        from ._batch_contract import prepare_core_batch_inputs

        inputs = prepare_core_batch_inputs(
            states,
            goals=goals,
            actions=actions,
            subgoal_layers=subgoal_layers,
            history_subgoals=history_subgoals,
        )
        return self._engine.encode_batch(
            inputs.states,
            goals=inputs.goals,
            actions=inputs.actions,
            subgoal_layers=inputs.subgoal_layers,
            history_subgoals=inputs.history_subgoals,
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
        **kwargs,
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
        return HGraphEncoderStream(self if self._runtime is not None else self._engine)

    def mutable_stream(self) -> HGraphMutableEncoderStream:
        """Create a mutable streaming encoder supporting update/remove."""
        return HGraphMutableEncoderStream(
            self if self._runtime is not None else self._engine
        )

    def _visualization_context(self) -> HGraphVisualizationContext:
        return HGraphVisualizationContext(
            symbol_type_id=self.symbol_type_id,
            include_lgan_edges=self.include_lgan_edges,
            lgan_edge_positions=frozenset(self._lgan_edge_positions),
        )

    def to_networkx(self, data: HeteroData) -> nx.MultiDiGraph:
        """Convert ``HeteroData`` to named NetworkX graph for plotting."""
        return hgraph_to_networkx(data)

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
        graph = data if isinstance(data, nx.Graph) else self.to_networkx(data)
        return draw_hgraph(
            graph,
            context=self._visualization_context(),
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


__all__ = ["HGraphEncoder", "HGraphEncoderStream", "HGraphMutableEncoderStream"]
