from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .._core import (
    DEFAULT_LGAN_NN_EDGE_POS,
    DEFAULT_LGAN_RR_EDGE_POS,
    DEFAULT_LGAN_TN_EDGE_POS,
    DEFAULT_PARENT_RELATION,
    TransitionDAG,
    FlatHorizonEncoderConfig,
    FlatHorizonEncoderEngine,
    FlatHorizonEncoderMode,
    FlatHorizonStreamEncoder as _FlatHorizonStreamEncoder,
    HorizonEncoderMode,
)
from ._rustworkx_dag import RXStateDAG, _normalize_dag_batch_data
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
from .common import (
    _advanced_domain,
    _advanced_state,
    _convert_batch_payload,
)
from ._lane_specs import (
    FLAT_HORIZON_LANE_SPEC,
    ensure_transition_dag,
    prepare_goal_inputs,
    validate_batch_optional_payloads,
    validate_single_optional_payloads,
)
from .flat import FlatRelationEncoder
from .types import (
    DomainInput,
    FlatEncoding,
    HistorySubgoalInput,
    StateInput,
    is_goal_literal_input,
    is_state_input,
    to_advanced_literal,
    to_advanced_state,
)


def _normalize_flat_horizon_mode(
    mode: FlatHorizonEncoderMode | HorizonEncoderMode | str | None,
) -> FlatHorizonEncoderMode | None:
    if mode is None:
        return None
    if isinstance(mode, FlatHorizonEncoderMode):
        return mode
    if isinstance(mode, str):
        normalized = mode.strip().lower()
        mapping = {
            "full": FlatHorizonEncoderMode.Full,
            "delta": FlatHorizonEncoderMode.Delta,
            "action": FlatHorizonEncoderMode.Action,
        }
        if normalized not in mapping:
            raise ValueError(f"Unsupported flat horizon transition_mode: {mode!r}")
        return mapping[normalized]
    name = getattr(mode, "name", None)
    if isinstance(name, str):
        return _normalize_flat_horizon_mode(name)
    if isinstance(mode, HorizonEncoderMode):
        return _normalize_flat_horizon_mode(mode.name)
    raise TypeError(
        "transition_mode must be a FlatHorizonEncoderMode, HorizonEncoderMode, "
        f"or str, got {type(mode)!r}"
    )


@dataclass
class FlatHorizonEncoderStream(StreamEncoderBase["FlatRelationData"]):
    """Append-only stream for flat horizon encodings.

    Each item is a root state plus an optional `TransitionDAG`. The flushed
    result matches direct horizon batch encoding.
    """

    _encoder: "FlatHorizonEncoder"

    def __post_init__(self) -> None:
        self._stream = _FlatHorizonStreamEncoder(self._encoder.engine)
        self._reset_builder()

    def append(
        self,
        root: StateInput,
        dag: TransitionDAG | RXStateDAG | None = None,
        *,
        goals: GoalBatchInput = None,
        subgoal_layers: SubgoalLayersInput = None,
    ) -> int:
        """Append one root/DAG input and return its stream id."""
        adv_root = _advanced_state(root)
        normalized_dag = ensure_transition_dag(root, dag)
        inputs = prepare_goal_inputs(root, goals, subgoal_layers)
        if dag is None:
            return self._coerce_stream_id(self._stream.append(adv_root, inputs))
        return self._coerce_stream_id(
            self._stream.append(adv_root, normalized_dag, inputs)
        )

    def _reset_builder(self) -> None:
        self._stream.reset()


@dataclass
class FlatHorizonMutableEncoderStream(StreamEncoderBase["FlatRelationData"]):
    """Mutable stream for flat horizon encodings."""

    _encoder: "FlatHorizonEncoder"

    def __post_init__(self) -> None:
        self._stream = _FlatHorizonStreamEncoder(self._encoder.engine)
        self._reset_builder()

    def append(
        self,
        root: StateInput,
        dag: TransitionDAG | RXStateDAG | None = None,
        *,
        goals: GoalBatchInput = None,
        subgoal_layers: SubgoalLayersInput = None,
    ) -> int:
        """Append one root/DAG input and return its stream id."""
        adv_root = _advanced_state(root)
        normalized_dag = ensure_transition_dag(root, dag)
        inputs = prepare_goal_inputs(root, goals, subgoal_layers)
        if dag is None:
            return self._coerce_stream_id(self._stream.append(adv_root, inputs))
        return self._coerce_stream_id(
            self._stream.append(adv_root, normalized_dag, inputs)
        )

    def remove(self, stream_id: int) -> None:
        self._stream.remove(stream_id)

    def update(
        self,
        stream_id: int,
        root: StateInput,
        dag: TransitionDAG | RXStateDAG | None = None,
        *,
        goals: GoalBatchInput = None,
        subgoal_layers: SubgoalLayersInput = None,
    ) -> None:
        """Replace one root/DAG input in place."""
        adv_root = _advanced_state(root)
        normalized_dag = ensure_transition_dag(root, dag)
        inputs = prepare_goal_inputs(root, goals, subgoal_layers)
        if dag is None:
            self._stream.update(stream_id, adv_root, inputs)
            return
        self._stream.update(stream_id, adv_root, normalized_dag, inputs)

    def _reset_builder(self) -> None:
        self._stream.reset()


class FlatHorizonEncoder(FlatRelationEncoder):
    """Encode a root state plus lookahead candidates as packed flat relations."""

    def __init__(
        self,
        domain: DomainInput,
        *,
        transition_mode: FlatHorizonEncoderMode
        | HorizonEncoderMode
        | str
        | None = None,
        target_symbol_prefix: str = "target:",
        parent_relation: str = DEFAULT_PARENT_RELATION,
        sibling_relation: str = "_sibling_",
        cousin_relation: str = "_cousin_",
        enable_parent_relation: bool = False,
        enable_sibling_relation: bool = False,
        enable_cousin_relation: bool = False,
        exclude_root_candidate: bool = True,
        max_goal_level: int = 0,
        support_literals: bool = False,
        include_static: bool = True,
        export_node_names: bool = True,
        ignore_zero_arity_relations: bool = True,
        ignore_actions: bool = True,
        include_lgan_edges: bool = False,
        lgan_tn_edge_pos: str = DEFAULT_LGAN_TN_EDGE_POS,
        lgan_nn_edge_pos: str = DEFAULT_LGAN_NN_EDGE_POS,
        lgan_rr_edge_pos: str = DEFAULT_LGAN_RR_EDGE_POS,
        goal_derivations: Any | None = None,
    ) -> None:
        """Create a flat horizon encoder.

        This lane reads a root state plus a `TransitionDAG` and creates
        candidate state rows. When `include_lgan_edges=True`, LGAN anchors are
        those candidate rows. There is no `lgan_anchor_sources` switch here.
        """
        config_kwargs: dict[str, Any] = {
            "max_goal_level": max_goal_level,
            "support_literals": support_literals,
            "include_static": include_static,
            "export_node_names": export_node_names,
            "ignore_zero_arity_relations": ignore_zero_arity_relations,
            "ignore_actions": ignore_actions,
            "include_lgan_edges": include_lgan_edges,
            "target_symbol_prefix": target_symbol_prefix,
            "parent_relation": parent_relation,
            "sibling_relation": sibling_relation,
            "cousin_relation": cousin_relation,
            "lgan_tn_edge_pos": lgan_tn_edge_pos,
            "lgan_nn_edge_pos": lgan_nn_edge_pos,
            "lgan_rr_edge_pos": lgan_rr_edge_pos,
            "enable_parent_relation": enable_parent_relation,
            "enable_sibling_relation": enable_sibling_relation,
            "enable_cousin_relation": enable_cousin_relation,
            "exclude_root_candidate": exclude_root_candidate,
        }
        normalized_mode = _normalize_flat_horizon_mode(transition_mode)
        if normalized_mode is not None:
            config_kwargs["transition_mode"] = normalized_mode
        if goal_derivations is not None:
            config_kwargs["goal_derivations"] = goal_derivations
        config = FlatHorizonEncoderConfig(**config_kwargs)
        self._engine = FlatHorizonEncoderEngine(_advanced_domain(domain), config)
        self._config = config
        self.entity_node_type = "entity"
        self.target_symbol_prefix = config.target_symbol_prefix
        self.parent_relation = config.parent_relation
        self.sibling_relation = config.sibling_relation
        self.cousin_relation = config.cousin_relation
        self.include_lgan_edges = bool(config.include_lgan_edges)
        self.lgan_tn_edge_pos = str(config.lgan_tn_edge_pos)
        self.lgan_nn_edge_pos = str(config.lgan_nn_edge_pos)
        self.lgan_rr_edge_pos = str(config.lgan_rr_edge_pos)
        self._lgan_edge_positions = {
            self.lgan_tn_edge_pos,
            self.lgan_nn_edge_pos,
            self.lgan_rr_edge_pos,
        }

    @property
    def engine(self) -> FlatHorizonEncoderEngine:
        """Expose the native flat horizon engine."""
        return self._engine

    @property
    def config(self) -> FlatHorizonEncoderConfig:
        """Expose the resolved native config."""
        return self._config

    @property
    def relation_dict(self):
        """Expose the relation schema used by the native engine."""
        return self._engine.relation_dict

    def _accepted_kwargs(self) -> set[str]:
        return {"dag", "dags", "history_subgoals", "history_max_steps"}

    def _encode(
        self,
        root: StateInput,
        *,
        goals: GoalBatchInput = None,
        actions: ActionBatchInput = None,
        subgoal_layers: SubgoalLayersInput = None,
        dag: TransitionDAG | RXStateDAG | None = None,
        history_subgoals: HistorySubgoalInput | None = None,
        history_max_steps: int | None = None,
    ) -> FlatEncoding:
        validate_single_optional_payloads(
            FLAT_HORIZON_LANE_SPEC,
            actions=actions,
            history_subgoals=history_subgoals,
            history_max_steps=history_max_steps,
        )
        adv_root = _advanced_state(root)
        dag = ensure_transition_dag(root, dag)
        inputs = prepare_goal_inputs(root, goals, subgoal_layers)
        return self._engine.encode(adv_root, dag, inputs)

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
    ) -> FlatEncoding:
        """Encode one root state and optional lookahead DAG.

        If `dag` is omitted, a one-node DAG for the root is used. `actions` and
        `history_subgoals` are accepted for API consistency but non-empty
        payloads are rejected on this lane.
        """
        return super().encode(
            root,
            goals=goals,
            actions=actions,
            subgoal_layers=subgoal_layers,
            dag=dag,
            history_subgoals=history_subgoals,
            history_max_steps=history_max_steps,
            include_metadata=include_metadata,
            **kwargs,
        )

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
    ) -> FlatEncoding:
        validate_batch_optional_payloads(
            FLAT_HORIZON_LANE_SPEC,
            actions=actions,
            history_subgoals=history_subgoals,
            history_max_steps=history_max_steps,
        )
        roots_for_core = _convert_batch_payload(
            roots,
            is_leaf=is_state_input,
            convert_leaf=to_advanced_state,
        )
        goals_for_core = _convert_batch_payload(
            goals,
            is_leaf=is_goal_literal_input,
            convert_leaf=to_advanced_literal,
        )
        subgoal_layers_for_core = _convert_batch_payload(
            subgoal_layers,
            is_leaf=is_goal_literal_input,
            convert_leaf=to_advanced_literal,
        )
        dags_for_core = _normalize_dag_batch_data(dags)
        return self._engine.encode_batch(
            roots_for_core,
            dags=dags_for_core,
            goals=goals_for_core,
            actions=None,
            subgoal_layers=subgoal_layers_for_core,
            history_subgoals=None,
            history_max_steps=None,
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
    ) -> FlatEncoding:
        """Encode many root/DAG inputs into one flat batch."""
        return super().encode_batch(
            roots,
            goals=goals,
            actions=actions,
            subgoal_layers=subgoal_layers,
            dags=dags,
            history_subgoals=history_subgoals,
            history_max_steps=history_max_steps,
            batch_attrs=batch_attrs,
            collate_spec=collate_spec,
            include_metadata=include_metadata,
            **kwargs,
        )

    def stream(self) -> FlatHorizonEncoderStream:
        """Return an append-only stream for root/DAG horizon inputs."""
        return FlatHorizonEncoderStream(self)

    def mutable_stream(self) -> FlatHorizonMutableEncoderStream:
        """Return a mutable stream with `append`, `update`, and `remove`."""
        return FlatHorizonMutableEncoderStream(self)


__all__ = [
    "FlatHorizonEncoder",
    "FlatHorizonEncoderStream",
    "FlatHorizonMutableEncoderStream",
]
