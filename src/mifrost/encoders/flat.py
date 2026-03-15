from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import networkx as nx
import torch

from .._core import (
    BatchBuilder,
    DEFAULT_LGAN_NN_EDGE_POS,
    DEFAULT_LGAN_RR_EDGE_POS,
    DEFAULT_LGAN_TN_EDGE_POS,
    FlatRelationEncoderConfig,
    FlatRelationEncoderEngine,
    FlatRelationMutableStreamEncoder as _FlatRelationMutableStreamEncoder,
    FlatRelationStreamEncoder as _FlatRelationStreamEncoder,
)
from ._action_contract import parse_flat_actions
from ._target_sources import TargetSource, normalize_target_sources
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
from .flat_data import FlatRelationData
from .types import (
    DomainInput,
    FlatEncoding,
    GoalLiteralInput,
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


@dataclass
class FlatRelationEncoderStream(StreamEncoderBase[FlatRelationData]):
    """Append-only stream for flat state encodings.

    Each appended item follows the same input contract as
    :meth:`FlatRelationEncoder.encode`. The flushed result is the same flat
    packed carrier you would get from direct batch encoding.
    """

    _encoder: "FlatRelationEncoder"

    def __post_init__(self) -> None:
        self._stream = _FlatRelationStreamEncoder(self._encoder.engine)
        self._reset_builder()

    def append(
        self,
        state: StateInput,
        *,
        goals: GoalBatchInput = None,
        actions: ActionBatchInput = None,
        subgoal_layers: SubgoalLayersInput = None,
        history_subgoals: HistorySubgoalInput | None = None,
        history_max_steps: int | None = None,
    ) -> int:
        """Append one state payload and return its stream id.

        `goals`, `actions`, `subgoal_layers`, and `history_subgoals` are all
        optional. If `goals` are omitted, the problem goals from `state` are
        used when subgoals or history are requested.
        """
        adv_state = _advanced_state(state)
        action_inputs = parse_flat_actions(actions)
        action_list = _prepare_actions(action_inputs)
        history_list = _prepare_history_subgoals(history_subgoals)
        if goals is None and subgoal_layers is None and not history_list:
            if action_list:
                return self._coerce_stream_id(
                    self._stream.append(adv_state, action_list)
                )
            return self._coerce_stream_id(self._stream.append(adv_state))

        goals_input = goals if goals is not None else default_goals_from_state(state)
        split_goals = _split_goals(goals_input, subgoal_layers)
        if history_list:
            return self._coerce_stream_id(
                self._stream.append(
                    adv_state,
                    split_goals,
                    action_list,
                    history_list,
                    history_max_steps,
                )
            )
        if action_list:
            return self._coerce_stream_id(
                self._stream.append(adv_state, split_goals, action_list)
            )
        return self._coerce_stream_id(self._stream.append(adv_state, split_goals))

    def _reset_builder(self) -> None:
        self._stream.reset()


@dataclass
class FlatRelationMutableEncoderStream(StreamEncoderBase[FlatRelationData]):
    """Mutable stream for flat state encodings.

    This stream accepts the same payloads as
    :meth:`FlatRelationEncoder.encode`, but also supports `update` and
    `remove`.
    """

    _encoder: "FlatRelationEncoder"

    def __post_init__(self) -> None:
        self._stream = _FlatRelationMutableStreamEncoder(self._encoder.engine)
        self._reset_builder()

    def append(
        self,
        state: StateInput,
        *,
        goals: GoalBatchInput = None,
        actions: ActionBatchInput = None,
        subgoal_layers: SubgoalLayersInput = None,
        history_subgoals: HistorySubgoalInput | None = None,
        history_max_steps: int | None = None,
    ) -> int:
        """Append one state payload and return its stream id."""
        adv_state = _advanced_state(state)
        action_inputs = parse_flat_actions(actions)
        action_list = _prepare_actions(action_inputs)
        history_list = _prepare_history_subgoals(history_subgoals)
        if goals is None and subgoal_layers is None and not history_list:
            if action_list:
                return self._coerce_stream_id(
                    self._stream.append(adv_state, action_list)
                )
            return self._coerce_stream_id(self._stream.append(adv_state))

        goals_input = goals if goals is not None else default_goals_from_state(state)
        split_goals = _split_goals(goals_input, subgoal_layers)
        if history_list:
            return self._coerce_stream_id(
                self._stream.append(
                    adv_state,
                    split_goals,
                    action_list,
                    history_list,
                    history_max_steps,
                )
            )
        if action_list:
            return self._coerce_stream_id(
                self._stream.append(adv_state, split_goals, action_list)
            )
        return self._coerce_stream_id(self._stream.append(adv_state, split_goals))

    def remove(self, stream_id: int) -> None:
        """Remove one previously appended item by id."""
        self._stream.remove(stream_id)

    def update(
        self,
        stream_id: int,
        state: StateInput,
        *,
        goals: GoalBatchInput = None,
        actions: ActionBatchInput = None,
        subgoal_layers: SubgoalLayersInput = None,
        history_subgoals: HistorySubgoalInput | None = None,
        history_max_steps: int | None = None,
    ) -> None:
        """Replace one previously appended item in place."""
        adv_state = _advanced_state(state)
        action_inputs = parse_flat_actions(actions)
        action_list = _prepare_actions(action_inputs)
        history_list = _prepare_history_subgoals(history_subgoals)
        if goals is None and subgoal_layers is None and not history_list:
            if action_list:
                self._stream.update(stream_id, adv_state, action_list)
                return
            self._stream.update(stream_id, adv_state)
            return

        goals_input = goals if goals is not None else default_goals_from_state(state)
        split_goals = _split_goals(goals_input, subgoal_layers)
        if history_list:
            self._stream.update(
                stream_id,
                adv_state,
                split_goals,
                action_list,
                history_list,
                history_max_steps,
            )
            return
        if action_list:
            self._stream.update(stream_id, adv_state, split_goals, action_list)
            return
        self._stream.update(stream_id, adv_state, split_goals)

    def _reset_builder(self) -> None:
        self._stream.reset()


class FlatRelationEncoder(EncoderBase[FlatRelationData]):
    """Encode one planning state as packed flat relations.

    The output uses one flat entity table and packed relation tensors instead
    of relation nodes. Optional goals, subgoals, explicit actions, and history
    payloads can add more relations and helper rows on that same table.
    """

    def __init__(
        self,
        domain: DomainInput,
        *,
        max_goal_level: int = 0,
        support_literals: bool = False,
        include_static: bool = True,
        export_node_names: bool = True,
        ignore_zero_arity_relations: bool = True,
        include_lgan_edges: bool = False,
        lgan_anchor_sources: Iterable[TargetSource | str] | None = None,
        target_sources: Iterable[TargetSource | str] | None = None,
        target_symbol_prefix: str = "target:",
        lgan_tn_edge_pos: str = DEFAULT_LGAN_TN_EDGE_POS,
        lgan_nn_edge_pos: str = DEFAULT_LGAN_NN_EDGE_POS,
        lgan_rr_edge_pos: str = DEFAULT_LGAN_RR_EDGE_POS,
        goal_satisfaction_derivations: Iterable[Any] | None = None,
    ) -> None:
        """Build a flat encoder for state-style workloads.

        Parameters follow the flat main state lane.

        `target_sources` answers "what should count as a selectable target?":

        - `action`: explicit grounded actions from `actions=...`
        - `goal`: literals from the root `goals=...` input
        - `subgoal`: literals from `subgoal_layers=...`
        - `history`: literals from `history_subgoals=...`
        - `state`: not used here; state targets belong to `FlatHorizonEncoder`

        `lgan_anchor_sources` is separate. It only creates extra LGAN anchor
        rows for `goal`, `subgoal`, and `history`, without turning them into
        prediction targets. `include_lgan_edges=True` emits the packed LGAN
        edge tensors.

        `state` targets are not supported on this lane. Use
        `FlatHorizonEncoder` or the flat transition encoders for state
        candidates.
        """
        normalized_target_sources = normalize_target_sources(target_sources)
        normalized_lgan_anchor_sources = normalize_target_sources(lgan_anchor_sources)

        def _validate_flat_sources(
            sources: set[TargetSource] | None,
            field_name: str,
        ) -> None:
            if sources is None:
                return
            unsupported_sources = sources.intersection({TargetSource.States})
            if unsupported_sources:
                raise ValueError(
                    "FlatRelationEncoder currently supports "
                    f"{field_name}={{'action', 'goal', 'subgoal', 'history'}} "
                    "only; 'state' is reserved for the upcoming flat "
                    "successor/horizon encoders"
                )

        _validate_flat_sources(normalized_target_sources, "target_sources")
        _validate_flat_sources(normalized_lgan_anchor_sources, "lgan_anchor_sources")
        config_kwargs: dict[str, Any] = {
            "max_goal_level": max_goal_level,
            "support_literals": support_literals,
            "include_static": include_static,
            "export_node_names": export_node_names,
            "ignore_zero_arity_relations": ignore_zero_arity_relations,
            "include_lgan_edges": include_lgan_edges,
            "target_symbol_prefix": target_symbol_prefix,
            "lgan_tn_edge_pos": lgan_tn_edge_pos,
            "lgan_nn_edge_pos": lgan_nn_edge_pos,
            "lgan_rr_edge_pos": lgan_rr_edge_pos,
        }
        if normalized_lgan_anchor_sources is not None:
            config_kwargs["lgan_anchor_sources"] = normalized_lgan_anchor_sources
        if normalized_target_sources is not None:
            config_kwargs["target_sources"] = normalized_target_sources
        if goal_satisfaction_derivations is not None:
            config_kwargs["goal_satisfaction_derivations"] = (
                goal_satisfaction_derivations
            )
        config = FlatRelationEncoderConfig(**config_kwargs)
        self._engine = FlatRelationEncoderEngine(_advanced_domain(domain), config)
        self._config = config
        self.entity_node_type = "entity"
        self.include_lgan_edges = bool(config.include_lgan_edges)
        self.target_sources = set(config.target_sources)
        self.lgan_anchor_sources = set(config.lgan_anchor_sources)
        self.target_symbol_prefix = str(config.target_symbol_prefix)
        self.lgan_tn_edge_pos = str(config.lgan_tn_edge_pos)
        self.lgan_nn_edge_pos = str(config.lgan_nn_edge_pos)
        self.lgan_rr_edge_pos = str(config.lgan_rr_edge_pos)
        self._lgan_edge_positions = {
            self.lgan_tn_edge_pos,
            self.lgan_nn_edge_pos,
            self.lgan_rr_edge_pos,
        }

    @property
    def engine(self) -> FlatRelationEncoderEngine:
        """Expose the native flat relation engine."""
        return self._engine

    @property
    def config(self) -> FlatRelationEncoderConfig:
        """Expose the resolved native config."""
        return self._config

    @property
    def relation_dict(self):
        """Expose the relation schema used by the native engine."""
        return self._engine.relation_dict

    @property
    def relation_names(self) -> tuple[str, ...]:
        """Expose the ordered flat relation names declared by the native engine."""
        return tuple(str(name) for name in self._engine.relation_names)

    @property
    def relation_arities(self) -> tuple[int, ...]:
        """Expose the ordered flat relation arities declared by the native engine."""
        return tuple(int(arity) for arity in self._engine.relation_arities)

    @property
    def relation_sources(self) -> tuple[str, ...]:
        """Expose the ordered flat relation source labels declared by the native engine."""
        return tuple(str(source) for source in self._engine.relation_sources)

    def _encode_one_into_builder(
        self,
        state: StateInput,
        builder: BatchBuilder,
        *,
        goals: GoalBatchInput = None,
        actions: ActionBatchInput = None,
        subgoal_layers: SubgoalLayersInput = None,
        history_subgoals: HistorySubgoalInput | None = None,
        history_max_steps: int | None = None,
    ) -> None:
        """Append one flat encoding step into a caller-owned builder."""
        adv_state = _advanced_state(state)
        action_inputs = parse_flat_actions(actions)
        action_list = _prepare_actions(action_inputs)
        history_list = _prepare_history_subgoals(history_subgoals)
        if goals is None and subgoal_layers is None and not history_list:
            if action_list:
                self._engine.encode(adv_state, action_list, builder)
                return
            self._engine.encode(adv_state, builder)
            return

        goals_input = goals if goals is not None else default_goals_from_state(state)
        split_goals = _split_goals(goals_input, subgoal_layers)
        if history_list:
            self._engine.encode(
                adv_state,
                split_goals,
                action_list,
                history_list,
                history_max_steps,
                builder,
            )
            return
        if action_list:
            self._engine.encode(adv_state, split_goals, action_list, builder)
            return
        self._engine.encode(adv_state, split_goals, builder)

    def _accepted_kwargs(self) -> set[str]:
        return {"history_subgoals", "history_max_steps"}

    def _encode(
        self,
        state: StateInput,
        *,
        goals: GoalBatchInput = None,
        actions: ActionBatchInput = None,
        subgoal_layers: SubgoalLayersInput = None,
        history_subgoals: HistorySubgoalInput | None = None,
        history_max_steps: int | None = None,
    ) -> FlatEncoding:
        builder = BatchBuilder()
        builder.set_graph_kind("flat")
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
        actions: ActionBatchInput = None,
        subgoal_layers: SubgoalLayersInput = None,
        history_subgoals: HistorySubgoalInput | None = None,
        history_max_steps: int | None = None,
        include_metadata: bool = True,
        **kwargs,
    ) -> FlatEncoding:
        """Encode one state into the native flat carrier.

        If `goals` are omitted, the problem goals from `state` are used when
        needed. `actions`, `subgoal_layers`, and `history_subgoals` are all
        optional. Use `encode_pyg()` when you want a `FlatRelationData`
        wrapper directly.
        """
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
    ) -> FlatEncoding:
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
        **kwargs,
    ) -> FlatEncoding:
        """Encode many states with shared or per-state optional payloads.

        Batch kwargs follow the same rules as `encode`: each optional payload
        may be shared for all states or given separately per state.
        """
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

    def stream(self) -> FlatRelationEncoderStream:
        """Return an append-only stream for flat state encodings."""
        return FlatRelationEncoderStream(self)

    def mutable_stream(self) -> FlatRelationMutableEncoderStream:
        """Return a mutable stream with `append`, `update`, and `remove`."""
        return FlatRelationMutableEncoderStream(self)

    def to_networkx(
        self,
        data: FlatRelationData,
        *,
        graph_index: int = 0,
        mode: str = "star",
    ) -> nx.MultiDiGraph:
        """Build a debug graph view for one encoded flat graph.

        The returned graph is only for inspection. It expands each relation
        instance into a synthetic node and can overlay LGAN edges when they are
        present in `data`.
        """
        if mode != "star":
            raise ValueError(f"Unsupported flat visualization mode: {mode!r}")

        graph = nx.MultiDiGraph()
        lgan_tn_edge_pos = str(getattr(data, "lgan_tn_edge_pos", self.lgan_tn_edge_pos))
        lgan_nn_edge_pos = str(getattr(data, "lgan_nn_edge_pos", self.lgan_nn_edge_pos))
        lgan_rr_edge_pos = str(getattr(data, "lgan_rr_edge_pos", self.lgan_rr_edge_pos))
        start, end = data.graph_node_range(graph_index)
        node_names = data.graph_node_names(graph_index)
        name_by_global = {start + idx: node_names[idx] for idx in range(end - start)}
        history_entity_dt_by_index = {
            int(global_idx): int(dt)
            for global_idx, dt in zip(
                data.graph_history_entity_indices(graph_index).tolist(),
                data.graph_history_entity_dt(graph_index).tolist(),
                strict=True,
            )
        }
        target_entity_indices = data.graph_target_entity_indices(graph_index)
        target_entity_group_ids = data.graph_target_entity_group_ids(graph_index)
        target_entity_groups = list(getattr(data, "target_entity_groups", ()))
        target_entity_group_by_index: dict[int, tuple[int | None, str | None]] = {}
        for global_idx, group_id in zip(
            target_entity_indices.tolist(),
            target_entity_group_ids.tolist(),
            strict=True,
        ):
            group_name = None
            if 0 <= group_id < len(target_entity_groups):
                group_name = str(target_entity_groups[group_id])
            target_entity_group_by_index[int(global_idx)] = (int(group_id), group_name)
        target_groups = list(getattr(data, "target_groups", ()))
        target_positions = data.graph_target_positions(graph_index).tolist()
        target_indices = data.graph_target_indices(graph_index).tolist()
        target_candidate_ids = data.graph_target_candidate_ids(graph_index).tolist()
        target_group_ids = data.graph_target_group_ids(graph_index).tolist()
        target_names = data.graph_target_names(graph_index)
        target_depths_tensor = getattr(data, "target_depths", None)
        target_depths = (
            data.graph_target_depths(graph_index).tolist()
            if target_depths_tensor is not None
            else [None] * len(target_positions)
        )
        target_rows_by_position: dict[int, list[dict[str, Any]]] = {}
        for (
            position,
            target_index,
            candidate_id,
            group_id,
            target_name,
            target_depth,
        ) in zip(
            target_positions,
            target_indices,
            target_candidate_ids,
            target_group_ids,
            target_names,
            target_depths,
            strict=True,
        ):
            group_name = None
            if 0 <= group_id < len(target_groups):
                group_name = str(target_groups[group_id])
            target_rows_by_position.setdefault(int(position), []).append(
                {
                    "target_index": int(target_index),
                    "target_candidate_id": int(candidate_id),
                    "target_group_id": int(group_id),
                    "target_group": group_name,
                    "target_name": str(target_name),
                    "target_depth": (
                        None if target_depth is None else int(target_depth)
                    ),
                }
            )

        for global_idx in range(start, end):
            label = name_by_global.get(global_idx, f"entity:{global_idx}")
            target_group_id, target_group_name = target_entity_group_by_index.get(
                global_idx, (None, None)
            )
            history_dt = history_entity_dt_by_index.get(global_idx)
            target_rows = target_rows_by_position.get(global_idx, [])
            if target_group_name is not None:
                entity_kind = "target_entity"
            elif history_dt is not None:
                entity_kind = "history_entity"
            else:
                entity_kind = "object"
            graph.add_node(
                label,
                type="entity",
                kind="entity",
                entity_kind=entity_kind,
                history_dt=history_dt,
                target_group_id=target_group_id,
                target_group=target_group_name,
                target_rows=target_rows,
                global_index=global_idx,
                graph_index=graph_index,
            )
            if len(target_rows) == 1:
                graph.nodes[label].update(target_rows[0])

        flattened = data.flattened_relations_view(graph_index=graph_index)
        schema = data.schema
        relation_cursor, _ = data.graph_relation_instance_range(graph_index)
        relation_node_by_global: dict[int, str] = {}
        for relation_idx, relation_name in enumerate(schema.names):
            instances = flattened[relation_name]
            source = (
                schema.sources[relation_idx]
                if relation_idx < len(schema.sources)
                else "relation"
            )
            for instance_idx, args in enumerate(instances.tolist()):
                relation_node = f"{relation_name}#{instance_idx}"
                graph.add_node(
                    relation_node,
                    type=relation_name,
                    kind="relation",
                    source=source,
                    relation_name=relation_name,
                    relation_id=relation_idx,
                    instance_index=instance_idx,
                    global_relation_index=relation_cursor,
                )
                relation_node_by_global[relation_cursor] = relation_node
                for slot, global_idx in enumerate(args):
                    entity_node = name_by_global.get(global_idx, f"entity:{global_idx}")
                    graph.add_edge(
                        relation_node,
                        entity_node,
                        position=str(slot),
                        slot=slot,
                    )
                relation_cursor += 1

        lgan_specs = (
            ("tn", data.graph_lgan_tn_edges(graph_index), lgan_tn_edge_pos),
            ("nn", data.graph_lgan_nn_edges(graph_index), lgan_nn_edge_pos),
        )
        for lgan_kind, edges, edge_pos in lgan_specs:
            if edges.numel() == 0:
                continue
            for relation_index, entity_index in edges.t().tolist():
                relation_node = relation_node_by_global.get(int(relation_index))
                if relation_node is None:
                    continue
                entity_node = name_by_global.get(
                    int(entity_index), f"entity:{int(entity_index)}"
                )
                graph.add_edge(
                    relation_node,
                    entity_node,
                    position=edge_pos,
                    lgan_kind=lgan_kind,
                    style="dashed",
                )

        rr_edges = data.graph_lgan_rr_edges(graph_index)
        if rr_edges.numel() > 0:
            for src_relation_index, dst_relation_index in rr_edges.t().tolist():
                src_relation_node = relation_node_by_global.get(int(src_relation_index))
                dst_relation_node = relation_node_by_global.get(int(dst_relation_index))
                if src_relation_node is None or dst_relation_node is None:
                    continue
                graph.add_edge(
                    src_relation_node,
                    dst_relation_node,
                    position=lgan_rr_edge_pos,
                    lgan_kind="rr",
                    style="dashed",
                )
        return graph

    def draw(
        self,
        data: FlatRelationData | nx.Graph,
        *,
        graph_index: int = 0,
        ax=None,
        with_labels: bool = True,
        edge_labels: bool = True,
        layout: dict | None = None,
    ):
        """Draw a flat debug graph with matplotlib.

        Pass either `FlatRelationData` or a graph produced by
        :meth:`to_networkx`. LGAN edges are shown as dashed overlays when they
        exist.
        """
        try:
            import matplotlib.pyplot as plt
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "FlatRelationEncoder.draw requires matplotlib to be installed"
            ) from exc

        graph = (
            data
            if isinstance(data, nx.Graph)
            else self.to_networkx(data, graph_index=graph_index)
        )
        if ax is None:
            _, ax = plt.subplots()

        pos = layout or nx.spring_layout(graph, seed=0)
        entity_nodes = [
            node for node, attrs in graph.nodes(data=True) if attrs["kind"] == "entity"
        ]
        relation_nodes = [
            node
            for node, attrs in graph.nodes(data=True)
            if attrs["kind"] == "relation"
        ]
        relation_sources = [
            graph.nodes[node].get("source", "relation") for node in relation_nodes
        ]

        if entity_nodes:
            entity_groups = [
                graph.nodes[node].get("target_group")
                or graph.nodes[node].get("entity_kind", "object")
                for node in entity_nodes
            ]
            entity_palette = {
                "object": "#f3efe0",
                "state": "#d6f0c0",
                "goal": "#f9d7dd",
                "subgoal": "#dbecc8",
                "action": "#d8ecff",
                "history": "#e6d7ff",
                "history_entity": "#e8def8",
                "target_entity": "#d8ecff",
            }
            nx.draw_networkx_nodes(
                graph,
                pos,
                nodelist=entity_nodes,
                node_color=[
                    entity_palette.get(group, "#d8ecff") for group in entity_groups
                ],
                edgecolors="#222222",
                linewidths=1.3,
                node_size=420,
                ax=ax,
            )
        if relation_nodes:
            source_palette = {
                "state": "#355c7d",
                "goal": "#c06c84",
                "goal_satisfaction": "#6c9a8b",
                "action": "#f08a24",
                "history": "#7b5ea7",
                "parent": "#4c6a92",
                "sibling": "#6f8f72",
                "cousin": "#8a6f9e",
                "relation": "#666666",
            }
            nx.draw_networkx_nodes(
                graph,
                pos,
                nodelist=relation_nodes,
                node_color=[
                    source_palette.get(source, "#666666") for source in relation_sources
                ],
                node_shape="s",
                edgecolors="#111111",
                linewidths=1.1,
                node_size=520,
                ax=ax,
            )

        normal_edges = []
        lgan_edges = []
        lgan_colors = []
        lgan_palette = {
            "tn": "#d35454",
            "nn": "#4f7cac",
            "rr": "#5a9367",
        }
        for u, v, attrs in graph.edges(data=True):
            lgan_kind = attrs.get("lgan_kind")
            if lgan_kind is None:
                normal_edges.append((u, v))
                continue
            lgan_edges.append((u, v))
            lgan_colors.append(lgan_palette.get(str(lgan_kind), "#666666"))

        if normal_edges:
            nx.draw_networkx_edges(
                graph,
                pos,
                edgelist=normal_edges,
                ax=ax,
                arrows=True,
                width=1.2,
                alpha=0.8,
            )
        if lgan_edges:
            nx.draw_networkx_edges(
                graph,
                pos,
                edgelist=lgan_edges,
                ax=ax,
                arrows=True,
                width=1.4,
                alpha=0.85,
                style="dashed",
                edge_color=lgan_colors,
            )

        if with_labels:
            nx.draw_networkx_labels(graph, pos, ax=ax, font_size=8)

        if edge_labels:
            labels = {
                tuple(edge_key): attrs.get("position")
                for *edge_key, attrs in graph.edges(keys=False, data=True)
                if attrs.get("position") is not None
            }
            if labels:
                nx.draw_networkx_edge_labels(
                    graph,
                    pos,
                    edge_labels=labels,
                    ax=ax,
                    font_size=7,
                )

        ax.set_axis_off()
        return ax


__all__ = [
    "FlatRelationEncoder",
    "FlatRelationEncoderStream",
    "FlatRelationMutableEncoderStream",
]
