from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import networkx as nx
import torch

from .._core import (
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
    GoalLiteralInput,
    HistorySubgoalInput,
    HomoEncoding,
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
    """Append-only stream wrapper for ``FlatRelationEncoderEngine``."""

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
    """Mutable stream wrapper for ``FlatRelationEncoderEngine``."""

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
    """Flat packed relation encoder for state/goal workloads."""

    def __init__(
        self,
        domain: DomainInput,
        *,
        max_goal_level: int = 0,
        support_literals: bool = False,
        include_static: bool = True,
        export_node_names: bool = True,
        ignore_zero_arity_relations: bool = True,
        target_sources: Iterable[TargetSource | str] | None = None,
        target_symbol_prefix: str = "target:",
        goal_satisfaction_derivations: Iterable[Any] | None = None,
    ) -> None:
        normalized_target_sources = normalize_target_sources(target_sources)
        if normalized_target_sources is not None:
            unsupported_sources = normalized_target_sources.intersection(
                {TargetSource.States}
            )
            if unsupported_sources:
                raise ValueError(
                    "FlatRelationEncoder currently supports target_sources="
                    "{'action', 'goal', 'subgoal', 'history'} only; 'state' "
                    "is reserved for the upcoming flat successor/horizon encoders"
                )
        config_kwargs: dict[str, Any] = {
            "max_goal_level": max_goal_level,
            "support_literals": support_literals,
            "include_static": include_static,
            "export_node_names": export_node_names,
            "ignore_zero_arity_relations": ignore_zero_arity_relations,
            "target_symbol_prefix": target_symbol_prefix,
        }
        if normalized_target_sources is not None:
            config_kwargs["target_sources"] = normalized_target_sources
        if goal_satisfaction_derivations is not None:
            config_kwargs["goal_satisfaction_derivations"] = (
                goal_satisfaction_derivations
            )
        config = FlatRelationEncoderConfig(**config_kwargs)
        self._engine = FlatRelationEncoderEngine(_advanced_domain(domain), config)
        self._config = config

    @property
    def engine(self) -> FlatRelationEncoderEngine:
        return self._engine

    @property
    def config(self) -> FlatRelationEncoderConfig:
        return self._config

    @property
    def relation_dict(self):
        return self._engine.relation_dict

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
    ) -> HomoEncoding:
        adv_state = _advanced_state(state)
        action_inputs = parse_flat_actions(actions)
        action_list = _prepare_actions(action_inputs)
        history_list = _prepare_history_subgoals(history_subgoals)
        if goals is None and subgoal_layers is None and not history_list:
            if action_list:
                return self._engine.encode(adv_state, action_list)
            return self._engine.encode(adv_state)
        goals_input = goals if goals is not None else default_goals_from_state(state)
        split_goals = _split_goals(goals_input, subgoal_layers)
        if history_list:
            return self._engine.encode(
                adv_state,
                split_goals,
                action_list,
                history_list,
                history_max_steps,
            )
        if action_list:
            return self._engine.encode(adv_state, split_goals, action_list)
        return self._engine.encode(adv_state, split_goals)

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
    ) -> HomoEncoding:
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
    ) -> HomoEncoding:
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
    ) -> HomoEncoding:
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
        return FlatRelationEncoderStream(self)

    def mutable_stream(self) -> FlatRelationMutableEncoderStream:
        return FlatRelationMutableEncoderStream(self)

    def to_networkx(
        self,
        data: FlatRelationData,
        *,
        graph_index: int = 0,
        mode: str = "star",
    ) -> nx.MultiDiGraph:
        if mode != "star":
            raise ValueError(f"Unsupported flat visualization mode: {mode!r}")

        graph = nx.MultiDiGraph()
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
                )
                for slot, global_idx in enumerate(args):
                    entity_node = name_by_global.get(global_idx, f"entity:{global_idx}")
                    graph.add_edge(
                        relation_node,
                        entity_node,
                        position=str(slot),
                        slot=slot,
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

        nx.draw_networkx_edges(graph, pos, ax=ax, arrows=True, width=1.2, alpha=0.8)

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
