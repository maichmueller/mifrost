from __future__ import annotations

from typing import Any, Iterable, Mapping

import networkx as nx
import torch

from .._core import FlatRelationEncoderConfig, FlatRelationEncoderEngine
from ._action_contract import parse_flat_actions
from ._target_sources import TargetSource, normalize_target_sources
from .base import (
    ActionBatchInput,
    ActionBatchParam,
    CollateSpecParam,
    EncoderBase,
    GoalBatchInput,
    GoalBatchParam,
    StateBatchInput,
    SubgoalLayersInput,
    SubgoalLayersBatchParam,
)
from .common import (
    _advanced_domain,
    _advanced_state,
    _convert_batch_payload,
    _prepare_actions,
    _split_goals,
)
from .flat_data import FlatRelationData
from .types import (
    DomainInput,
    GoalLiteralInput,
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
        goal_satisfaction_derivations: Iterable[object] | None = None,
    ) -> None:
        normalized_target_sources = normalize_target_sources(target_sources)
        if normalized_target_sources is not None:
            unsupported_sources = normalized_target_sources.intersection(
                {TargetSource.States, TargetSource.History}
            )
            if unsupported_sources:
                raise ValueError(
                    "FlatRelationEncoder currently supports target_sources="
                    "{'action', 'goal', 'subgoal'} only; 'state' and 'history' "
                    "are reserved for the upcoming flat successor/horizon encoders"
                )
        config_kwargs: dict[str, object] = {
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

    def _encode(
        self,
        state: StateInput,
        *,
        goals: GoalBatchInput = None,
        actions: ActionBatchInput = None,
        subgoal_layers: SubgoalLayersInput = None,
    ) -> HomoEncoding:
        adv_state = _advanced_state(state)
        action_inputs = parse_flat_actions(actions)
        action_list = _prepare_actions(action_inputs)
        if goals is None and subgoal_layers is None:
            if action_list:
                return self._engine.encode(adv_state, action_list)
            return self._engine.encode(adv_state)
        goals_input = goals if goals is not None else default_goals_from_state(state)
        split_goals = _split_goals(goals_input, subgoal_layers)
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
        include_metadata: bool = True,
        **kwargs: object,
    ) -> HomoEncoding:
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
        return self._engine.encode_batch(
            states_for_core,
            goals=goals_for_core,
            actions=actions_for_core,
            subgoal_layers=subgoal_layers_for_core,
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
        **kwargs: object,
    ) -> HomoEncoding:
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

        for global_idx in range(start, end):
            label = name_by_global.get(global_idx, f"entity:{global_idx}")
            target_group_id, target_group_name = target_entity_group_by_index.get(
                global_idx, (None, None)
            )
            entity_kind = "target_entity" if target_group_name is not None else "object"
            graph.add_node(
                label,
                type="entity",
                kind="entity",
                entity_kind=entity_kind,
                target_group_id=target_group_id,
                target_group=target_group_name,
                global_index=global_idx,
                graph_index=graph_index,
            )

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
                "goal": "#f9d7dd",
                "subgoal": "#dbecc8",
                "action": "#d8ecff",
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


__all__ = ["FlatRelationEncoder"]
