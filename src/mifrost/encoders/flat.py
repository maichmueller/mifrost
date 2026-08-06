from __future__ import annotations

import colorsys
from dataclasses import dataclass
import hashlib
from typing import TYPE_CHECKING, Any, Iterable, Mapping

if TYPE_CHECKING:
    import networkx as nx

from .._core import (
    BatchBuilder,
    DEFAULT_LGAN_NN_EDGE_POS,
    DEFAULT_LGAN_RR_EDGE_POS,
    DEFAULT_LGAN_TN_EDGE_POS,
    FlatRelationEncoderConfig,
)
from ..backends._flat_runtime import FlatBackendName, create_flat_runtime
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
from ._flat_validation import (
    validate_subgoal_layers_state_payload,
)
from .flat_data import FlatRelationData
from .types import (
    DomainInput,
    FlatEncoding,
    HistorySubgoalInput,
    StateInput,
)


def _default_flat_debug_layout(
    graph: nx.Graph,
) -> dict[Any, tuple[float, float]]:
    """Lay out flat debug graphs in stable columns to avoid edge collapse."""
    import networkx as nx

    if graph.number_of_nodes() == 0:
        return {}

    def subset_for(attrs: Mapping[str, Any]) -> int:
        if attrs.get("kind") == "relation":
            return 1
        if (
            attrs.get("target_group") is not None
            or attrs.get("entity_kind") == "target_entity"
        ):
            return 0
        if attrs.get("entity_kind") == "history_entity":
            return 3
        return 2

    def sort_key(node: Any) -> tuple[Any, ...]:
        attrs = graph.nodes[node]
        return (
            subset_for(attrs),
            str(
                attrs.get("target_group")
                or attrs.get("entity_kind")
                or attrs.get("source")
                or ""
            ),
            str(attrs.get("relation_name") or node),
            int(attrs.get("instance_index", 0)),
            str(node),
        )

    layout_graph = nx.DiGraph()
    for node in sorted(graph.nodes, key=sort_key):
        attrs = dict(graph.nodes[node])
        attrs["_flat_subset"] = subset_for(attrs)
        layout_graph.add_node(node, **attrs)

    pos = nx.multipartite_layout(
        layout_graph,
        subset_key="_flat_subset",
        align="vertical",
        scale=2.0,
    )
    return {
        node: (float(coords[0]) * 3.0, float(coords[1]) * 6.0)
        for node, coords in pos.items()
    }


def _relation_name_color(relation_name: str) -> str:
    """Return a stable color derived only from the relation name."""
    digest = hashlib.blake2b(relation_name.encode("utf-8"), digest_size=4).digest()
    hue = int.from_bytes(digest[:2], "big") / 65535.0
    saturation = 0.45 + (digest[2] / 255.0) * 0.25
    value = 0.70 + (digest[3] / 255.0) * 0.18
    red, green, blue = colorsys.hsv_to_rgb(hue, saturation, value)
    return "#{:02x}{:02x}{:02x}".format(
        int(red * 255),
        int(green * 255),
        int(blue * 255),
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
        self._stream = self._encoder._runtime.make_stream(mutable=False)
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
        subgoal_layers_list = None if subgoal_layers is None else list(subgoal_layers)
        validate_subgoal_layers_state_payload(
            subgoal_layers_list,
            state_index=0,
            max_goal_level=int(self._encoder.config.max_goal_level),
        )
        return self._coerce_stream_id(
            self._stream.append(
                state,
                goals=goals,
                actions=actions,
                subgoal_layers=subgoal_layers_list,
                history_subgoals=history_subgoals,
                history_max_steps=history_max_steps,
            )
        )

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
        self._stream = self._encoder._runtime.make_stream(mutable=True)
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
        subgoal_layers_list = None if subgoal_layers is None else list(subgoal_layers)
        validate_subgoal_layers_state_payload(
            subgoal_layers_list,
            state_index=0,
            max_goal_level=int(self._encoder.config.max_goal_level),
        )
        return self._coerce_stream_id(
            self._stream.append(
                state,
                goals=goals,
                actions=actions,
                subgoal_layers=subgoal_layers_list,
                history_subgoals=history_subgoals,
                history_max_steps=history_max_steps,
            )
        )

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
        subgoal_layers_list = None if subgoal_layers is None else list(subgoal_layers)
        validate_subgoal_layers_state_payload(
            subgoal_layers_list,
            state_index=0,
            max_goal_level=int(self._encoder.config.max_goal_level),
        )
        self._stream.update(
            stream_id,
            state,
            goals=goals,
            actions=actions,
            subgoal_layers=subgoal_layers_list,
            history_subgoals=history_subgoals,
            history_max_steps=history_max_steps,
        )

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
        backend: FlatBackendName | str | None = None,
        max_goal_level: int = 0,
        support_literals: bool = False,
        include_static: bool = True,
        export_node_names: bool = True,
        ignore_zero_arity_relations: bool = True,
        use_predicate_virtual_nodes: bool = False,
        include_lgan_edges: bool = False,
        lgan_anchor_sources: Iterable[TargetSource | str] | None = None,
        target_sources: Iterable[TargetSource | str] | None = None,
        target_symbol_prefix: str = "target:",
        lgan_tn_edge_pos: str = DEFAULT_LGAN_TN_EDGE_POS,
        lgan_nn_edge_pos: str = DEFAULT_LGAN_NN_EDGE_POS,
        lgan_rr_edge_pos: str = DEFAULT_LGAN_RR_EDGE_POS,
        pack_relation_args_relation_major: bool = False,
        goal_derivations: Iterable[Any] | None = None,
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
            unsupported_sources = sources.intersection({TargetSource.states})
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
            "use_predicate_virtual_nodes": use_predicate_virtual_nodes,
            "include_lgan_edges": include_lgan_edges,
            "target_symbol_prefix": target_symbol_prefix,
            "lgan_tn_edge_pos": lgan_tn_edge_pos,
            "lgan_nn_edge_pos": lgan_nn_edge_pos,
            "lgan_rr_edge_pos": lgan_rr_edge_pos,
            "pack_relation_args_relation_major": pack_relation_args_relation_major,
        }
        if normalized_lgan_anchor_sources is not None:
            config_kwargs["lgan_anchor_sources"] = normalized_lgan_anchor_sources
        if normalized_target_sources is not None:
            config_kwargs["target_sources"] = normalized_target_sources
        if goal_derivations is not None:
            config_kwargs["goal_derivations"] = goal_derivations
        config = FlatRelationEncoderConfig(**config_kwargs)
        self._runtime = create_flat_runtime(domain, config, backend=backend)
        self._engine = self._runtime.engine
        self._config = config
        self.backend = self._runtime.backend_name
        self.entity_node_type = "entity"
        self.use_predicate_virtual_nodes = bool(config.use_predicate_virtual_nodes)
        self.include_lgan_edges = bool(config.include_lgan_edges)
        self.target_sources = set(config.target_sources)
        self.lgan_anchor_sources = set(config.lgan_anchor_sources)
        self.target_symbol_prefix = str(config.target_symbol_prefix)
        self.lgan_tn_edge_pos = str(config.lgan_tn_edge_pos)
        self.lgan_nn_edge_pos = str(config.lgan_nn_edge_pos)
        self.lgan_rr_edge_pos = str(config.lgan_rr_edge_pos)
        self.pack_relation_args_relation_major = bool(
            config.pack_relation_args_relation_major
        )
        self._lgan_edge_positions = {
            self.lgan_tn_edge_pos,
            self.lgan_nn_edge_pos,
            self.lgan_rr_edge_pos,
        }

    @property
    def engine(self) -> Any:
        """Expose the native flat relation engine."""
        return self._engine

    @property
    def config(self) -> FlatRelationEncoderConfig:
        """Expose the resolved native config."""
        return self._config

    @property
    def relation_dict(self):
        """Expose the relation schema used by the native engine."""
        return self._runtime.relation_dict

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

    @property
    def relation_logical_arities(self) -> tuple[int, ...]:
        """Expose the logical flat relation arities declared by the native engine."""
        return tuple(int(arity) for arity in self._engine.relation_logical_arities)

    @property
    def relation_encoded_arities(self) -> tuple[int, ...]:
        """Expose the encoded flat relation arities declared by the native engine."""
        return tuple(int(arity) for arity in self._engine.relation_encoded_arities)

    @property
    def relation_slot_roles(self) -> tuple[int, ...]:
        """Expose flattened per-relation slot-role ids."""
        return tuple(int(role_id) for role_id in self._engine.relation_slot_roles)

    @property
    def relation_slot_role_offsets(self) -> tuple[int, ...]:
        """Expose offsets into `relation_slot_roles` for each relation."""
        return tuple(int(offset) for offset in self._engine.relation_slot_role_offsets)

    @property
    def slot_role_names(self) -> tuple[str, ...]:
        """Expose the ordered slot-role labels used by flat schema metadata."""
        return tuple(str(name) for name in self._engine.slot_role_names)

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
        subgoal_layers_list = None if subgoal_layers is None else list(subgoal_layers)
        validate_subgoal_layers_state_payload(
            subgoal_layers_list,
            state_index=0,
            max_goal_level=int(self._config.max_goal_level),
        )
        self._runtime.append_into_builder(
            state,
            builder,
            goals=goals,
            actions=actions,
            subgoal_layers=subgoal_layers_list,
            history_subgoals=history_subgoals,
            history_max_steps=history_max_steps,
        )

    def _accepted_kwargs(self) -> set[str]:
        return super()._accepted_kwargs() | {"history_subgoals", "history_max_steps"}

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
        subgoal_layers_list = None if subgoal_layers is None else list(subgoal_layers)
        validate_subgoal_layers_state_payload(
            subgoal_layers_list,
            state_index=0,
            max_goal_level=int(self._config.max_goal_level),
        )
        return self._runtime.encode(
            state,
            goals=goals,
            actions=actions,
            subgoal_layers=subgoal_layers_list,
            history_subgoals=history_subgoals,
            history_max_steps=history_max_steps,
        )

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
        return self._runtime.encode_batch(
            states,
            goals=goals,
            actions=actions,
            subgoal_layers=subgoal_layers,
            history_subgoals=history_subgoals,
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
        import networkx as nx

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
        entity_roles = data.graph_entity_roles(graph_index)
        entity_role_by_index = {
            start + idx: entity_roles[idx]
            for idx in range(min(len(entity_roles), end - start))
        }

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
            entity_role = entity_role_by_index.get(global_idx)
            graph.add_node(
                label,
                type="entity",
                kind="entity",
                entity_kind=entity_kind,
                entity_role=entity_role,
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
                slot_roles = (
                    schema.slot_roles[relation_idx]
                    if relation_idx < len(schema.slot_roles)
                    else ()
                )
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
                    slot_role = slot_roles[slot] if slot < len(slot_roles) else None
                    graph.add_edge(
                        relation_node,
                        entity_node,
                        position=str(slot),
                        slot=slot,
                        slot_role=slot_role,
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
        import networkx as nx

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

        pos = layout or _default_flat_debug_layout(graph)
        entity_nodes = [
            node for node, attrs in graph.nodes(data=True) if attrs["kind"] == "entity"
        ]
        relation_nodes = [
            node
            for node, attrs in graph.nodes(data=True)
            if attrs["kind"] == "relation"
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
            nx.draw_networkx_nodes(
                graph,
                pos,
                nodelist=relation_nodes,
                node_color=[
                    _relation_name_color(
                        str(graph.nodes[node].get("relation_name", node))
                    )
                    for node in relation_nodes
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
            if graph.is_multigraph():
                labels = {
                    (u, v, key): attrs.get("position")
                    for u, v, key, attrs in graph.edges(keys=True, data=True)
                    if attrs.get("position") is not None
                }
            else:
                labels = {
                    (u, v): attrs.get("position")
                    for u, v, attrs in graph.edges(data=True)
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
