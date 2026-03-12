from __future__ import annotations

from typing import Any

import torch

from mifrost._core import BatchBuilder, BatchEncoding
from mifrost.encoders.base import (
    ActionBatchInput,
    ActionBatchParam,
    GoalBatchInput,
    GoalBatchParam,
    HistorySubgoalsBatchParam,
    StateBatchInput,
    SubgoalLayersInput,
    SubgoalLayersBatchParam,
)
from mifrost.encoders.flat import FlatRelationEncoder
from mifrost.encoders.types import FlatEncoding, HistorySubgoalInput, StateInput


def _as_tensor(value: Any, *, dtype: torch.dtype) -> torch.Tensor:
    if torch.is_tensor(value):
        return value.to(dtype=dtype).view(-1)
    try:
        return torch.utils.dlpack.from_dlpack(value).to(dtype=dtype).view(-1)
    except Exception:
        return torch.as_tensor(value, dtype=dtype).view(-1)


class ExampleTargetMaskFlatEncoder(FlatRelationEncoder):
    """Example flat encoder that adds dense target-row annotations."""

    entity_is_target_field = "entity_is_target"
    target_entity_count_field = "target_entity_count"

    def _with_target_mask_fields(self, encoding: BatchEncoding) -> BatchEncoding:
        target_entity_indices_obj = getattr(encoding, "target_entity_indices", ())
        target_entity_sizes_obj = getattr(encoding, "target_entity_sizes", ())

        target_entity_indices = _as_tensor(target_entity_indices_obj, dtype=torch.long)
        target_entity_sizes = _as_tensor(target_entity_sizes_obj, dtype=torch.long)
        if target_entity_sizes.numel() == 0:
            target_entity_sizes = torch.zeros(
                (int(encoding.num_graphs),), dtype=torch.long
            )

        entity_is_target = torch.zeros((int(encoding.num_nodes),), dtype=torch.float32)
        if target_entity_indices.numel() > 0:
            entity_is_target[target_entity_indices] = 1.0

        setattr(encoding, self.entity_is_target_field, entity_is_target)
        setattr(encoding, self.target_entity_count_field, target_entity_sizes)
        return encoding

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
        return self._with_target_mask_fields(builder.build())

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
        base = super()._encode_batch(
            states,
            goals=goals,
            actions=actions,
            subgoal_layers=subgoal_layers,
            history_subgoals=history_subgoals,
            history_max_steps=history_max_steps,
        )
        return self._with_target_mask_fields(base)
