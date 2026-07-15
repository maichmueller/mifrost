"""Pymimir runtime for the public flat relation encoder."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Literal, cast

from .._core import (
    FlatRelationEncoderEngine,
    FlatRelationMutableStreamEncoder,
    FlatRelationStreamEncoder,
)
from ..encoders._batch_contract import prepare_core_batch_inputs
from ..encoders._lane_specs import prepare_optional_payloads
from ..encoders._flat_validation import validate_subgoal_layers_batch_payload
from ..encoders.common import _advanced_domain, _advanced_state, _split_goals
from ..encoders.types import default_goals_from_state


class PymimirFlatRuntime:
    """Preserve the historical native Pymimir flat fast path."""

    backend_name: Literal["pymimir"] = "pymimir"

    def __init__(self, domain: object, config: Any) -> None:
        self.engine = FlatRelationEncoderEngine(_advanced_domain(domain), config)

    @property
    def relation_dict(self) -> Any:
        return self.engine.relation_dict

    @staticmethod
    def _single_payload(
        state: object,
        *,
        goals: object = None,
        actions: object = None,
        subgoal_layers: object = None,
        history_subgoals: object = None,
    ) -> tuple[Any, Any, list[Any], list[tuple[int, list[Any]]]]:
        adv_state = _advanced_state(state)
        payloads = prepare_optional_payloads(
            actions=cast(Iterable[Any] | None, actions),
            history_subgoals=cast(Iterable[Any] | None, history_subgoals),
        )
        subgoal_values = (
            None
            if subgoal_layers is None
            else list(cast(Iterable[Any], subgoal_layers))
        )
        if goals is None and subgoal_layers is None and not payloads.history_subgoals:
            split_goals = None
        else:
            goal_values = (
                goals if goals is not None else default_goals_from_state(state)
            )
            split_goals = _split_goals(cast(Iterable[Any], goal_values), subgoal_values)
        return (
            adv_state,
            split_goals,
            payloads.actions,
            payloads.history_subgoals,
        )

    def append_into_builder(
        self,
        state: object,
        builder: Any,
        *,
        goals: object = None,
        actions: object = None,
        subgoal_layers: object = None,
        history_subgoals: object = None,
        history_max_steps: int | None = None,
    ) -> None:
        adv_state, split_goals, action_values, history_values = self._single_payload(
            state,
            goals=goals,
            actions=actions,
            subgoal_layers=subgoal_layers,
            history_subgoals=history_subgoals,
        )
        if split_goals is None:
            if action_values:
                self.engine.encode(adv_state, action_values, builder)
            else:
                self.engine.encode(adv_state, builder)
            return
        if history_values:
            if history_max_steps is None:
                self.engine.encode(
                    adv_state, split_goals, action_values, history_values, builder
                )
            else:
                self.engine.encode(
                    adv_state,
                    split_goals,
                    action_values,
                    history_values,
                    history_max_steps,
                    builder,
                )
            return
        if action_values:
            self.engine.encode(adv_state, split_goals, action_values, builder)
        else:
            self.engine.encode(adv_state, split_goals, builder)

    def encode_one(self, state: object, **kwargs: Any) -> Any:
        from .._core import BatchBuilder

        builder = BatchBuilder()
        builder.set_graph_kind("flat")
        self.append_into_builder(state, builder, **kwargs)
        builder.next_graph()
        encoding = builder.build()
        self.engine.finalize_batch_encoding(encoding)
        return encoding

    def encode_batch(
        self,
        states: object,
        *,
        goals: object = None,
        actions: object = None,
        subgoal_layers: object = None,
        history_subgoals: object = None,
        history_max_steps: int | None = None,
    ) -> Any:
        inputs = prepare_core_batch_inputs(
            states,
            goals=goals,
            actions=actions,
            subgoal_layers=subgoal_layers,
            history_subgoals=history_subgoals,
        )
        state_count = len(inputs.states) if hasattr(inputs.states, "__len__") else 1
        validate_subgoal_layers_batch_payload(
            inputs.subgoal_layers,
            state_count=state_count,
            max_goal_level=int(self.engine.config.max_goal_level),
        )
        return self.engine.encode_batch(
            inputs.states,
            goals=inputs.goals,
            actions=inputs.actions,
            subgoal_layers=inputs.subgoal_layers,
            history_subgoals=inputs.history_subgoals,
            history_max_steps=history_max_steps,
        )

    def make_stream(self, *, mutable: bool) -> Any:
        stream_type = (
            FlatRelationMutableStreamEncoder if mutable else FlatRelationStreamEncoder
        )
        return _PymimirFlatStream(self, stream_type(self.engine), mutable=mutable)


class _PymimirFlatStream:
    def __init__(
        self, runtime: PymimirFlatRuntime, native: Any, *, mutable: bool
    ) -> None:
        self._runtime = runtime
        self._native = native
        self._mutable = mutable

    def _invoke(
        self,
        method: Any,
        state: object,
        *,
        goals: object = None,
        actions: object = None,
        subgoal_layers: object = None,
        history_subgoals: object = None,
        history_max_steps: int | None = None,
    ) -> Any:
        adv_state, split_goals, action_values, history_values = (
            self._runtime._single_payload(
                state,
                goals=goals,
                actions=actions,
                subgoal_layers=subgoal_layers,
                history_subgoals=history_subgoals,
            )
        )
        if split_goals is None:
            return (
                method(adv_state, action_values) if action_values else method(adv_state)
            )
        if history_values:
            if history_max_steps is None:
                return method(adv_state, split_goals, action_values, history_values)
            return method(
                adv_state,
                split_goals,
                action_values,
                history_values,
                history_max_steps,
            )
        if action_values:
            return method(adv_state, split_goals, action_values)
        return method(adv_state, split_goals)

    def append(self, state: object, **kwargs: Any) -> Any:
        return self._invoke(self._native.append, state, **kwargs)

    def update(self, stream_id: int, state: object, **kwargs: Any) -> None:
        if not self._mutable:
            raise NotImplementedError("update is not implemented for this stream")
        self._invoke(
            lambda *args: self._native.update(stream_id, *args), state, **kwargs
        )

    def remove(self, stream_id: int) -> None:
        if not self._mutable:
            raise NotImplementedError("remove is not implemented for this stream")
        self._native.remove(stream_id)

    def set_reuse_removed(self, value: bool) -> None:
        self._native.set_reuse_removed(bool(value))

    def flush(self) -> Any:
        return self._native.flush()

    def reset(self) -> None:
        self._native.reset()


__all__ = ["PymimirFlatRuntime"]
