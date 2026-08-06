"""Pymimir runtime for the public heterogeneous graph encoder."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any, Literal, cast

from ..encoders._batch_contract import prepare_core_batch_inputs
from .pymimir_types import default_goals_from_state
from .pymimir_common import _advanced_domain, _advanced_state, _split_goals
from .pymimir_lane_specs import prepare_optional_payloads


class PymimirHGraphRuntime:
    """Preserve the historical native Pymimir HGraph paths."""

    backend_name: Literal["pymimir"] = "pymimir"

    def __init__(self, domain: object, config: Any) -> None:
        from .._core import HGraphEncoderConfig, HGraphEncoderEngine

        native_config = HGraphEncoderConfig(
            symbol_type_id=str(config.symbol_type_id),
            target_symbol_prefix=str(config.target_symbol_prefix),
            ignore_actions=bool(config.ignore_actions),
            add_nullary_predicates=bool(config.add_nullary_predicates),
            include_lgan_edges=bool(config.include_lgan_edges),
            lgan_anchor_sources=set(config.lgan_anchor_sources),
            include_static=bool(config.include_static),
            include_empty_edge_types=bool(config.include_empty_edge_types),
            export_node_names=bool(config.export_node_names),
            target_sources=set(config.target_sources),
            max_goal_level=int(config.max_goal_level),
            support_literals=bool(config.support_literals),
            goal_derivations=set(config.goal_derivations),
            nullary_object_name=str(config.nullary_object_name),
            lgan_tn_edge_pos=str(config.lgan_tn_edge_pos),
            lgan_nn_edge_pos=str(config.lgan_nn_edge_pos),
            lgan_rr_edge_pos=str(config.lgan_rr_edge_pos),
            history_link_relation=str(config.history_link_relation),
        )
        self.engine = HGraphEncoderEngine(_advanced_domain(domain), native_config)

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
    ) -> tuple[Any, Any | None, list[Any], list[Any]]:
        native_state = _advanced_state(state)
        payloads = prepare_optional_payloads(
            actions=cast(Iterable[Any] | None, actions),
            history_subgoals=cast(Iterable[Any] | None, history_subgoals),
        )
        if (
            goals is None
            and subgoal_layers is None
            and not payloads.actions
            and not payloads.history_subgoals
        ):
            return native_state, None, [], []
        goal_values = goals if goals is not None else default_goals_from_state(state)
        layers = (
            None
            if subgoal_layers is None
            else list(cast(Iterable[Iterable[Any]], subgoal_layers))
        )
        return (
            native_state,
            _split_goals(cast(Iterable[Any], goal_values), layers),
            payloads.actions,
            payloads.history_subgoals,
        )

    def append_into_builder(
        self,
        state: object,
        builder: Any,
        *,
        history_max_steps: int | None = None,
        **kwargs: Any,
    ) -> None:
        native_state, goals, actions, history = self._single_payload(state, **kwargs)
        if goals is None:
            self.engine.encode(native_state, builder)
        elif history:
            if history_max_steps is None:
                self.engine.encode(native_state, goals, actions, history, builder)
            else:
                self.engine.encode(
                    native_state,
                    goals,
                    actions,
                    history,
                    history_max_steps,
                    builder,
                )
        else:
            self.engine.encode(native_state, goals, actions, builder)

    def encode(self, state: object, **kwargs: Any) -> Any:
        from .._core import BatchBuilder

        builder = BatchBuilder()
        builder.set_graph_kind("hetero")
        self.append_into_builder(state, builder, **kwargs)
        builder.next_graph()
        return builder.build()

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
        return self.engine.encode_batch(
            inputs.states,
            goals=inputs.goals,
            actions=inputs.actions,
            subgoal_layers=inputs.subgoal_layers,
            history_subgoals=inputs.history_subgoals,
            history_max_steps=history_max_steps,
        )

    def update_relations(self, relation_dict: Any) -> None:
        from .. import _core

        if isinstance(relation_dict, _core.RelationDict):
            native = relation_dict
        elif isinstance(relation_dict, Mapping):
            native = _core.RelationDict(dict(relation_dict))
        else:
            raise TypeError(
                "update_relations expects mifrost.RelationDict or a mapping[str, int]"
            )
        self.engine.update_relations(native)

    def make_stream(self, *, mutable: bool) -> Any:
        from .._core import HGraphMutableStreamEncoder, HGraphStreamEncoder

        stream_type = HGraphMutableStreamEncoder if mutable else HGraphStreamEncoder
        return _PymimirHGraphStream(self, stream_type(self.engine), mutable=mutable)


class _PymimirHGraphStream:
    def __init__(
        self, runtime: PymimirHGraphRuntime, native: Any, *, mutable: bool
    ) -> None:
        self._runtime = runtime
        self._native = native
        self._mutable = mutable

    def _invoke(self, method: Any, state: object, **kwargs: Any) -> Any:
        history_max_steps = kwargs.pop("history_max_steps", None)
        native_state, goals, actions, history = self._runtime._single_payload(
            state, **kwargs
        )
        if goals is None:
            return method(native_state)
        if history:
            return method(native_state, goals, actions, history, history_max_steps)
        return method(native_state, goals, actions)

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


__all__ = ["PymimirHGraphRuntime"]
