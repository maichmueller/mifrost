"""Pymimir runtime for public transition HGraph encoders."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any, Literal, cast

from ..encoders._batch_contract import prepare_core_batch_inputs
from .pymimir_types import default_goals_from_state
from .pymimir_common import _advanced_domain, _advanced_state, _split_goals


class PymimirTransitionRuntime:
    """Preserve native Pymimir successor engine, parser, and stream paths."""

    backend_name: Literal["pymimir"] = "pymimir"

    def __init__(self, domain: object, config: Any) -> None:
        from .. import _core, _neutral_core

        mode = (
            _core.SuccessorEncoderMode.delta
            if config.successor_mode == _neutral_core.SemanticSuccessorEncoderMode.delta
            else _core.SuccessorEncoderMode.full
        )
        self._encoder_name = (
            "TransitionEffectsHGraphEncoder"
            if mode == _core.SuccessorEncoderMode.delta
            else "TransitionHGraphEncoder"
        )
        native_config = _core.SuccessorEncoderConfig(
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
            successor_mode=mode,
            successor_suffix=str(config.successor_suffix),
            include_successor_goal_satisfaction=bool(
                config.include_successor_goal_satisfaction
            ),
        )
        self.engine = _core.SuccessorHGraphEncoderEngine(
            _advanced_domain(domain), native_config
        )

    @property
    def relation_dict(self) -> Any:
        return self.engine.relation_dict

    @staticmethod
    def _single_payload(
        current: object,
        successor: object,
        *,
        goals: object = None,
        subgoal_layers: object = None,
    ) -> tuple[Any, Any, Any]:
        native_current = _advanced_state(current)
        native_successor = _advanced_state(successor)
        goal_values = goals if goals is not None else default_goals_from_state(current)
        layers = (
            None
            if subgoal_layers is None
            else list(cast(Iterable[Iterable[Any]], subgoal_layers))
        )
        return (
            native_current,
            native_successor,
            _split_goals(cast(Iterable[Any], goal_values), layers),
        )

    def append_into_builder(
        self,
        current: object,
        successor: object,
        builder: Any,
        **kwargs: Any,
    ) -> None:
        native_current, native_successor, goals = self._single_payload(
            current, successor, **kwargs
        )
        self.engine.encode(native_current, native_successor, goals, builder)

    def encode_one(self, current: object, successor: object, **kwargs: Any) -> Any:
        from .._core import BatchBuilder

        builder = BatchBuilder()
        builder.set_graph_kind("hetero")
        self.append_into_builder(current, successor, builder, **kwargs)
        builder.next_graph()
        return builder.build()

    def encode_batch(
        self,
        states: object,
        *,
        successors: object,
        goals: object = None,
        subgoal_layers: object = None,
    ) -> Any:
        inputs = prepare_core_batch_inputs(
            states,
            goals=goals,
            subgoal_layers=subgoal_layers,
            successors=successors,
        )
        return self.engine.encode_batch(
            self._encoder_name,
            inputs.states,
            inputs.successors,
            inputs.goals,
            None,
            inputs.subgoal_layers,
            None,
            None,
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

    def make_stream(self) -> Any:
        from .._core import TransitionStreamEncoder

        return _PymimirTransitionStream(self, TransitionStreamEncoder(self.engine))


class _PymimirTransitionStream:
    def __init__(self, runtime: PymimirTransitionRuntime, native: Any) -> None:
        self._runtime = runtime
        self._native = native

    def append(self, current: object, successor: object, **kwargs: Any) -> Any:
        payload = self._runtime._single_payload(current, successor, **kwargs)
        return self._native.append(*payload)

    def update(
        self,
        stream_id: int,
        current: object,
        successor: object,
        **kwargs: Any,
    ) -> None:
        payload = self._runtime._single_payload(current, successor, **kwargs)
        self._native.update(stream_id, *payload)

    def remove(self, stream_id: int) -> None:
        self._native.remove(stream_id)

    def set_reuse_removed(self, value: bool) -> None:
        self._native.set_reuse_removed(bool(value))

    def flush(self) -> Any:
        return self._native.flush()

    def reset(self) -> None:
        self._native.reset()


__all__ = ["PymimirTransitionRuntime"]
