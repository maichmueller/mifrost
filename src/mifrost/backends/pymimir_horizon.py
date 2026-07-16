"""Pymimir runtime for the public heterogeneous Horizon encoder."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Literal, cast

from ..encoders._batch_contract import (
    parse_dags_batch_param,
    parse_states_batch,
    prepare_core_batch_inputs,
)
from ..encoders._rustworkx_dag import RXStateDAG, _normalize_dag_batch_data
from .pymimir_common import _advanced_domain, _advanced_state
from .pymimir_lane_specs import ensure_transition_dag, prepare_goal_inputs

if TYPE_CHECKING:
    from .._core import TransitionDAG
else:
    TransitionDAG = Any


def _enum_name(value: object) -> str:
    text = str(value)
    return text.rsplit(".", 1)[-1]


class PymimirHorizonRuntime:
    """Preserve historical native Pymimir Horizon paths and identities."""

    backend_name: Literal["pymimir"] = "pymimir"

    def __init__(self, domain: object, config: Any) -> None:
        from .. import _core

        native_config = _core.HorizonEncoderConfig(
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
            transition_mode=getattr(
                _core.HorizonEncoderMode, _enum_name(config.transition_mode)
            ),
            parent_relation=str(config.parent_relation),
            sibling_relation=str(config.sibling_relation),
            cousin_relation=str(config.cousin_relation),
            enable_parent_relation=bool(config.enable_parent_relation),
            enable_sibling_relation=bool(config.enable_sibling_relation),
            enable_cousin_relation=bool(config.enable_cousin_relation),
            root_policy=config.root_policy,
        )
        self.engine = _core.HorizonHGraphEncoderEngine(
            _advanced_domain(domain), native_config
        )

    @property
    def relation_dict(self) -> Any:
        return self.engine.relation_dict

    @staticmethod
    def _payload(
        root: object,
        dag: object,
        **kwargs: Any,
    ) -> tuple[Any, Any, Any]:
        return (
            _advanced_state(root),
            ensure_transition_dag(root, cast(TransitionDAG | RXStateDAG | None, dag)),
            prepare_goal_inputs(
                root, kwargs.get("goals"), kwargs.get("subgoal_layers")
            ),
        )

    def append_into_builder(
        self,
        root: object,
        builder: Any,
        *,
        dag: object = None,
        **kwargs: Any,
    ) -> None:
        self.engine.encode(*self._payload(root, dag, **kwargs), builder)

    def encode_one(self, root: object, dag: object = None, **kwargs: Any) -> Any:
        return self.engine.encode(*self._payload(root, dag, **kwargs))

    def encode_batch(
        self,
        roots: object,
        *,
        dags: object = None,
        goals: object = None,
        subgoal_layers: object = None,
    ) -> Any:
        inputs = prepare_core_batch_inputs(
            roots, goals=goals, subgoal_layers=subgoal_layers
        )
        parsed_roots = parse_states_batch(inputs.states)
        normalized_dags = _normalize_dag_batch_data(dags)
        parsed_dags = (
            [None] * len(parsed_roots)
            if normalized_dags is None
            else parse_dags_batch_param(normalized_dags, state_count=len(parsed_roots))
        )
        for root, dag in zip(parsed_roots, parsed_dags, strict=True):
            if dag is not None and dag.root().get_index() != root.get_index():
                raise ValueError("dag root must match root state")
        return self.engine.encode_batch(
            parsed_roots,
            dags=parsed_dags,
            goals=inputs.goals,
            actions=inputs.actions,
            subgoal_layers=inputs.subgoal_layers,
            history_subgoals=inputs.history_subgoals,
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
        from .._core import HorizonStreamEncoder

        return _PymimirHorizonStream(self, HorizonStreamEncoder(self.engine))


class _PymimirHorizonStream:
    def __init__(self, runtime: PymimirHorizonRuntime, native: Any) -> None:
        self._runtime = runtime
        self._native = native

    def append(self, root: object, dag: object = None, **kwargs: Any) -> Any:
        return self._native.append(*self._runtime._payload(root, dag, **kwargs))

    def update(
        self,
        stream_id: int,
        root: object,
        dag: object = None,
        **kwargs: Any,
    ) -> None:
        self._native.update(stream_id, *self._runtime._payload(root, dag, **kwargs))

    def remove(self, stream_id: int) -> None:
        self._native.remove(stream_id)

    def set_reuse_removed(self, value: bool) -> None:
        self._native.set_reuse_removed(bool(value))

    def flush(self) -> Any:
        return self._native.flush()

    def reset(self) -> None:
        self._native.reset()


__all__ = ["PymimirHorizonRuntime"]
