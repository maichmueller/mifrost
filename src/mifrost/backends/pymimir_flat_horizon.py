"""Pymimir runtime for the public Flat Horizon encoder."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Literal, cast

from ..encoders._batch_contract import (
    parse_dags_batch_param,
    parse_states_batch,
    prepare_core_batch_inputs,
)
from ..encoders._flat_validation import validate_subgoal_layers_batch_payload
from ..encoders._rustworkx_dag import _normalize_dag_batch_data
from .pymimir_common import _advanced_domain, _advanced_state
from .pymimir_lane_specs import ensure_transition_dag, prepare_goal_inputs


def _enum_name(value: object) -> str:
    return str(value).rsplit(".", 1)[-1]


class PymimirFlatHorizonRuntime:
    """Preserve the historical native Pymimir Flat Horizon fast path."""

    backend_name: Literal["pymimir"] = "pymimir"

    def __init__(self, domain: object, config: Any) -> None:
        from .. import _core

        config_type = getattr(_core, "FlatHorizonEncoderConfig")
        mode_type = getattr(_core, "FlatHorizonEncoderMode")
        native_config = config_type(
            max_goal_level=int(config.max_goal_level),
            support_literals=bool(config.support_literals),
            include_static=bool(config.include_static),
            export_node_names=bool(config.export_node_names),
            ignore_zero_arity_relations=bool(config.ignore_zero_arity_relations),
            ignore_actions=bool(config.ignore_actions),
            use_predicate_virtual_nodes=bool(config.use_predicate_virtual_nodes),
            include_lgan_edges=bool(config.include_lgan_edges),
            transition_mode=getattr(mode_type, _enum_name(config.transition_mode)),
            target_symbol_prefix=str(config.target_symbol_prefix),
            parent_relation=str(config.parent_relation),
            sibling_relation=str(config.sibling_relation),
            cousin_relation=str(config.cousin_relation),
            lgan_tn_edge_pos=str(config.lgan_tn_edge_pos),
            lgan_nn_edge_pos=str(config.lgan_nn_edge_pos),
            lgan_rr_edge_pos=str(config.lgan_rr_edge_pos),
            enable_parent_relation=bool(config.enable_parent_relation),
            enable_sibling_relation=bool(config.enable_sibling_relation),
            enable_cousin_relation=bool(config.enable_cousin_relation),
            root_policy=config.root_policy,
            pack_relation_args_relation_major=bool(
                config.pack_relation_args_relation_major
            ),
            goal_derivations=set(config.goal_derivations),
        )
        engine_type = getattr(_core, "FlatHorizonEncoderEngine")
        self.engine = engine_type(_advanced_domain(domain), native_config)

    @property
    def relation_dict(self) -> Any:
        return self.engine.relation_dict

    @staticmethod
    def _payload(
        root: object,
        dag: object,
        *,
        goals: object = None,
        subgoal_layers: object = None,
    ) -> tuple[Any, Any, Any]:
        return (
            _advanced_state(root),
            ensure_transition_dag(root, cast(Any, dag)),
            prepare_goal_inputs(
                root,
                cast(Iterable[Any] | None, goals),
                cast(Iterable[Iterable[Any]] | None, subgoal_layers),
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

    def encode(self, root: object, dag: object = None, **kwargs: Any) -> Any:
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
        parsed_roots = cast(list[Any], parse_states_batch(inputs.states))
        validate_subgoal_layers_batch_payload(
            subgoal_layers,
            state_count=len(parsed_roots),
            max_goal_level=int(self.engine.config.max_goal_level),
        )
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
            actions=None,
            subgoal_layers=inputs.subgoal_layers,
            history_subgoals=None,
            history_max_steps=None,
        )

    def make_stream(self) -> Any:
        from .. import _core

        return _PymimirFlatHorizonStream(
            self, getattr(_core, "FlatHorizonStreamEncoder")(self.engine)
        )


class _PymimirFlatHorizonStream:
    def __init__(self, runtime: PymimirFlatHorizonRuntime, native: Any) -> None:
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


__all__ = ["PymimirFlatHorizonRuntime"]
