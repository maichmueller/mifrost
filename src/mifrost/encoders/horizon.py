from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from torch_geometric.data import HeteroData

from .._core import (
    BatchBuilder,
    GoalInputs,
    HorizonEncoderConfig,
    HorizonHGraphEncoderEngine,
    TransitionDAG,
)
from .base import EncoderBase, StreamEncoderBase
from .common import _advanced_domain, _advanced_state, _parts_to_pyg, _split_goals


def _ensure_dag(root: Any, dag: TransitionDAG | None) -> TransitionDAG:
    if dag is not None:
        return dag
    adv_root = _advanced_state(root)
    return TransitionDAG(adv_root)


def _prepare_horizon_goals(
    root: Any,
    goals: Iterable[Any] | None,
    subgoal_layers: Iterable[Iterable[Any]] | None,
) -> GoalInputs:
    if goals is None:
        if hasattr(root, "get_problem"):
            goals = list(root.get_problem().get_goal_condition().get_literals())
        else:
            raise ValueError("goals must be provided when passing an advanced state")
    inputs, _ = _split_goals(goals, subgoal_layers)
    return inputs


def _is_literal(value: Any) -> bool:
    if hasattr(value, "_advanced_ground_literal"):
        return True
    try:
        import pymimir.advanced.formalism as af

        return isinstance(
            value,
            (af.StaticGroundLiteral, af.FluentGroundLiteral, af.DerivedGroundLiteral),
        )
    except Exception:
        return False


@dataclass
class HorizonEncoderStream(StreamEncoderBase[HeteroData]):
    _engine: HorizonHGraphEncoderEngine

    def __post_init__(self) -> None:
        self._reset_builder()

    def append(
        self,
        root: Any,
        dag: TransitionDAG | None = None,
        *,
        goals: Iterable[Any] | None = None,
        subgoal_layers: Iterable[Iterable[Any]] | None = None,
    ) -> None:
        adv_root = _advanced_state(root)
        dag = _ensure_dag(root, dag)
        inputs = _prepare_horizon_goals(root, goals, subgoal_layers)
        self._engine.encode(adv_root, dag, inputs, self._builder)
        if hasattr(self._builder, "next_graph"):
            self._builder.next_graph()

    def _reset_builder(self) -> None:
        self._builder = BatchBuilder()
        self._builder.set_graph_kind("hetero")

    def _parts_to_pyg(
        self,
        parts: Mapping[str, Any],
        *,
        as_batch: bool,
        include_metadata: bool = True,
    ) -> HeteroData:
        return _parts_to_pyg(
            parts, as_batch=as_batch, include_metadata=include_metadata
        )


class HorizonEncoder(EncoderBase[HeteroData]):
    def __init__(
        self,
        domain: Any,
        *,
        transition_mode: Any | None = None,
        target_symbol_prefix: str | None = None,
        parent_relation: str | None = None,
        sibling_relation: str | None = None,
        cousin_relation: str | None = None,
        enable_parent_relation: bool | None = None,
        enable_sibling_relation: bool | None = None,
        enable_cousin_relation: bool | None = None,
        exclude_root_candidate: bool | None = None,
        max_goal_level: int | None = None,
    ) -> None:
        config = HorizonEncoderConfig()
        if transition_mode is not None:
            config.transition_mode = transition_mode
        if target_symbol_prefix is not None:
            config.target_symbol_prefix = target_symbol_prefix
        if parent_relation is not None:
            config.parent_relation = parent_relation
        if sibling_relation is not None:
            config.sibling_relation = sibling_relation
        if cousin_relation is not None:
            config.cousin_relation = cousin_relation
        if enable_parent_relation is not None:
            config.enable_parent_relation = enable_parent_relation
        if enable_sibling_relation is not None:
            config.enable_sibling_relation = enable_sibling_relation
        if enable_cousin_relation is not None:
            config.enable_cousin_relation = enable_cousin_relation
        if exclude_root_candidate is not None:
            config.exclude_root_candidate = exclude_root_candidate
        if max_goal_level is not None:
            config.max_goal_level = max_goal_level
        self._engine = HorizonHGraphEncoderEngine(_advanced_domain(domain), config)

    @property
    def engine(self) -> HorizonHGraphEncoderEngine:
        return self._engine

    def encode_parts(
        self,
        root: Any,
        dag: TransitionDAG | None = None,
        *,
        goals: Iterable[Any] | None = None,
        subgoal_layers: Iterable[Iterable[Any]] | None = None,
    ) -> Mapping[str, Any]:
        adv_root = _advanced_state(root)
        dag = _ensure_dag(root, dag)
        inputs = _prepare_horizon_goals(root, goals, subgoal_layers)
        return self._engine.encode(adv_root, dag, inputs)

    def encode(
        self,
        root: Any,
        dag: TransitionDAG | None = None,
        *,
        goals: Iterable[Any] | None = None,
        subgoal_layers: Iterable[Iterable[Any]] | None = None,
        include_metadata: bool = True,
        **kwargs: Any,
    ) -> HeteroData:
        return super().encode(
            root,
            goals=goals,
            subgoal_layers=subgoal_layers,
            dag=dag,
            include_metadata=include_metadata,
            **kwargs,
        )

    def _encode_batch_parts(
        self,
        roots: Iterable[Any] | Any,
        dags: Iterable[TransitionDAG] | TransitionDAG | None = None,
        *,
        goals: Iterable[Any] | None = None,
        subgoal_layers: Iterable[Iterable[Any]] | None = None,
    ) -> Mapping[str, Any]:
        is_state_like = hasattr(roots, "get_problem") or hasattr(
            roots, "_advanced_state"
        )
        if is_state_like:
            root_list = [roots]
        else:
            if isinstance(roots, (str, bytes)):
                raise TypeError("encode_batch expects a state or an iterable of states")
            root_list = list(roots)

        if dags is None:
            dag_list = [None] * len(root_list)
        elif isinstance(dags, TransitionDAG):
            dag_list = [dags]
        else:
            dag_list = list(dags)
        if len(dag_list) != len(root_list):
            raise ValueError("dags length must match roots length")

        goals_per_state: list[Iterable[Any]] | None = None
        if (
            goals is not None
            and isinstance(goals, Sequence)
            and len(goals) == len(root_list)
        ):
            first = goals[0] if goals else None
            if first is not None and not _is_literal(first):
                goals_per_state = list(goals)

        shared_inputs: GoalInputs | None = None
        if goals is not None and goals_per_state is None:
            shared_inputs, _ = _split_goals(goals, subgoal_layers)

        builder = BatchBuilder()
        builder.set_graph_kind("hetero")
        for idx, root in enumerate(root_list):
            adv_root = _advanced_state(root)
            dag = _ensure_dag(root, dag_list[idx])
            if goals_per_state is not None:
                inputs, _ = _split_goals(goals_per_state[idx], subgoal_layers)
            else:
                inputs = (
                    shared_inputs
                    if shared_inputs is not None
                    else _prepare_horizon_goals(root, None, subgoal_layers)
                )
            self._engine.encode(adv_root, dag, inputs, builder)
            if hasattr(builder, "next_graph"):
                builder.next_graph()
        return builder.build_parts()

    def encode_batch(
        self,
        roots: Iterable[Any] | Any,
        dags: Iterable[TransitionDAG] | TransitionDAG | None = None,
        *,
        goals: Iterable[Any] | None = None,
        subgoal_layers: Iterable[Iterable[Any]] | None = None,
        include_metadata: bool = True,
        **kwargs: Any,
    ) -> HeteroData:
        return super().encode_batch(
            roots,
            goals=goals,
            subgoal_layers=subgoal_layers,
            dags=dags,
            include_metadata=include_metadata,
            **kwargs,
        )

    def _accepted_kwargs(self) -> set[str]:
        return {"dag", "dags"}

    def _parts_to_pyg(
        self,
        parts: Mapping[str, Any],
        *,
        as_batch: bool,
        include_metadata: bool = True,
    ) -> HeteroData:
        return _parts_to_pyg(
            parts, as_batch=as_batch, include_metadata=include_metadata
        )

    def encode_batch_parts(
        self,
        roots: Iterable[Any] | Any,
        dags: Iterable[TransitionDAG] | TransitionDAG | None = None,
        *,
        goals: Iterable[Any] | None = None,
        subgoal_layers: Iterable[Iterable[Any]] | None = None,
    ) -> Mapping[str, Any]:
        return self._encode_batch_parts(
            roots, dags, goals=goals, subgoal_layers=subgoal_layers
        )

    def stream(self) -> HorizonEncoderStream:
        return HorizonEncoderStream(self._engine)


__all__ = ["HorizonEncoder", "HorizonEncoderStream"]
