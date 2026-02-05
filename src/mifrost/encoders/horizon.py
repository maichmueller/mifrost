from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

from torch_geometric.data import HeteroData

from .._core import (
    BatchBuilder,
    GoalInputs,
    HorizonEncoderConfig,
    HorizonHGraphEncoderEngine,
    TransitionDAG,
)
from .base import (
    EncoderBase,
    GoalBatchInput,
    StateBatchInput,
    StreamEncoderBase,
    SubgoalLayersInput,
)
from .common import _advanced_domain, _advanced_state, _parts_to_pyg, _split_goals
from .types import (
    DomainInput,
    GoalLiteralInput,
    StateInput,
    default_goals_from_state,
    is_goal_literal_input,
    is_state_input,
)


def _ensure_dag(root: StateInput, dag: TransitionDAG | None) -> TransitionDAG:
    """Return an explicit DAG or create a default single-root DAG."""
    if dag is not None:
        return dag
    adv_root = _advanced_state(root)
    return TransitionDAG(adv_root)


def _prepare_horizon_goals(
    root: StateInput,
    goals: GoalBatchInput,
    subgoal_layers: SubgoalLayersInput,
) -> GoalInputs:
    """Resolve user-provided or problem-default goals into ``GoalInputs``."""
    if goals is None:
        goals = default_goals_from_state(root)
    inputs, _ = _split_goals(goals, subgoal_layers)
    return inputs


def _is_literal(value: object) -> bool:
    """Best-effort literal type probe for per-state goal detection."""
    return is_goal_literal_input(value)


@dataclass
class HorizonEncoderStream(StreamEncoderBase[HeteroData]):
    """Streaming wrapper for ``HorizonHGraphEncoderEngine``."""

    _engine: HorizonHGraphEncoderEngine

    def __post_init__(self) -> None:
        """Initialize an empty hetero builder for streaming."""
        self._reset_builder()

    def append(
        self,
        root: StateInput,
        dag: TransitionDAG | None = None,
        *,
        goals: Iterable[GoalLiteralInput] | None = None,
        subgoal_layers: Iterable[Iterable[GoalLiteralInput]] | None = None,
    ) -> None:
        """Append one root/DAG encoding to the stream."""
        adv_root = _advanced_state(root)
        dag = _ensure_dag(root, dag)
        inputs = _prepare_horizon_goals(root, goals, subgoal_layers)
        self._engine.encode(adv_root, dag, inputs, self._builder)
        self._builder.next_graph()

    def _reset_builder(self) -> None:
        """Reset stream accumulation state."""
        self._builder = BatchBuilder()
        self._builder.set_graph_kind("hetero")

    def _parts_to_pyg(
        self,
        parts: Mapping[str, object],
        *,
        as_batch: bool,
        include_metadata: bool = True,
    ) -> HeteroData:
        return _parts_to_pyg(
            parts, as_batch=as_batch, include_metadata=include_metadata
        )


class HorizonEncoder(EncoderBase[HeteroData]):
    """
    Horizon lookahead encoder backed by ``HorizonHGraphEncoderEngine``.

    This encoder combines a root state, a ``TransitionDAG`` and goals into one
    hetero graph representation.
    """

    def __init__(
        self,
        domain: DomainInput,
        *,
        transition_mode: object | None = None,
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
        """Create a horizon encoder for one domain."""
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
        """Expose the underlying C++ horizon engine."""
        return self._engine

    def encode_parts(
        self,
        root: StateInput,
        dag: TransitionDAG | None = None,
        *,
        goals: GoalBatchInput = None,
        subgoal_layers: SubgoalLayersInput = None,
    ) -> Mapping[str, object]:
        """Encode one root/DAG pair into parts."""
        adv_root = _advanced_state(root)
        dag = _ensure_dag(root, dag)
        inputs = _prepare_horizon_goals(root, goals, subgoal_layers)
        return self._engine.encode(adv_root, dag, inputs)

    def encode(
        self,
        root: StateInput,
        dag: TransitionDAG | None = None,
        *,
        goals: GoalBatchInput = None,
        subgoal_layers: SubgoalLayersInput = None,
        include_metadata: bool = True,
        **kwargs: object,
    ) -> HeteroData:
        """Encode one root/DAG pair into ``HeteroData``."""
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
        roots: StateBatchInput,
        dags: Iterable[TransitionDAG] | TransitionDAG | None = None,
        *,
        goals: GoalBatchInput | Sequence[Iterable[GoalLiteralInput]] = None,
        subgoal_layers: SubgoalLayersInput = None,
    ) -> Mapping[str, object]:
        """Internal batch implementation shared by public batch APIs."""
        if is_state_input(roots):
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

        goals_per_state: list[Iterable[GoalLiteralInput]] | None = None
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
            builder.next_graph()
        return builder.build_parts()

    def encode_batch(
        self,
        roots: StateBatchInput,
        dags: Iterable[TransitionDAG] | TransitionDAG | None = None,
        *,
        goals: GoalBatchInput | Sequence[Iterable[GoalLiteralInput]] = None,
        subgoal_layers: SubgoalLayersInput = None,
        include_metadata: bool = True,
        **kwargs: object,
    ) -> HeteroData:
        """Encode one or many root/DAG pairs into batched ``HeteroData``."""
        return super().encode_batch(
            roots,
            goals=goals,
            subgoal_layers=subgoal_layers,
            dags=dags,
            include_metadata=include_metadata,
            **kwargs,
        )

    def _accepted_kwargs(self) -> set[str]:
        """Accept transition DAG kwargs in the generic base API."""
        return {"dag", "dags"}

    def _parts_to_pyg(
        self,
        parts: Mapping[str, object],
        *,
        as_batch: bool,
        include_metadata: bool = True,
    ) -> HeteroData:
        return _parts_to_pyg(
            parts, as_batch=as_batch, include_metadata=include_metadata
        )

    def encode_batch_parts(
        self,
        roots: StateBatchInput,
        dags: Iterable[TransitionDAG] | TransitionDAG | None = None,
        *,
        goals: GoalBatchInput | Sequence[Iterable[GoalLiteralInput]] = None,
        subgoal_layers: SubgoalLayersInput = None,
    ) -> Mapping[str, object]:
        """Encode one or many root/DAG pairs into batch parts."""
        return self._encode_batch_parts(
            roots, dags, goals=goals, subgoal_layers=subgoal_layers
        )

    def stream(self) -> HorizonEncoderStream:
        """Create a streaming encoder sharing this encoder's C++ engine."""
        return HorizonEncoderStream(self._engine)


__all__ = ["HorizonEncoder", "HorizonEncoderStream"]
