from __future__ import annotations

from collections.abc import Iterable as IterableABC
from collections.abc import Sequence as SequenceABC
from typing import TYPE_CHECKING, Any, TypeAlias, Optional

from .._core import TransitionDAG
from .types import BatchParam

if TYPE_CHECKING:
    import rustworkx as rx
    from pymimir.pymimir.advanced.search import State
    from pymimir.pymimir.advanced.formalism import GroundAction

    RXStateDAG: TypeAlias = rx.PyDiGraph[State, Optional[GroundAction]]
else:
    RXStateDAG: TypeAlias = Any


def _load_rustworkx():
    try:
        import rustworkx as rx
    except ModuleNotFoundError:
        return None
    return rx


def _is_rustworkx_digraph(value: object) -> bool:
    rx = _load_rustworkx()
    return rx is not None and isinstance(value, rx.PyDiGraph)


def transition_dag_from_rustworkx(
    graph: RXStateDAG,
    *,
    fallback_missing_candidate_id_to_node_index: bool = False,
) -> TransitionDAG:
    return TransitionDAG.from_rustworkx(
        graph,
        fallback_missing_candidate_id_to_node_index=(
            fallback_missing_candidate_id_to_node_index
        ),
    )


def _normalize_dag_leaf(value: object) -> object:
    if value is None or isinstance(value, TransitionDAG):
        return value
    if _is_rustworkx_digraph(value):
        return transition_dag_from_rustworkx(value)
    return value


def _normalize_dag_batch_data(value: object) -> object:
    if value is None:
        return None
    if isinstance(value, BatchParam):
        if value.kind == "none":
            return BatchParam.none()
        if value.kind == "shared":
            return BatchParam.shared(_normalize_dag_leaf(value.value))
        if value.kind == "separate":
            if not isinstance(value.value, SequenceABC) or isinstance(
                value.value, (str, bytes, bytearray)
            ):
                raise TypeError("BatchParam(separate) value must be a sequence")
            return BatchParam.separate(
                None if entry is None else _normalize_dag_leaf(entry)
                for entry in value.value
            )
        raise ValueError("BatchParam.kind must be 'shared', 'separate', or 'none'")

    normalized_leaf = _normalize_dag_leaf(value)
    if normalized_leaf is not value:
        return normalized_leaf

    if isinstance(value, (str, bytes, bytearray)):
        return value
    if isinstance(value, tuple):
        return tuple(_normalize_dag_batch_data(item) for item in value)
    if isinstance(value, list):
        return [_normalize_dag_batch_data(item) for item in value]
    if isinstance(value, SequenceABC):
        return [_normalize_dag_batch_data(item) for item in value]
    if isinstance(value, IterableABC):
        return (_normalize_dag_batch_data(item) for item in value)
    return value
