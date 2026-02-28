from __future__ import annotations

from collections.abc import Iterable as IterableABC
from collections.abc import Sequence as SequenceABC
from typing import TYPE_CHECKING, Any, TypeAlias, Optional

from .._core import TransitionDAG
from .types import BatchParam, to_advanced_action, to_advanced_state
from pymimir.pymimir.advanced.search import State
from pymimir.pymimir.advanced.formalism import GroundAction

if TYPE_CHECKING:
    import rustworkx as rx

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


def transition_dag_from_rustworkx(graph: RXStateDAG) -> TransitionDAG:
    if not _is_rustworkx_digraph(graph):
        raise TypeError(
            "transition_dag_from_rustworkx expects a rustworkx.PyDiGraph, "
            f"got {type(graph)!r}"
        )

    node_indices = graph.node_indices()
    if not node_indices:
        raise ValueError("rustworkx PyDiGraph must not be empty")

    root_candidates = [
        node_idx for node_idx in node_indices if graph.in_degree(node_idx) == 0
    ]
    if len(root_candidates) != 1:
        raise ValueError("rustworkx PyDiGraph must contain exactly one root node")
    root_idx = root_candidates[0]

    node_states: dict[int, State] = {}

    def _state_for(node_idx: int) -> State:
        if node_idx in node_states:
            return node_states[node_idx]
        try:
            state = to_advanced_state(graph.get_node_data(node_idx))
        except TypeError as exc:
            raise TypeError(
                f"rustworkx PyDiGraph node payload at index {node_idx} must be a state input"
            ) from exc
        node_states[node_idx] = state
        return state

    available_nodes = {root_idx}
    pending_edges = graph.weighted_edge_list()
    records: list[tuple[State, State, Optional[GroundAction]]] = []

    while pending_edges:
        next_pending = []
        progressed = False

        for src_idx, dst_idx, payload in pending_edges:
            if src_idx not in available_nodes:
                next_pending.append((src_idx, dst_idx, payload))
                continue

            try:
                action = None if payload is None else to_advanced_action(payload)
            except TypeError as exc:
                raise TypeError(
                    "rustworkx PyDiGraph edge payload at "
                    f"({src_idx}, {dst_idx}) must be an action input or None"
                ) from exc

            records.append((_state_for(src_idx), _state_for(dst_idx), action))
            available_nodes.add(dst_idx)
            progressed = True

        if not progressed:
            raise ValueError(
                "rustworkx PyDiGraph could not be imported into TransitionDAG; "
                "ensure it is a single rooted DAG/tree"
            )

        pending_edges = next_pending
    dag = TransitionDAG(_state_for(root_idx))
    dag.register_transitions(records)
    return dag


def _normalize_dag_leaf(value: object) -> object:
    if value is None or isinstance(value, TransitionDAG):
        return value
    if _is_rustworkx_digraph(value):
        return transition_dag_from_rustworkx(value)
    return value


def _normalize_dag_batch_payload(value: object) -> object:
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
        return tuple(_normalize_dag_batch_payload(item) for item in value)
    if isinstance(value, list):
        return [_normalize_dag_batch_payload(item) for item in value]
    if isinstance(value, SequenceABC):
        return [_normalize_dag_batch_payload(item) for item in value]
    if isinstance(value, IterableABC):
        return (_normalize_dag_batch_payload(item) for item in value)
    return value
