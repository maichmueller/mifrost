from __future__ import annotations

from collections.abc import Iterable as IterableABC
from collections.abc import Mapping as MappingABC
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


def _mapping_or_attr(payload: object, key: str) -> tuple[object | None, bool]:
    if isinstance(payload, MappingABC) and key in payload:
        return payload[key], True
    if hasattr(payload, key):
        return getattr(payload, key), True
    return None, False


def _resolve_node_state_payload(payload: object) -> object:
    state_payload, has_state = _mapping_or_attr(payload, "state")
    return state_payload if has_state else payload


def _resolve_node_candidate_id_payload(payload: object) -> tuple[object | None, bool]:
    return _mapping_or_attr(payload, "candidate_id")


def _parse_candidate_id(
    raw_candidate_id: object,
    *,
    node_idx: int,
) -> int:
    if isinstance(raw_candidate_id, bool) or not isinstance(raw_candidate_id, int):
        raise TypeError(
            "rustworkx PyDiGraph node payload candidate_id at "
            f"index {node_idx} must be an int or None"
        )
    return int(raw_candidate_id)


def transition_dag_from_rustworkx(
    graph: RXStateDAG,
    *,
    fallback_missing_candidate_id_to_node_index: bool = False,
) -> TransitionDAG:
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
    node_candidate_ids: dict[int, int | None] = {}

    for node_idx in node_indices:
        payload = graph.get_node_data(node_idx)

        try:
            state = to_advanced_state(_resolve_node_state_payload(payload))
        except TypeError as exc:
            raise TypeError(
                f"rustworkx PyDiGraph node payload at index {node_idx} must be a state input"
            ) from exc
        node_states[node_idx] = state

        raw_candidate_id, has_candidate_id = _resolve_node_candidate_id_payload(payload)
        if not has_candidate_id or raw_candidate_id is None:
            node_candidate_ids[node_idx] = None
        else:
            node_candidate_ids[node_idx] = _parse_candidate_id(
                raw_candidate_id, node_idx=node_idx
            )

    non_root_node_indices = [idx for idx in node_indices if idx != root_idx]
    explicit_candidate_id_nodes = [
        idx for idx in non_root_node_indices if node_candidate_ids[idx] is not None
    ]
    if (
        explicit_candidate_id_nodes
        and len(explicit_candidate_id_nodes) != len(non_root_node_indices)
        and not fallback_missing_candidate_id_to_node_index
    ):
        missing_idx = next(
            idx for idx in non_root_node_indices if node_candidate_ids[idx] is None
        )
        raise ValueError(
            "rustworkx PyDiGraph has partial candidate_id coverage; "
            f"missing candidate_id for node index {missing_idx}"
        )

    resolved_candidate_ids: dict[int, int | None] = {}
    for node_idx in non_root_node_indices:
        candidate_id = node_candidate_ids[node_idx]
        if candidate_id is None and fallback_missing_candidate_id_to_node_index:
            candidate_id = int(node_idx)
        resolved_candidate_ids[node_idx] = candidate_id

    seen_candidate_ids: set[int] = set()
    for node_idx in non_root_node_indices:
        candidate_id = resolved_candidate_ids[node_idx]
        if candidate_id is None:
            continue
        if candidate_id in seen_candidate_ids:
            raise ValueError(
                "rustworkx PyDiGraph has duplicate candidate_id "
                f"{candidate_id} for non-root nodes"
            )
        seen_candidate_ids.add(candidate_id)

    available_nodes = {root_idx}
    pending_edges = graph.weighted_edge_list()
    records: list[tuple[State, State, Optional[GroundAction], Optional[int]]] = []

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

            records.append(
                (
                    node_states[src_idx],
                    node_states[dst_idx],
                    action,
                    resolved_candidate_ids.get(dst_idx),
                )
            )
            available_nodes.add(dst_idx)
            progressed = True

        if not progressed:
            raise ValueError(
                "rustworkx PyDiGraph could not be imported into TransitionDAG; "
                "ensure it is a single rooted DAG/tree"
            )

        pending_edges = next_pending
    dag = TransitionDAG(node_states[root_idx])
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
