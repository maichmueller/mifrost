from __future__ import annotations

import json
import platform
import sys
from pathlib import Path
from typing import Any

import pymimir


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _as_advanced_state(state: Any) -> Any:
    return getattr(state, "_advanced_state", state)


def _as_advanced_action(action: Any) -> Any:
    return getattr(action, "_advanced_ground_action", action)


def load_problem(domain: str = "blocks", problem: str = "smedium"):
    root = repo_root()
    domain_path = root / "data" / "pddl" / domain / "domain.pddl"
    problem_path = root / "data" / "pddl" / domain / f"{problem}.pddl"

    dom = pymimir.Domain(domain_path)
    prob = pymimir.Problem(dom, problem_path, mode="lifted")
    state = prob.get_initial_state()

    context = prob._search_context
    space, _ = pymimir.advanced.datasets.StateSpace.create(
        context, pymimir.advanced.datasets.StateSpaceOptions()
    )
    space = pymimir.wrapper_datasets.StateSpaceSampler(
        pymimir.advanced.datasets.StateSpaceSampler(space), prob
    )
    return space, dom, prob, state


def first_successor(space: Any, state: Any):
    transitions = list(space.get_forward_transitions(state))
    for action, target in transitions:
        if target is None:
            continue
        try:
            if target.get_index() == state.get_index():
                continue
        except Exception:
            pass
        return action, target
    raise RuntimeError("No non-trivial successor transition available")


def to_py_list(value: Any, *, max_items: int = 10) -> list[Any]:
    if value is None:
        return []
    if hasattr(value, "tolist") and callable(value.tolist):
        out = value.tolist()
        if isinstance(out, list):
            return out[:max_items]
    try:
        out = list(value)
        return out[:max_items]
    except TypeError:
        return [value]


def print_encoding_summary(encoding: Any, *, label: str) -> None:
    node_types = [str(t) for t in getattr(encoding, "node_types", [])]
    edge_types_raw = getattr(encoding, "edge_types", [])
    edge_types = ["|".join(map(str, et)) for et in edge_types_raw]
    node_types_sorted = sorted(node_types)
    edge_types_sorted = sorted(edge_types)

    fp = (
        encoding.schema_fingerprint()
        if hasattr(encoding, "schema_fingerprint")
        else None
    )

    summary = {
        "label": label,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "graph_kind": getattr(encoding, "graph_kind", None),
        "num_graphs": int(getattr(encoding, "num_graphs", 0)),
        "num_nodes": int(getattr(encoding, "num_nodes", 0)),
        "num_edges": int(getattr(encoding, "num_edges", 0)),
        "schema_fingerprint": int(fp) if fp is not None else None,
        "node_types_count": len(node_types_sorted),
        "edge_types_count": len(edge_types_sorted),
        "node_types_head": node_types_sorted[:10],
        "edge_types_head": edge_types_sorted[:10],
    }

    print(f"== {label} ==")
    print(f"python: {summary['python']}")
    print(f"platform: {summary['platform']}")
    print(f"graph_kind: {summary['graph_kind']}")
    print(f"num_graphs: {summary['num_graphs']}")
    print(f"num_nodes: {summary['num_nodes']}")
    print(f"num_edges: {summary['num_edges']}")
    print(f"schema_fingerprint: {summary['schema_fingerprint']}")
    print(f"node_types_count: {summary['node_types_count']}")
    print(f"edge_types_count: {summary['edge_types_count']}")
    print(f"node_types_head: {summary['node_types_head']}")
    print(f"edge_types_head: {summary['edge_types_head']}")

    print("SNAPSHOT_JSON_BEGIN")
    print(json.dumps(summary, sort_keys=True, indent=2))
    print("SNAPSHOT_JSON_END")


__all__ = [
    "load_problem",
    "first_successor",
    "print_encoding_summary",
    "to_py_list",
    "_as_advanced_state",
    "_as_advanced_action",
]
