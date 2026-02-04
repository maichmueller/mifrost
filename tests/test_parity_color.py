from __future__ import annotations

from typing import Any, Iterable, Tuple

import networkx as nx
import pytest

import mifrost
from tests.conftest import load_problem
from tests.ground_truth.pyencoding_ref.color_encoder import ColorGraphEncoder


SMALL_CASES = [
    ("blocks", "probBLOCKS-4-0"),
    ("gripper", "gripper_b-5"),
    ("delivery", "instance_2x2_p-2_0"),
]


def _canonical_nodes(graph: nx.Graph) -> list[tuple[str, tuple[tuple[str, Any], ...]]]:
    nodes = []
    for node, attrs in graph.nodes(data=True):
        nodes.append((str(node), tuple(sorted(attrs.items()))))
    return sorted(nodes)


def _canonical_edges(
    graph: nx.Graph,
) -> list[tuple[str, str, tuple[tuple[str, Any], ...]]]:
    edges = []
    if graph.is_multigraph():
        for u, v, _key, attrs in graph.edges(keys=True, data=True):
            u_label, v_label = str(u), str(v)
            if not graph.is_directed() and u_label > v_label:
                u_label, v_label = v_label, u_label
            edges.append((u_label, v_label, tuple(sorted(attrs.items()))))
    else:
        for u, v, attrs in graph.edges(data=True):
            u_label, v_label = str(u), str(v)
            if not graph.is_directed() and u_label > v_label:
                u_label, v_label = v_label, u_label
            edges.append((u_label, v_label, tuple(sorted(attrs.items()))))
    return sorted(edges)


@pytest.mark.parametrize(
    ("domain", "problem"),
    SMALL_CASES,
    ids=[f"{domain}:{problem}" for domain, problem in SMALL_CASES],
)
@pytest.mark.parametrize("edge_features", [False, True])
@pytest.mark.parametrize("predicate_nodes", [False, True])
def test_color_encoder_parity(
    domain: str,
    problem: str,
    edge_features: bool,
    predicate_nodes: bool,
):
    domain_obj, problem_obj, state, _domain_path, _problem_path = load_problem(
        domain, problem
    )
    goals = list(problem_obj.get_goal_condition().get_literals())

    ref_encoder = ColorGraphEncoder(
        domain_obj,
        edge_features=edge_features,
        enable_global_predicate_nodes=predicate_nodes,
    )
    ref_data = ref_encoder.encode(state, goals=goals)
    ref_graph = ref_encoder.to_networkx(ref_data)

    cpp_encoder = mifrost.ColorEncoder(
        domain_obj,
        edge_features=edge_features,
        enable_global_predicate_nodes=predicate_nodes,
    )
    cpp_data = cpp_encoder.encode(state, goals=goals)
    cpp_graph = cpp_encoder.to_networkx(cpp_data)

    assert _canonical_nodes(ref_graph) == _canonical_nodes(cpp_graph)
    assert _canonical_edges(ref_graph) == _canonical_edges(cpp_graph)
