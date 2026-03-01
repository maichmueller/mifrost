from __future__ import annotations

import ast
from pathlib import Path

import pytest

import mifrost
from mifrost.encoders import transition_dag_from_rustworkx

from .test_utils import adv_action, adv_state

rx = pytest.importorskip("rustworkx")


def _first_distinct_changed_transitions(space, root, count: int = 2):
    transitions = []
    seen_targets: set[str] = set()
    root_repr = str(adv_state(root))
    for action, target in space.get_forward_transitions(root):
        if action is None or target is None:
            continue
        target_repr = str(adv_state(target))
        if target_repr == root_repr or target_repr in seen_targets:
            continue
        seen_targets.add(target_repr)
        transitions.append((action, target))
        if len(transitions) >= count:
            return transitions
    pytest.skip("Fixture does not provide enough distinct changed transitions.")


def _assert_same_dag(left: mifrost.TransitionDAG, right: mifrost.TransitionDAG) -> None:
    assert left.root() == right.root()
    assert left.transitions() == right.transitions()

    left_nodes = left.nodes()
    right_nodes = right.nodes()
    assert len(left_nodes) == len(right_nodes)
    for left_node, right_node in zip(left_nodes, right_nodes):
        assert left_node.state == right_node.state
        assert left_node.index == right_node.index
        assert left_node.depth == right_node.depth
        assert left_node.action == right_node.action
        assert left_node.candidate_id == right_node.candidate_id


def test_transition_dag_from_rustworkx_is_reexported():
    assert mifrost.transition_dag_from_rustworkx is transition_dag_from_rustworkx


def test_rustworkx_dag_helper_uses_lazy_runtime_import():
    import mifrost.encoders._rustworkx_dag as helper

    tree = ast.parse(Path(helper.__file__).read_text())
    for node in tree.body:
        if isinstance(node, ast.Import):
            assert all(alias.name != "rustworkx" for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert node.module != "rustworkx"


def test_transition_dag_from_rustworkx_single_node(small_blocks):
    _space, _domain, problem = small_blocks
    root = problem.get_initial_state()

    graph = rx.PyDiGraph()
    graph.add_node(root)

    dag = transition_dag_from_rustworkx(graph)
    assert dag.root() == adv_state(root)
    assert dag.transitions() == []
    assert len(dag.nodes()) == 1


def test_transition_dag_from_rustworkx_one_edge_matches_manual_dag(small_blocks):
    space, _domain, problem = small_blocks
    root = problem.get_initial_state()
    (action, successor), _ = _first_distinct_changed_transitions(space, root, count=2)

    graph = rx.PyDiGraph()
    root_idx = graph.add_node(root)
    succ_idx = graph.add_node(successor)
    graph.add_edge(root_idx, succ_idx, action)

    converted = transition_dag_from_rustworkx(graph)

    manual = mifrost.TransitionDAG(adv_state(root))
    manual.register_transition(
        adv_state(root), adv_state(successor), adv_action(action)
    )

    _assert_same_dag(converted, manual)


def test_transition_dag_from_rustworkx_multi_level_chain_matches_manual_dag(
    small_blocks,
):
    space, _domain, problem = small_blocks
    root = problem.get_initial_state()
    (action0, succ0), (action1, succ1) = _first_distinct_changed_transitions(
        space, root, count=2
    )

    graph = rx.PyDiGraph()
    root_idx = graph.add_node(root)
    succ0_idx = graph.add_node(succ0)
    succ1_idx = graph.add_node(succ1)
    graph.add_edge(root_idx, succ0_idx, action0)
    graph.add_edge(succ0_idx, succ1_idx, action1)

    converted = transition_dag_from_rustworkx(graph)

    manual = mifrost.TransitionDAG(adv_state(root))
    manual.register_transition(adv_state(root), adv_state(succ0), adv_action(action0))
    manual.register_transition(adv_state(succ0), adv_state(succ1), adv_action(action1))

    _assert_same_dag(converted, manual)


def test_transition_dag_from_rustworkx_imports_reachable_multiple_parent_child(
    small_blocks,
):
    space, _domain, problem = small_blocks
    root = problem.get_initial_state()
    (action0, succ0), (action1, succ1) = _first_distinct_changed_transitions(
        space, root, count=2
    )

    graph = rx.PyDiGraph()
    root_idx = graph.add_node(root)
    succ0_idx = graph.add_node(succ0)
    succ1_idx = graph.add_node(succ1)
    graph.add_edge(root_idx, succ0_idx, action0)
    graph.add_edge(root_idx, succ1_idx, action1)
    graph.add_edge(succ1_idx, succ0_idx, action0)

    dag = transition_dag_from_rustworkx(graph)

    assert len(dag.transitions()) == 3
    assert dag.action(dag.index(adv_state(succ0))) == adv_action(action0)


def test_transition_dag_from_rustworkx_rejects_empty_graph():
    with pytest.raises(ValueError, match="must not be empty"):
        transition_dag_from_rustworkx(rx.PyDiGraph())


def test_transition_dag_from_rustworkx_requires_single_in_degree_zero_root(
    small_blocks,
):
    space, _domain, problem = small_blocks
    root = problem.get_initial_state()
    (_action, succ0), _ = _first_distinct_changed_transitions(space, root, count=2)

    graph = rx.PyDiGraph()
    graph.add_node(root)
    graph.add_node(succ0)

    with pytest.raises(ValueError, match="exactly one root node"):
        transition_dag_from_rustworkx(graph)


def test_transition_dag_from_rustworkx_rejects_edges_not_reachable_from_root(
    small_blocks,
):
    space, _domain, problem = small_blocks
    root = problem.get_initial_state()
    (action0, succ0), (action1, succ1) = _first_distinct_changed_transitions(
        space, root, count=2
    )

    graph = rx.PyDiGraph(check_cycle=False)
    root_idx = graph.add_node(root)
    succ0_idx = graph.add_node(succ0)
    succ1_idx = graph.add_node(succ1)
    # Reachable branch from the unique root.
    graph.add_edge(root_idx, succ0_idx, action0)
    # Disconnected component (no path from root), intentionally invalid for current import rule.
    graph.add_edge(succ1_idx, succ1_idx, action1)

    with pytest.raises(ValueError, match="could not be imported into TransitionDAG"):
        transition_dag_from_rustworkx(graph)


def test_transition_dag_from_rustworkx_rejects_invalid_node_payload(small_blocks):
    space, _domain, problem = small_blocks
    root = problem.get_initial_state()
    (_action, succ0), _ = _first_distinct_changed_transitions(space, root, count=2)

    graph = rx.PyDiGraph()
    root_idx = graph.add_node(root)
    bad_idx = graph.add_node(object())
    graph.add_edge(root_idx, bad_idx, None)

    with pytest.raises(TypeError, match="node payload at index"):
        transition_dag_from_rustworkx(graph)


def test_transition_dag_from_rustworkx_rejects_invalid_edge_payload(small_blocks):
    space, _domain, problem = small_blocks
    root = problem.get_initial_state()
    (_action, succ0), _ = _first_distinct_changed_transitions(space, root, count=2)

    graph = rx.PyDiGraph()
    root_idx = graph.add_node(root)
    succ_idx = graph.add_node(succ0)
    graph.add_edge(root_idx, succ_idx, object())

    with pytest.raises(TypeError, match="edge payload at"):
        transition_dag_from_rustworkx(graph)


def test_transition_dag_from_rustworkx_preserves_parallel_edges_in_transition_list(
    small_blocks,
):
    space, _domain, problem = small_blocks
    root = problem.get_initial_state()
    (action0, succ0), _ = _first_distinct_changed_transitions(space, root, count=2)

    graph = rx.PyDiGraph(multigraph=True)
    root_idx = graph.add_node(root)
    succ_idx = graph.add_node(succ0)
    graph.add_edge(root_idx, succ_idx, action0)
    graph.add_edge(root_idx, succ_idx, action0)

    dag = transition_dag_from_rustworkx(graph)

    assert dag.transitions() == [(0, 1), (0, 1)]
    assert dag.action(1) == adv_action(action0)


def test_transition_dag_from_rustworkx_reads_candidate_id_from_mapping_payload(
    small_blocks,
):
    space, _domain, problem = small_blocks
    root = problem.get_initial_state()
    (action0, succ0), _ = _first_distinct_changed_transitions(space, root, count=2)

    graph = rx.PyDiGraph()
    root_idx = graph.add_node({"state": root})
    succ_idx = graph.add_node({"state": succ0, "candidate_id": 123})
    graph.add_edge(root_idx, succ_idx, action0)

    dag = transition_dag_from_rustworkx(graph)
    assert dag.nodes()[0].candidate_id is None
    assert dag.nodes()[dag.index(adv_state(succ0))].candidate_id == 123


def test_transition_dag_from_rustworkx_rejects_partial_candidate_id_metadata(
    small_blocks,
):
    space, _domain, problem = small_blocks
    root = problem.get_initial_state()
    (action0, succ0), (action1, succ1) = _first_distinct_changed_transitions(
        space, root, count=2
    )

    graph = rx.PyDiGraph()
    root_idx = graph.add_node({"state": root})
    succ0_idx = graph.add_node({"state": succ0, "candidate_id": 1})
    succ1_idx = graph.add_node({"state": succ1})
    graph.add_edge(root_idx, succ0_idx, action0)
    graph.add_edge(root_idx, succ1_idx, action1)

    with pytest.raises(ValueError, match="partial candidate_id coverage"):
        transition_dag_from_rustworkx(graph)


def test_transition_dag_from_rustworkx_can_fill_missing_candidate_ids_with_node_index(
    small_blocks,
):
    space, _domain, problem = small_blocks
    root = problem.get_initial_state()
    (action0, succ0), (action1, succ1) = _first_distinct_changed_transitions(
        space, root, count=2
    )

    graph = rx.PyDiGraph()
    root_idx = graph.add_node({"state": root})
    succ0_idx = graph.add_node({"state": succ0, "candidate_id": 7})
    succ1_idx = graph.add_node({"state": succ1})
    graph.add_edge(root_idx, succ0_idx, action0)
    graph.add_edge(root_idx, succ1_idx, action1)

    dag = transition_dag_from_rustworkx(
        graph, fallback_missing_candidate_id_to_node_index=True
    )
    succ0_node_idx = dag.index(adv_state(succ0))
    succ1_node_idx = dag.index(adv_state(succ1))
    assert dag.nodes()[succ0_node_idx].candidate_id == 7
    assert dag.nodes()[succ1_node_idx].candidate_id == succ1_idx


def test_transition_dag_from_rustworkx_rejects_duplicate_candidate_ids(small_blocks):
    space, _domain, problem = small_blocks
    root = problem.get_initial_state()
    (action0, succ0), (action1, succ1) = _first_distinct_changed_transitions(
        space, root, count=2
    )

    graph = rx.PyDiGraph()
    root_idx = graph.add_node({"state": root})
    succ0_idx = graph.add_node({"state": succ0, "candidate_id": 5})
    succ1_idx = graph.add_node({"state": succ1, "candidate_id": 5})
    graph.add_edge(root_idx, succ0_idx, action0)
    graph.add_edge(root_idx, succ1_idx, action1)

    with pytest.raises(ValueError, match="duplicate candidate_id"):
        transition_dag_from_rustworkx(graph)
