from __future__ import annotations

from collections.abc import Callable

import mifrost
import pytest


def _predicates() -> list[mifrost.SemanticPredicateSpec]:
    return [
        mifrost.SemanticPredicateSpec(
            mifrost.SemanticPredicateCategory.fluent, "at", 1
        ),
        mifrost.SemanticPredicateSpec(
            mifrost.SemanticPredicateCategory.static, "ready", 0
        ),
    ]


def _actions() -> list[mifrost.SemanticActionSpec]:
    return [
        mifrost.SemanticActionSpec("move", 2),
        mifrost.SemanticActionSpec("finish", 0),
    ]


def _state(object_index: int = 0) -> mifrost.SemanticFlatRelationInput:
    value = mifrost.SemanticFlatRelationInput()
    value.objects = ["a", "b"]
    value.state_facts = [
        mifrost.SemanticAtom(0, [object_index]),
        mifrost.SemanticAtom(1, []),
    ]
    value.goals = [mifrost.SemanticLiteral(mifrost.SemanticAtom(0, [1]))]
    value.actions = [mifrost.SemanticGroundAction(1, [])]
    value.subgoal_layers = [
        [mifrost.SemanticLiteral(mifrost.SemanticAtom(0, [0]), False)]
    ]
    value.history = [
        mifrost.SemanticHistoryEntry(
            -1,
            [mifrost.SemanticLiteral(mifrost.SemanticAtom(0, [object_index]))],
        )
    ]
    value.history_max_steps = 2
    return value


def _node(
    index: int,
    depth: int,
    *,
    object_index: int = 0,
    action: mifrost.SemanticGroundAction | None = None,
    candidate_id: int | None = None,
    deltas: list[mifrost.SemanticLiteral] | None = None,
    name: str | None = None,
) -> mifrost.SemanticTransitionNode:
    return mifrost.SemanticTransitionNode(
        _state(object_index),
        index,
        depth,
        incoming_action=action,
        candidate_id=candidate_id,
        delta_literals=deltas,
        display_name=name,
    )


def _nodes() -> list[mifrost.SemanticTransitionNode]:
    return [
        _node(0, 0, name="root"),
        _node(
            1,
            1,
            object_index=1,
            action=mifrost.SemanticGroundAction(0, [0, 1]),
            candidate_id=101,
            deltas=[mifrost.SemanticLiteral(mifrost.SemanticAtom(0, [1]))],
            name="left",
        ),
        _node(
            2,
            1,
            action=mifrost.SemanticGroundAction(1, []),
            candidate_id=202,
            name="right",
        ),
        _node(
            3,
            2,
            object_index=1,
            action=mifrost.SemanticGroundAction(0, [1, 0]),
            candidate_id=303,
        ),
    ]


def _edges() -> list[tuple[int, int]]:
    return [(2, 3), (0, 2), (1, 3), (0, 1)]


def _dag(
    nodes: list[mifrost.SemanticTransitionNode] | None = None,
    edges: list[tuple[int, int]] | None = None,
) -> mifrost.SemanticTransitionDAG:
    return mifrost.SemanticTransitionDAG(
        _predicates(),
        _actions(),
        _nodes() if nodes is None else nodes,
        _edges() if edges is None else edges,
    )


def _invalid_state_fact(nodes: list[mifrost.SemanticTransitionNode]) -> None:
    nodes[1].state.state_facts = [mifrost.SemanticAtom(99, [0])]


def _invalid_object_names(nodes: list[mifrost.SemanticTransitionNode]) -> None:
    for node in nodes:
        node.state.objects = ["a", "a"]


def _invalid_goal_arity(nodes: list[mifrost.SemanticTransitionNode]) -> None:
    nodes[1].state.goals = [mifrost.SemanticLiteral(mifrost.SemanticAtom(0, []))]


def _invalid_subgoal_object(nodes: list[mifrost.SemanticTransitionNode]) -> None:
    nodes[1].state.subgoal_layers = [
        [mifrost.SemanticLiteral(mifrost.SemanticAtom(0, [2]))]
    ]


def _invalid_history_dt(nodes: list[mifrost.SemanticTransitionNode]) -> None:
    nodes[1].state.history = [
        mifrost.SemanticHistoryEntry(
            0, [mifrost.SemanticLiteral(mifrost.SemanticAtom(0, [0]))]
        )
    ]


def _invalid_state_action(nodes: list[mifrost.SemanticTransitionNode]) -> None:
    nodes[1].state.actions = [mifrost.SemanticGroundAction(99, [])]


def _invalid_incoming_action(nodes: list[mifrost.SemanticTransitionNode]) -> None:
    nodes[1].incoming_action = mifrost.SemanticGroundAction(0, [])


def _invalid_delta_literal(nodes: list[mifrost.SemanticTransitionNode]) -> None:
    nodes[1].delta_literals = [mifrost.SemanticLiteral(mifrost.SemanticAtom(0, [3]))]


def test_semantic_transition_dag_owns_deterministic_graph() -> None:
    dag = _dag()

    assert len(dag) == 4
    assert dag.root.index == 0
    assert dag.root.depth == 0
    assert dag.root.state.objects == ["a", "b"]
    assert dag.edges == [(0, 1), (0, 2), (1, 3), (2, 3)]
    assert dag.children(0) == [1, 2]
    assert dag.children(3) == []
    assert dag.parents(3) == [1, 2]
    assert [node.index for node in dag.nodes] == [0, 1, 2, 3]
    assert [node.depth for node in dag.nodes] == [0, 1, 1, 2]
    assert dag.nodes[1].incoming_action.action == 0
    assert dag.nodes[1].incoming_action.arguments == [0, 1]
    assert dag.nodes[1].delta_literals[0].positive
    assert dag.nodes[1].display_name == "left"
    assert dag.candidate_id_coverage() == mifrost.SemanticCandidateIdCoverage.complete
    assert dag.candidate_ids() == [101, 202, 303]
    dag.validate_candidate_ids()

    with pytest.raises(IndexError, match="node index out of range"):
        dag.children(-1)
    with pytest.raises(IndexError, match="node index out of range"):
        dag.parents(4)


def test_semantic_transition_dag_returns_owned_node_snapshots() -> None:
    dag = _dag()
    root = dag.root
    root.state.objects = ["changed"]

    assert dag.root.state.objects == ["a", "b"]
    with pytest.raises(AttributeError):
        dag.root = root


def test_semantic_transition_dag_allows_root_only_empty_schema() -> None:
    dag = mifrost.SemanticTransitionDAG(
        [],
        [],
        [mifrost.SemanticTransitionNode(mifrost.SemanticFlatRelationInput(), 0, 0)],
        [],
    )

    assert len(dag) == 1
    assert dag.predicates == []
    assert dag.actions == []
    assert dag.edges == []
    assert dag.candidate_ids() == []


def test_semantic_transition_dag_candidate_validation_primitives() -> None:
    nodes = _nodes()
    for node in nodes[1:]:
        node.candidate_id = None
    implicit = _dag(nodes)
    assert implicit.candidate_id_coverage() == getattr(
        mifrost.SemanticCandidateIdCoverage, "none"
    )
    assert implicit.candidate_ids() == []

    nodes[1].candidate_id = 7
    partial = _dag(nodes)
    assert (
        partial.candidate_id_coverage() == mifrost.SemanticCandidateIdCoverage.partial
    )
    with pytest.raises(ValueError, match="missing candidate_id for node index 2"):
        partial.validate_candidate_ids()

    nodes[2].candidate_id = 7
    nodes[3].candidate_id = 8
    duplicate = _dag(nodes)
    with pytest.raises(ValueError, match="duplicate candidate_id 7"):
        duplicate.candidate_ids()

    nodes[2].candidate_id = 9
    nodes[0].candidate_id = 42
    explicit = _dag(nodes)
    assert explicit.candidate_ids() == [7, 9, 8]
    assert explicit.candidate_ids(include_root=True) == [42, 7, 9, 8]


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda nodes: setattr(nodes[2], "index", 7), "stable and contiguous"),
        (
            lambda nodes: setattr(nodes[0], "depth", 1),
            "root must have index and depth 0",
        ),
        (lambda nodes: setattr(nodes[1], "depth", -1), "depth must be non-negative"),
        (
            lambda nodes: setattr(nodes[2].state, "objects", ["b", "a"]),
            "identical ordered object tables",
        ),
        (lambda nodes: setattr(nodes[1], "display_name", ""), "display name"),
        (
            lambda nodes: setattr(
                nodes[0], "incoming_action", mifrost.SemanticGroundAction(1, [])
            ),
            "root cannot have an incoming action",
        ),
        (lambda nodes: setattr(nodes[0], "delta_literals", []), "root cannot have"),
    ],
)
def test_semantic_transition_dag_rejects_node_invariant_violations(
    mutate: Callable[[list[mifrost.SemanticTransitionNode]], None],
    message: str,
) -> None:
    nodes = _nodes()
    mutate(nodes)
    with pytest.raises(ValueError, match=message):
        _dag(nodes)


def test_semantic_transition_dag_requires_a_root() -> None:
    with pytest.raises(ValueError, match="nonempty node list"):
        _dag([], [])


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (_invalid_object_names, "object names must be unique"),
        (
            _invalid_state_fact,
            "predicate index out of range",
        ),
        (
            _invalid_goal_arity,
            "schema arity",
        ),
        (
            _invalid_subgoal_object,
            "object index out of range",
        ),
        (_invalid_history_dt, "negative dt"),
        (
            lambda nodes: setattr(nodes[1].state, "history_max_steps", -1),
            "history_max_steps",
        ),
        (
            _invalid_state_action,
            "action index out of range",
        ),
        (
            _invalid_incoming_action,
            "incoming action argument count",
        ),
        (
            _invalid_delta_literal,
            "delta literal object index",
        ),
    ],
)
def test_semantic_transition_dag_rejects_invalid_semantic_payloads(
    mutate: Callable[[list[mifrost.SemanticTransitionNode]], None],
    message: str,
) -> None:
    nodes = _nodes()
    mutate(nodes)
    with pytest.raises(ValueError, match=message):
        _dag(nodes)


def test_semantic_transition_dag_validates_schema() -> None:
    predicates = _predicates()
    predicates[1].name = predicates[0].name
    with pytest.raises(ValueError, match="unique predicate names"):
        mifrost.SemanticTransitionDAG(predicates, _actions(), _nodes(), _edges())

    actions = _actions()
    actions[0].arity = -1
    with pytest.raises(ValueError, match="action arity must be non-negative"):
        mifrost.SemanticTransitionDAG(_predicates(), actions, _nodes(), _edges())


@pytest.mark.parametrize(
    ("edges", "message"),
    [
        ([(0, 4)], "edge endpoint out of range"),
        ([(0, 0)], "self edges"),
        ([(0, 1), (0, 1)], "duplicate edge"),
        ([(0, 1), (1, 2), (2, 1), (2, 3)], "acyclic"),
        ([(0, 1), (0, 2)], "reachable from root"),
    ],
)
def test_semantic_transition_dag_rejects_malformed_topology(
    edges: list[tuple[int, int]], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _dag(edges=edges)


@pytest.mark.parametrize("depth", [1, 3])
def test_semantic_transition_dag_requires_exact_shortest_path_depth(depth: int) -> None:
    nodes = _nodes()
    nodes[3].depth = depth
    with pytest.raises(ValueError, match="shortest-path depth"):
        _dag(nodes)
