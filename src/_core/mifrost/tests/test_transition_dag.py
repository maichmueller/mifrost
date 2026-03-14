import pytest
import pymimir as mimir
import mifrost
import tempfile
import os


@pytest.fixture
def test_setup():
    """Create a minimal PDDL domain and problem for testing."""
    domain_str = """(define (domain test)
    (:requirements :strips :negative-preconditions)
    (:predicates (p ?x) (q ?x ?y))
    (:action test-action
        :parameters (?x)
        :precondition (not (p ?x))
        :effect (p ?x)))
"""
    problem_str = """(define (problem test-problem)
    (:domain test)
    (:objects a b c)
    (:init)
    (:goal (p c)))
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".pddl", delete=False) as df:
        df.write(domain_str)
        domain_file = df.name

    with tempfile.NamedTemporaryFile(mode="w", suffix=".pddl", delete=False) as pf:
        pf.write(problem_str)
        problem_file = pf.name

    try:
        domain = mimir.Domain(domain_file)
        prob = mimir.Problem(domain, problem_file)
        yield prob
    finally:
        os.unlink(domain_file)
        os.unlink(problem_file)


def test_transition_dag_basics(test_setup):
    prob = test_setup
    root = prob.get_initial_state()
    root_adv = root._advanced_state

    dag = mifrost.TransitionDAG(root_adv)

    assert dag.root() == root_adv
    assert dag.root_index() == 0
    assert dag.contains(root_adv)
    assert dag.index(root_adv) == 0

    # Check nodes()
    nodes = dag.nodes()
    assert len(nodes) == 1
    assert nodes[0].state == root_adv
    assert nodes[0].index == 0
    assert nodes[0].depth == 0
    assert nodes[0].action is None
    assert nodes[0].candidate_id is None


def test_transition_dag_transitions(test_setup):
    prob = test_setup
    domain = prob.get_domain()
    root = prob.get_initial_state()
    root_adv = root._advanced_state

    # Find the action 'test-action a'
    obj_a = prob.get_object("a")
    action_schema = domain.get_action("test-action")
    action_a = prob.new_ground_action(action_schema, [obj_a])
    action_a_adv = action_a._advanced_ground_action

    # Apply action to get next state
    succ = action_a.apply(root)
    succ_adv = succ._advanced_state

    dag = mifrost.TransitionDAG(root_adv)
    p_idx, c_idx = dag.register_transition(root_adv, succ_adv, action_a_adv)

    assert p_idx == 0
    assert c_idx == 1
    assert dag.contains(succ_adv)
    assert dag.index(succ_adv) == 1
    assert dag.depth(1) == 1
    assert dag.action(1) == action_a_adv

    # Check transitions()
    ts = dag.transitions()
    assert len(ts) == 1
    assert ts[0] == (0, 1)

    # Check successors()
    s = dag.successors()
    assert len(s) == 1
    assert s[0].state == succ_adv
    assert s[0].index == 1
    assert s[0].candidate_id is None


def test_transition_dag_candidate_id_assignment_and_conflicts(test_setup):
    prob = test_setup
    domain = prob.get_domain()
    root = prob.get_initial_state()
    root_adv = root._advanced_state

    obj_a = prob.get_object("a")
    action_schema = domain.get_action("test-action")
    action_a = prob.new_ground_action(action_schema, [obj_a])
    action_a_adv = action_a._advanced_ground_action
    succ = action_a.apply(root)
    succ_adv = succ._advanced_state

    dag = mifrost.TransitionDAG(root_adv)
    dag.register_transition(root_adv, succ_adv, action_a_adv, candidate_id=11)
    assert dag.nodes()[1].candidate_id == 11

    # Re-registering with the same candidate_id is allowed.
    dag.register_transition(root_adv, succ_adv, action_a_adv, candidate_id=11)
    assert dag.nodes()[1].candidate_id == 11

    with pytest.raises(ValueError, match="conflicting candidate_id"):
        dag.register_transition(root_adv, succ_adv, action_a_adv, candidate_id=12)


def test_transition_dag_delta_literals_assignment_and_conflicts(test_setup):
    prob = test_setup
    domain = prob.get_domain()
    root = prob.get_initial_state()
    root_adv = root._advanced_state

    obj_a = prob.get_object("a")
    action_schema = domain.get_action("test-action")
    action_a = prob.new_ground_action(action_schema, [obj_a])
    action_a_adv = action_a._advanced_ground_action
    succ = action_a.apply(root)
    succ_adv = succ._advanced_state
    succ_atoms = list(succ.get_atoms(ignore_static=True))
    assert succ_atoms
    delta_literals = [mimir.GroundLiteral.new(succ_atoms[0], True, prob)]

    dag = mifrost.TransitionDAG(root_adv)
    dag.register_transition(
        root_adv,
        succ_adv,
        action_a_adv,
        candidate_id=11,
        delta_literals=delta_literals,
    )
    assert dag.nodes()[1].delta_literals is not None
    assert len(dag.nodes()[1].delta_literals) == 1
    assert str(dag.nodes()[1].delta_literals[0]) == str(delta_literals[0])

    dag.register_transition(
        root_adv,
        succ_adv,
        action_a_adv,
        candidate_id=11,
        delta_literals=delta_literals,
    )
    assert len(dag.nodes()[1].delta_literals) == 1

    with pytest.raises(ValueError, match="conflicting delta_literals"):
        dag.register_transition(
            root_adv,
            succ_adv,
            action_a_adv,
            candidate_id=11,
            delta_literals=[],
        )


def test_transition_dag_dag_property(test_setup):
    prob = test_setup
    root = prob.get_initial_state()
    root_adv = root._advanced_state
    domain = prob.get_domain()

    # Create two paths to the same state
    # (root -> a -> ab) and (root -> b -> ab)
    obj_a = prob.get_object("a")
    obj_b = prob.get_object("b")
    action_schema = domain.get_action("test-action")

    action_a = prob.new_ground_action(action_schema, [obj_a])
    action_b = prob.new_ground_action(action_schema, [obj_b])

    action_a_adv = action_a._advanced_ground_action
    action_b_adv = action_b._advanced_ground_action

    state_a = action_a.apply(root)
    state_a_adv = state_a._advanced_state
    state_b = action_b.apply(root)
    state_b_adv = state_b._advanced_state

    dag = mifrost.TransitionDAG(root_adv)
    dag.register_transition(root_adv, state_a_adv, action_a_adv)
    dag.register_transition(root_adv, state_b_adv, action_b_adv)

    # Now common child
    # In PDDL, applying a then b might lead to same state as b then a
    state_ab = action_b.apply(state_a)
    state_ab_adv = state_ab._advanced_state

    dag.register_transition(state_a_adv, state_ab_adv, action_b_adv)
    dag.register_transition(state_b_adv, state_ab_adv, action_a_adv)

    assert dag.contains(state_ab_adv)
    # state_ab should have index 3 (root=0, a=1, b=2, ab=3)
    assert dag.index(state_ab_adv) == 3
    assert len(dag.transitions()) == 4
    assert (1, 3) in dag.transitions()
    assert (2, 3) in dag.transitions()

    # Children of root
    assert set(dag.children(0)) == {1, 2}
    # Children of a
    assert dag.children(1) == [3]
    # Children of b
    assert dag.children(2) == [3]


def test_transition_dag_register_transitions_bulk_matches_incremental(test_setup):
    prob = test_setup
    domain = prob.get_domain()
    root = prob.get_initial_state()
    root_adv = root._advanced_state

    obj_a = prob.get_object("a")
    obj_b = prob.get_object("b")
    action_schema = domain.get_action("test-action")

    action_a = prob.new_ground_action(action_schema, [obj_a])
    action_b = prob.new_ground_action(action_schema, [obj_b])
    action_a_adv = action_a._advanced_ground_action
    action_b_adv = action_b._advanced_ground_action

    state_a = action_a.apply(root)
    state_b = action_b.apply(root)
    state_a_adv = state_a._advanced_state
    state_b_adv = state_b._advanced_state

    incremental = mifrost.TransitionDAG(root_adv)
    incremental.register_transition(root_adv, state_a_adv, action_a_adv)
    incremental.register_transition(root_adv, state_b_adv, action_b_adv)

    bulk = mifrost.TransitionDAG(root_adv)
    bulk.register_transitions(
        [
            (root_adv, state_a_adv, action_a_adv),
            (root_adv, state_b_adv, action_b_adv, 42),
        ]
    )

    assert bulk.transitions() == incremental.transitions()
    assert [node.depth for node in bulk.nodes()] == [
        node.depth for node in incremental.nodes()
    ]
    assert [node.action for node in bulk.nodes()] == [
        node.action for node in incremental.nodes()
    ]
    assert bulk.nodes()[1].candidate_id is None
    assert bulk.nodes()[2].candidate_id == 42


def test_transition_dag_register_transitions_accepts_delta_literals(test_setup):
    prob = test_setup
    domain = prob.get_domain()
    root = prob.get_initial_state()
    root_adv = root._advanced_state

    obj_a = prob.get_object("a")
    action_schema = domain.get_action("test-action")
    action_a = prob.new_ground_action(action_schema, [obj_a])
    action_a_adv = action_a._advanced_ground_action
    state_a = action_a.apply(root)
    state_a_adv = state_a._advanced_state
    state_a_atoms = list(state_a.get_atoms(ignore_static=True))
    assert state_a_atoms
    delta_literals = [mimir.GroundLiteral.new(state_a_atoms[0], True, prob)]

    bulk = mifrost.TransitionDAG(root_adv)
    bulk.register_transitions(
        [
            (root_adv, state_a_adv, action_a_adv, 7, delta_literals),
        ]
    )

    assert bulk.nodes()[1].candidate_id == 7
    assert [str(literal) for literal in bulk.nodes()[1].delta_literals] == [
        str(literal) for literal in delta_literals
    ]


def test_transition_dag_register_transitions_accepts_unordered_edges(test_setup):
    prob = test_setup
    domain = prob.get_domain()
    root = prob.get_initial_state()
    root_adv = root._advanced_state

    obj_a = prob.get_object("a")
    obj_b = prob.get_object("b")
    action_schema = domain.get_action("test-action")

    action_a = prob.new_ground_action(action_schema, [obj_a])
    action_b = prob.new_ground_action(action_schema, [obj_b])
    action_a_adv = action_a._advanced_ground_action
    action_b_adv = action_b._advanced_ground_action

    state_a = action_a.apply(root)
    state_b = action_b.apply(root)
    state_a_adv = state_a._advanced_state
    state_b_adv = state_b._advanced_state

    bulk = mifrost.TransitionDAG(root_adv)
    bulk.register_transitions(
        [
            (state_a_adv, state_b_adv, action_b_adv),
            (root_adv, state_a_adv, action_a_adv),
        ]
    )

    # Edge order is preserved by insertion; index assignment stays deterministic.
    assert bulk.transitions() == [(0, 1), (1, 2)]
    assert bulk.depth(0) == 0
    assert bulk.depth(1) == 1
    assert bulk.depth(2) == 2
    assert bulk.nodes()[1].candidate_id is None
    assert bulk.nodes()[2].candidate_id is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
