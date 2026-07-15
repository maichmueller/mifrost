"""Executable semantic contract shared by Pymimir and PyTyr."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import pymimir
import pytest
from pypddl.formalism import ParserOptions
from pyyggdrasil.execution import ExecutionContext
from pytyr.formalism.planning import Parser
from pytyr.planning.lifted import (
    AxiomEvaluatorFactory,
    StateRepositoryFactory,
    SuccessorGeneratorFactory,
    Task,
)

from mifrost.backends.pymimir import PymimirSnapshotReader
from mifrost.backends.pytyr import PyTyrSnapshotReader
from mifrost.backends.semantic import PredicateCategory, SnapshotReader


ROOT = Path(__file__).resolve().parents[2]
PARITY_CASES = (
    ("blocks", "small"),
    ("blocks_multiple", "blocks_b-2_v-1"),
    ("gripper", "gripper_b-1"),
    ("spanner", "small"),
    ("visitall", "visitall_x-2_y-1_r-50"),
    ("delivery", "instance_2x2_p-1_0"),
    ("reward", "instance_3x3_0"),
)


def _pddl_paths(domain: str, problem: str) -> tuple[Path, Path]:
    directory = ROOT / "data" / "pddl" / domain
    return directory / "domain.pddl", directory / f"{problem}.pddl"


def _pymimir_problem(domain_path: Path, problem_path: Path):
    domain = pymimir.Domain(domain_path)
    return pymimir.Problem(domain, problem_path, mode="lifted")


def _pytyr_problem(domain_path: Path, problem_path: Path):
    options = ParserOptions()
    parser = Parser(str(domain_path), options)
    planning_task = parser.parse_task(str(problem_path), options)
    task = Task(planning_task)
    context = ExecutionContext(1)
    axiom_evaluator = AxiomEvaluatorFactory().create(task, context)
    state_repository = StateRepositoryFactory().create(task, axiom_evaluator)
    successor_generator = SuccessorGeneratorFactory().create(
        task, context, state_repository
    )
    return planning_task, successor_generator


def _backend_pair(domain: str = "blocks", problem: str = "small"):
    domain_path, problem_path = _pddl_paths(domain, problem)
    pymimir_problem = _pymimir_problem(domain_path, problem_path)
    planning_task, successor_generator = _pytyr_problem(domain_path, problem_path)
    return (
        PymimirSnapshotReader(pymimir_problem),
        pymimir_problem,
        PyTyrSnapshotReader(planning_task),
        successor_generator,
    )


def _edge_case_pair(tmp_path: Path):
    domain_path = tmp_path / "domain.pddl"
    problem_path = tmp_path / "problem.pddl"
    domain_path.write_text(
        dedent(
            """
            (define (domain semantic-edge)
              (:requirements
                :strips :typing :negative-preconditions :equality
                :derived-predicates)
              (:types item)
              (:predicates
                (linked ?from - item ?to - item)
                (enabled)
                (marked ?value - item)
                (reachable ?value - item))
              (:derived (reachable ?value - item) (marked ?value))
              (:action mark
                :parameters (?value - item ?other - item)
                :precondition
                  (and
                    (enabled)
                    (linked ?value ?other)
                    (not (= ?value ?other)))
                :effect (marked ?value)))
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    problem_path.write_text(
        dedent(
            """
            (define (problem semantic-edge-problem)
              (:domain semantic-edge)
              (:objects a b - item)
              (:init (enabled) (linked a b) (marked a))
              (:goal (and (enabled) (reachable a) (not (marked b)))))
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    pymimir_problem = _pymimir_problem(domain_path, problem_path)
    planning_task, successor_generator = _pytyr_problem(domain_path, problem_path)
    return (
        PymimirSnapshotReader(pymimir_problem),
        pymimir_problem,
        PyTyrSnapshotReader(planning_task),
        successor_generator,
    )


@pytest.fixture
def backend_pair():
    return _backend_pair()


def test_readers_satisfy_backend_neutral_contract(backend_pair) -> None:
    pymimir_reader, _, pytyr_reader, _ = backend_pair

    assert isinstance(pymimir_reader, SnapshotReader)
    assert isinstance(pytyr_reader, SnapshotReader)
    assert pymimir_reader.backend_name != pytyr_reader.backend_name


def test_domain_and_problem_snapshots_have_semantic_parity(backend_pair) -> None:
    pymimir_reader, _, pytyr_reader, _ = backend_pair

    assert pymimir_reader.domain_snapshot() == pytyr_reader.domain_snapshot()
    assert pymimir_reader.problem_snapshot() == pytyr_reader.problem_snapshot()


@pytest.mark.parametrize(("domain", "problem"), PARITY_CASES)
def test_domain_problem_and_root_state_parity_across_fixtures(
    domain: str, problem: str
) -> None:
    pymimir_reader, pymimir_problem, pytyr_reader, successor_generator = _backend_pair(
        domain, problem
    )

    assert pymimir_reader.domain_snapshot() == pytyr_reader.domain_snapshot()
    assert pymimir_reader.problem_snapshot() == pytyr_reader.problem_snapshot()
    assert pymimir_reader.state_snapshot(
        pymimir_problem.get_initial_state()
    ) == pytyr_reader.state_snapshot(successor_generator.get_initial_node().get_state())


def test_root_state_and_zero_arity_fact_have_semantic_parity(backend_pair) -> None:
    pymimir_reader, pymimir_problem, pytyr_reader, successor_generator = backend_pair

    pymimir_state = pymimir_problem.get_initial_state()
    pytyr_state = successor_generator.get_initial_node().get_state()
    pymimir_snapshot = pymimir_reader.state_snapshot(pymimir_state)
    pytyr_snapshot = pytyr_reader.state_snapshot(pytyr_state)

    assert pymimir_snapshot == pytyr_snapshot
    assert any(
        atom.predicate.category is PredicateCategory.FLUENT
        and atom.predicate.name == "handempty"
        and atom.objects == ()
        for atom in pymimir_snapshot.fluent_atoms
    )


def test_typed_equality_negative_goal_and_derived_fact_parity(
    tmp_path: Path,
) -> None:
    pymimir_reader, pymimir_problem, pytyr_reader, successor_generator = (
        _edge_case_pair(tmp_path)
    )

    assert pymimir_reader.domain_snapshot() == pytyr_reader.domain_snapshot()
    pymimir_problem_snapshot = pymimir_reader.problem_snapshot()
    assert pymimir_problem_snapshot == pytyr_reader.problem_snapshot()

    pymimir_state = pymimir_reader.state_snapshot(pymimir_problem.get_initial_state())
    assert pymimir_state == pytyr_reader.state_snapshot(
        successor_generator.get_initial_node().get_state()
    )

    assert any(
        atom.predicate.category is PredicateCategory.STATIC
        and atom.predicate.name == "enabled"
        and atom.objects == ()
        for atom in pymimir_state.static_atoms
    )
    assert any(
        atom.predicate.category is PredicateCategory.DERIVED
        and atom.predicate.name == "reachable"
        and atom.objects == ("a",)
        for atom in pymimir_state.derived_atoms
    )
    assert any(
        not literal.polarity
        and literal.atom.predicate.name == "marked"
        and literal.atom.objects == ("b",)
        for literal in pymimir_problem_snapshot.goals
    )


def test_pyTyr_no_value_fdr_fact_is_omitted(backend_pair) -> None:
    _, _, pytyr_reader, _ = backend_pair

    class NoValueFact:
        def has_value(self) -> bool:
            return False

        def get_atom(self):
            raise AssertionError("a no-value FDR fact has no propositional atom")

    class StateWithNoValueFact:
        def static_atoms(self):
            return ()

        def fluent_facts(self):
            return (NoValueFact(),)

        def derived_atoms(self):
            return ()

    snapshot = pytyr_reader.state_snapshot(StateWithNoValueFact())

    assert snapshot.atoms == ()


def test_matching_successor_actions_and_states_have_semantic_parity(
    backend_pair,
) -> None:
    pymimir_reader, pymimir_problem, pytyr_reader, successor_generator = backend_pair
    pymimir_root = pymimir_problem.get_initial_state()
    pytyr_root = successor_generator.get_initial_node()

    pymimir_successors = {
        pymimir_reader.action_key(action): pymimir_reader.state_snapshot(
            action.apply(pymimir_root)
        )
        for action in pymimir_root.generate_applicable_actions()
    }
    pytyr_successors = {
        pytyr_reader.action_key(labeled.label): pytyr_reader.state_snapshot(
            labeled.node.get_state()
        )
        for labeled in successor_generator.get_labeled_successor_nodes(pytyr_root)
    }

    assert pymimir_successors == pytyr_successors


def test_independent_readers_can_alternate_in_one_process(backend_pair) -> None:
    pymimir_reader, pymimir_problem, pytyr_reader, successor_generator = backend_pair

    expected_domain = pymimir_reader.domain_snapshot()
    expected_state = pymimir_reader.state_snapshot(pymimir_problem.get_initial_state())
    for _ in range(3):
        assert pytyr_reader.domain_snapshot() == expected_domain
        assert pymimir_reader.domain_snapshot() == expected_domain
        assert (
            pytyr_reader.state_snapshot(
                successor_generator.get_initial_node().get_state()
            )
            == expected_state
        )
        assert (
            pymimir_reader.state_snapshot(pymimir_problem.get_initial_state())
            == expected_state
        )


def test_readers_reject_objects_from_the_other_backend(backend_pair) -> None:
    pymimir_reader, pymimir_problem, pytyr_reader, successor_generator = backend_pair
    pymimir_state = pymimir_problem.get_initial_state()
    pytyr_state = successor_generator.get_initial_node().get_state()

    with pytest.raises(TypeError, match="pymimir.State"):
        pymimir_reader.state_snapshot(pytyr_state)
    with pytest.raises(TypeError, match="pytyr state"):
        pytyr_reader.state_snapshot(pymimir_state)
