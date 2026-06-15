"""Test helpers for encoder fixtures.

These tests are intended to run against an installed `mifrost` (wheel or
editable). Do not force-import from the repository `src/` tree here, as that
breaks wheel tests by shadowing the compiled extension module.
"""

from pathlib import Path

import pymimir
from pymimir import Domain, Problem


class Transition:
    def __init__(self, action, target):
        self.action = action
        self.target = target


def problem_setup(
    domain_name, problem_name
) -> tuple[pymimir.wrapper_datasets.StateSpaceSampler, Domain, Problem]:
    domain, problem, state, domain_path, problem_path = load_problem(
        domain_name, problem_name
    )
    context = problem._search_context
    space, _ = pymimir.advanced.datasets.StateSpace.create(
        context, pymimir.advanced.datasets.StateSpaceOptions()
    )
    space = pymimir.wrapper_datasets.StateSpaceSampler(
        pymimir.advanced.datasets.StateSpaceSampler(space), problem
    )
    return space, domain, problem


def load_problem(
    domain: str = "blocks", problem: str = "smedium"
) -> tuple[pymimir.Domain, pymimir.Problem, pymimir.State, Path, Path]:
    root = Path(__file__).resolve().parents[1]
    domain_path = root / "data" / "pddl" / domain / "domain.pddl"
    problem_path = root / "data" / "pddl" / domain / f"{problem}.pddl"

    domain = pymimir.Domain(domain_path)
    problem = pymimir.Problem(domain, problem_path, mode="lifted")
    state = problem.get_initial_state()
    return domain, problem, state, domain_path, problem_path
