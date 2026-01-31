"""Test helpers to avoid editable-import hooks during local runs."""

import pathlib
import sys
from pathlib import Path
from typing import Any

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
    domain: str = "blocks", problem: str = "probBLOCKS-4-0"
) -> tuple[pymimir.Domain, pymimir.Problem, pymimir.State, Path, Path]:
    root = Path(__file__).resolve().parents[1]
    domain_path = root / "data" / "pddl" / domain / "domain.pddl"
    problem_path = root / "data" / "pddl" / domain / f"{problem}.pddl"

    domain = pymimir.Domain(domain_path)
    problem = pymimir.Problem(domain, problem_path, mode="lifted")
    state = problem.get_initial_state()
    return domain, problem, state, domain_path, problem_path


def _strip_scikit_build_editable() -> None:
    sys.meta_path = [
        finder
        for finder in sys.meta_path
        if finder.__class__.__module__ != "_mifrost_editable"
    ]


_strip_scikit_build_editable()
