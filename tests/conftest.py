"""Test helpers to avoid editable-import hooks during local runs."""

import pathlib
import sys
from pathlib import Path
from typing import Any

import pymimir


def problem_setup(
    domain_name, problem_name
) -> tuple[
    pymimir.wrapper_datasets.StateSpaceSampler,
    pymimir.wrapper_formalism.Domain,
    pymimir.wrapper_formalism.Problem,
]:
    domain, problem, state, domain_path, problem_path = load_problem(
        domain_name, problem_name
    )

    space = pymimir.wrapper_datasets.AdvancedStateSpace.create(problem._search_context)
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
