"""Shared helpers for encoder example scripts."""

from __future__ import annotations

from pathlib import Path

import pymimir


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_problem(
    *, domain: str = "blocks", problem: str = "probBLOCKS-4-0"
) -> tuple[pymimir.Domain, pymimir.Problem, pymimir.State]:
    root = repo_root()
    domain_path = root / "data" / "pddl" / domain / "domain.pddl"
    problem_path = root / "data" / "pddl" / domain / f"{problem}.pddl"
    domain_obj = pymimir.Domain(domain_path)
    problem_obj = pymimir.Problem(domain_obj, problem_path, mode="lifted")
    return domain_obj, problem_obj, problem_obj.get_initial_state()


def load_space(
    *, domain: str = "blocks", problem: str = "probBLOCKS-4-0"
) -> tuple[pymimir.wrapper_datasets.StateSpaceSampler, pymimir.Domain, pymimir.Problem]:
    domain_obj, problem_obj, _ = load_problem(domain=domain, problem=problem)
    state_space, _ = pymimir.advanced.datasets.StateSpace.create(
        problem_obj._search_context,
        pymimir.advanced.datasets.StateSpaceOptions(),
    )
    space = pymimir.wrapper_datasets.StateSpaceSampler(
        pymimir.advanced.datasets.StateSpaceSampler(state_space),
        problem_obj,
    )
    return space, domain_obj, problem_obj


def first_transition(
    space: pymimir.wrapper_datasets.StateSpaceSampler,
    state: pymimir.State,
) -> tuple[pymimir.GroundAction, pymimir.State]:
    transitions = list(space.get_forward_transitions(state))
    if not transitions:
        raise RuntimeError("No successor found from state.")
    return transitions[0]
