"""Shared helpers for encoder example scripts."""

from __future__ import annotations

from itertools import product
from pathlib import Path
from typing import TYPE_CHECKING

import pymimir

if TYPE_CHECKING:
    import matplotlib


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_problem(
    *, domain: str = "blocks", problem: str = "smedium"
) -> tuple[pymimir.Domain, pymimir.Problem, pymimir.State]:
    root = repo_root()
    domain_path = root / "data" / "pddl" / domain / "domain.pddl"
    problem_path = root / "data" / "pddl" / domain / f"{problem}.pddl"
    domain_obj = pymimir.Domain(domain_path)
    problem_obj = pymimir.Problem(domain_obj, problem_path, mode="lifted")
    return domain_obj, problem_obj, problem_obj.get_initial_state()


def _type_name_closure(type_obj) -> set[str]:
    names: set[str] = set()
    stack = [type_obj]
    while stack:
        current = stack.pop()
        if current is None or not hasattr(current, "get_name"):
            continue
        name = current.get_name()
        if name in names:
            continue
        names.add(name)
        if hasattr(current, "get_bases"):
            stack.extend(list(current.get_bases()))
    return names


def _matching_wrapper_objects(
    problem: pymimir.Problem, parameter, advanced_objects
) -> list[pymimir.Object]:
    if not hasattr(parameter, "get_bases"):
        return list(problem.get_objects())

    allowed_type_names: set[str] = set()
    for base in parameter.get_bases():
        allowed_type_names.update(_type_name_closure(base))
    if not allowed_type_names:
        return list(problem.get_objects())

    matching_names = [
        obj.get_name()
        for obj in advanced_objects
        if any(
            type_name in allowed_type_names
            for base in obj.get_bases()
            for type_name in _type_name_closure(base)
        )
    ]
    return [problem.get_object(name) for name in matching_names]


def false_problem_literals(
    problem: pymimir.Problem,
    state: pymimir.State,
    *,
    include_static: bool = True,
    include_fluent: bool = True,
    include_derived: bool = True,
) -> list[pymimir.GroundLiteral]:
    """Return all positive literals that are well-typed but false in ``state``."""
    domain = problem.get_domain()
    advanced_problem = getattr(problem, "_advanced_problem", None)
    advanced_objects = (
        list(advanced_problem.get_problem_and_domain_objects())
        if advanced_problem is not None
        else list(problem.get_objects())
    )
    false_literals: list[pymimir.GroundLiteral] = []

    predicates = domain.get_predicates(
        ignore_static=not include_static,
        ignore_fluent=not include_fluent,
        ignore_derived=not include_derived,
    )
    for predicate in predicates:
        if hasattr(predicate, "get_typed_parameters"):
            wrapper_object_pools = [
                _matching_wrapper_objects(problem, parameter, advanced_objects)
                for parameter in predicate.get_typed_parameters()
            ]
        else:
            objects = list(problem.get_objects())
            wrapper_object_pools = [objects] * predicate.get_arity()

        assignments = product(*wrapper_object_pools) if wrapper_object_pools else [()]
        for objects in assignments:
            atom = problem.new_ground_atom(predicate, list(objects))
            literal = problem.new_ground_literal(atom, True)
            if not state.literal_holds(literal):
                false_literals.append(literal)

    return false_literals


def load_space(
    *, domain: str = "blocks", problem: str = "smedium"
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


def plot_output_dir() -> Path:
    out_dir = Path(__file__).resolve().parent / "_plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def save_plot(name: str, dpi=300) -> Path:
    from matplotlib import pyplot as plt

    out_path = plot_output_dir() / name
    # plt.tight_layout()
    plt.savefig(out_path, dpi=dpi)
    plt.close()
    return out_path


def begin_plot(*, figsize: tuple[float, float] = (14, 10)) -> matplotlib.figure.Figure:
    from matplotlib import pyplot as plt

    return plt.figure(figsize=figsize)
