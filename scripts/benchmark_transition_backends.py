from __future__ import annotations

import argparse
from collections import deque
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import resource
import sys
from typing import Any

import mifrost

from benchmark_flat_backends import _cycle, _measure


ROOT = Path(__file__).resolve().parents[1]


def _pymimir_transitions(
    domain: str,
    problem: str,
    max_states: int,
) -> tuple[Any, Any, list[tuple[Any, Any]]]:
    import pymimir

    directory = ROOT / "data" / "pddl" / domain
    domain_value = pymimir.Domain(directory / "domain.pddl")
    problem_value = pymimir.Problem(
        domain_value, directory / f"{problem}.pddl", mode="grounded"
    )
    root = problem_value.get_initial_state()
    states = [root]
    seen = {str(root)}
    queue = deque(states)
    transitions: list[tuple[Any, Any]] = []
    while queue and len(states) < max_states:
        current = queue.popleft()
        for action in current.generate_applicable_actions():
            successor = action.apply(current)
            transitions.append((current, successor))
            key = str(successor)
            if key in seen:
                continue
            seen.add(key)
            states.append(successor)
            queue.append(successor)
            if len(states) >= max_states:
                break
    if not transitions:
        raise RuntimeError("Pymimir fixture produced no transitions")
    return domain_value, problem_value, transitions


def _pytyr_transitions(
    domain: str,
    problem: str,
    max_states: int,
) -> tuple[Any, Any, list[tuple[Any, Any]]]:
    from pypddl.formalism import ParserOptions
    from pyyggdrasil.execution import ExecutionContext
    from pytyr.formalism.planning import Parser
    from pytyr.planning.lifted import (
        AxiomEvaluatorFactory,
        StateRepositoryFactory,
        SuccessorGeneratorFactory,
        Task,
    )

    from mifrost.backends.pytyr import PyTyrSnapshotReader

    directory = ROOT / "data" / "pddl" / domain
    options = ParserOptions()
    planning_task = Parser(str(directory / "domain.pddl"), options).parse_task(
        str(directory / f"{problem}.pddl"), options
    )
    task = Task(planning_task)
    context = ExecutionContext(1)
    evaluator = AxiomEvaluatorFactory().create(task, context)
    repository = StateRepositoryFactory().create(task)
    generator = SuccessorGeneratorFactory().create(task, context)
    root = generator.get_initial_node(repository, evaluator)
    nodes = [root]
    seen = {str(root.get_state())}
    queue = deque(nodes)
    transitions: list[tuple[Any, Any]] = []
    while queue and len(nodes) < max_states:
        current = queue.popleft()
        for labeled in generator.get_labeled_successor_nodes(
            current, repository, evaluator
        ):
            successor = labeled.node
            transitions.append((current.get_state(), successor.get_state()))
            key = str(successor.get_state())
            if key in seen:
                continue
            seen.add(key)
            nodes.append(successor)
            queue.append(successor)
            if len(nodes) >= max_states:
                break
    if not transitions:
        raise RuntimeError("PyTyr fixture produced no transitions")
    return planning_task, PyTyrSnapshotReader(planning_task), transitions


def _stream_batch(
    encoder: Any,
    transitions: list[tuple[Any, Any]],
) -> Any:
    stream = encoder.stream()
    for current, successor in transitions:
        stream.append(current, successor)
    return stream.flush()


def _backend_specs(
    backend: str,
    domain_or_task: Any,
    transitions: list[tuple[Any, Any]],
    goals: list[Any],
) -> list[tuple[str, str, int, Any]]:
    size = len(transitions)
    currents = [current for current, _successor in transitions]
    successors = [successor for _current, successor in transitions]
    specs: list[tuple[str, str, int, Any]] = []
    for mode, encoder_type in (
        ("full", mifrost.TransitionHGraphEncoder),
        ("delta", mifrost.TransitionEffectsHGraphEncoder),
    ):
        encoder = encoder_type(domain_or_task)
        specs.extend(
            [
                (
                    backend,
                    f"{mode}.batch_default",
                    size,
                    lambda encoder=encoder: encoder.encode_batch(
                        currents, successors=successors
                    ),
                ),
                (
                    backend,
                    f"{mode}.batch_explicit_goals",
                    size,
                    lambda encoder=encoder: encoder.encode_batch(
                        currents,
                        successors=successors,
                        goals=goals,
                    ),
                ),
                (
                    backend,
                    f"{mode}.single_loop",
                    size,
                    lambda encoder=encoder: [
                        encoder.encode(current, successor=successor)
                        for current, successor in transitions
                    ],
                ),
                (
                    backend,
                    f"{mode}.stream_end_to_end",
                    size,
                    lambda encoder=encoder: _stream_batch(encoder, transitions),
                ),
            ]
        )
    return specs


def _module_path(name: str) -> str | None:
    module = sys.modules.get(name)
    value = getattr(module, "__file__", None)
    return None if value is None else str(Path(value).resolve())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", default="blocks")
    parser.add_argument("--problem", default="smedium")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-states", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--repeats", type=int, default=200)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--round-offset", type=int, default=0)
    backend_group = parser.add_mutually_exclusive_group()
    backend_group.add_argument("--pymimir-only", action="store_true")
    backend_group.add_argument("--pytyr-only", action="store_true")
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()

    specs_by_backend: dict[str, list[tuple[str, str, int, Any]]] = {}
    transition_pool_sizes: dict[str, int] = {}
    if not args.pytyr_only:
        pymimir_domain, pymimir_problem, pymimir_pool = _pymimir_transitions(
            args.domain, args.problem, args.max_states
        )
        pymimir_transitions = _cycle(pymimir_pool, args.batch_size)
        pymimir_goals = list(pymimir_problem.get_goal_condition().get_literals())
        specs_by_backend["pymimir"] = _backend_specs(
            "pymimir",
            pymimir_domain,
            pymimir_transitions,
            pymimir_goals,
        )
        transition_pool_sizes["pymimir"] = len(pymimir_pool)

    if not args.pymimir_only:
        pytyr_task, pytyr_reader, pytyr_pool = _pytyr_transitions(
            args.domain, args.problem, args.max_states
        )
        pytyr_transitions = _cycle(pytyr_pool, args.batch_size)
        pytyr_goals = list(pytyr_reader.problem_snapshot().goals)
        specs_by_backend["pytyr"] = _backend_specs(
            "pytyr",
            pytyr_task,
            pytyr_transitions,
            pytyr_goals,
        )
        transition_pool_sizes["pytyr"] = len(pytyr_pool)

    rounds = []
    backend_order = list(specs_by_backend)
    for round_index in range(args.rounds):
        absolute_round = round_index + args.round_offset
        order = (
            backend_order if absolute_round % 2 == 0 else list(reversed(backend_order))
        )
        results = []
        for backend in order:
            specs = specs_by_backend[backend]
            path_order = specs if absolute_round % 2 == 0 else list(reversed(specs))
            results.extend(
                _measure(
                    result_backend,
                    path,
                    n_items,
                    fn,
                    warmup=args.warmup,
                    repeats=args.repeats,
                )
                for result_backend, path, n_items, fn in path_order
            )
        rounds.append(
            {
                "round": absolute_round + 1,
                "backend_order": order,
                "results": [asdict(result) for result in results],
            }
        )

    usage = resource.getrusage(resource.RUSAGE_SELF)
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "mifrost_version": getattr(mifrost, "__version__", "unknown"),
        "modules": {
            "mifrost": str(Path(mifrost.__file__).resolve()),
            "neutral_core": _module_path("mifrost._neutral_core"),
            "pymimir_adapter": _module_path("mifrost._pymimir_adapter"),
            "pytyr_adapter": _module_path("mifrost._pytyr_adapter"),
        },
        "process": {
            "pid": os.getpid(),
            "maximum_resident_set_size": int(usage.ru_maxrss),
        },
        "domain": args.domain,
        "problem": args.problem,
        "batch_size": args.batch_size,
        "max_states": args.max_states,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "round_count": args.rounds,
        "transition_pool_sizes": transition_pool_sizes,
        "rounds": rounds,
    }
    report = json.dumps(payload, indent=2)
    print(report)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(report + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
