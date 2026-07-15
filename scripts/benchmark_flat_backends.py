from __future__ import annotations

import argparse
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import statistics
import time
from typing import Any, Callable

import pymimir
from pypddl.formalism import ParserOptions
from pyyggdrasil.execution import ExecutionContext
from pytyr.formalism.planning import Parser
from pytyr.planning.lifted import (
    AxiomEvaluatorFactory,
    StateRepositoryFactory,
    SuccessorGeneratorFactory,
    Task,
)

import mifrost


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Result:
    backend: str
    path: str
    n_items: int
    median_ms: float
    mean_ms: float
    min_ms: float
    max_ms: float
    stdev_ms: float


def _times(fn: Callable[[], Any], *, warmup: int, repeats: int) -> list[float]:
    for _ in range(warmup):
        fn()
    values = []
    for _ in range(repeats):
        start = time.perf_counter()
        result = fn()
        values.append((time.perf_counter() - start) * 1000.0)
        _ = result
    return values


def _measure(
    backend: str,
    path: str,
    n_items: int,
    fn: Callable[[], Any],
    *,
    warmup: int,
    repeats: int,
) -> Result:
    values = _times(fn, warmup=warmup, repeats=repeats)
    return Result(
        backend=backend,
        path=path,
        n_items=n_items,
        median_ms=statistics.median(values),
        mean_ms=statistics.fmean(values),
        min_ms=min(values),
        max_ms=max(values),
        stdev_ms=statistics.pstdev(values),
    )


def _cycle(values: list[Any], size: int) -> list[Any]:
    return [values[index % len(values)] for index in range(size)]


def _pymimir_inputs(
    domain: str, problem: str, max_states: int
) -> tuple[Any, Any, list[Any]]:
    directory = ROOT / "data" / "pddl" / domain
    domain_value = pymimir.Domain(directory / "domain.pddl")
    problem_value = pymimir.Problem(
        domain_value, directory / f"{problem}.pddl", mode="grounded"
    )
    states = [problem_value.get_initial_state()]
    seen = {str(states[0])}
    queue = deque(states)
    while queue and len(states) < max_states:
        state = queue.popleft()
        for action in state.generate_applicable_actions():
            successor = action.apply(state)
            key = str(successor)
            if key in seen:
                continue
            seen.add(key)
            states.append(successor)
            queue.append(successor)
            if len(states) >= max_states:
                break
    return domain_value, problem_value, states


def _pytyr_inputs(
    domain: str, problem: str, max_states: int
) -> tuple[Any, Any, list[Any]]:
    from mifrost.backends.pytyr import PyTyrSnapshotReader

    directory = ROOT / "data" / "pddl" / domain
    options = ParserOptions()
    planning_task = Parser(str(directory / "domain.pddl"), options).parse_task(
        str(directory / f"{problem}.pddl"), options
    )
    task = Task(planning_task)
    context = ExecutionContext(1)
    evaluator = AxiomEvaluatorFactory().create(task, context)
    repository = StateRepositoryFactory().create(task, evaluator)
    generator = SuccessorGeneratorFactory().create(task, context, repository)
    root = generator.get_initial_node()
    nodes = [root]
    seen = {str(root.get_state())}
    queue = deque(nodes)
    while queue and len(nodes) < max_states:
        node = queue.popleft()
        for successor in generator.get_labeled_successor_nodes(node):
            key = str(successor.node.get_state())
            if key in seen:
                continue
            seen.add(key)
            nodes.append(successor.node)
            queue.append(successor.node)
            if len(nodes) >= max_states:
                break
    return (
        planning_task,
        PyTyrSnapshotReader(planning_task),
        [node.get_state() for node in nodes],
    )


def _stream_batch(encoder: Any, states: list[Any]) -> Any:
    stream = encoder.stream()
    for state in states:
        stream.append(state)
    return stream.flush()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", default="blocks")
    parser.add_argument("--problem", default="smedium")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-states", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()

    pymimir_domain, pymimir_problem, pymimir_pool = _pymimir_inputs(
        args.domain, args.problem, args.max_states
    )
    pytyr_task, pytyr_reader, pytyr_pool = _pytyr_inputs(
        args.domain, args.problem, args.max_states
    )
    pymimir_states = _cycle(pymimir_pool, args.batch_size)
    pytyr_states = _cycle(pytyr_pool, args.batch_size)
    pymimir_goals = list(pymimir_problem.get_goal_condition().get_literals())
    pytyr_goals = list(pytyr_reader.problem_snapshot().goals)
    pymimir_encoder = mifrost.FlatRelationEncoder(pymimir_domain)
    pytyr_encoder = mifrost.FlatRelationEncoder(pytyr_task)

    specs = (
        (
            "pymimir",
            "batch_default",
            args.batch_size,
            lambda: pymimir_encoder.encode_batch(pymimir_states),
        ),
        (
            "pytyr",
            "batch_default",
            args.batch_size,
            lambda: pytyr_encoder.encode_batch(pytyr_states),
        ),
        (
            "pymimir",
            "batch_explicit_goals",
            args.batch_size,
            lambda: pymimir_encoder.encode_batch(pymimir_states, goals=pymimir_goals),
        ),
        (
            "pytyr",
            "batch_explicit_goals",
            args.batch_size,
            lambda: pytyr_encoder.encode_batch(pytyr_states, goals=pytyr_goals),
        ),
        (
            "pymimir",
            "single_loop",
            args.batch_size,
            lambda: [pymimir_encoder.encode(state) for state in pymimir_states],
        ),
        (
            "pytyr",
            "single_loop",
            args.batch_size,
            lambda: [pytyr_encoder.encode(state) for state in pytyr_states],
        ),
        (
            "pymimir",
            "stream_end_to_end",
            args.batch_size,
            lambda: _stream_batch(pymimir_encoder, pymimir_states),
        ),
        (
            "pytyr",
            "stream_end_to_end",
            args.batch_size,
            lambda: _stream_batch(pytyr_encoder, pytyr_states),
        ),
    )
    results = [
        _measure(
            backend,
            path,
            n_items,
            fn,
            warmup=args.warmup,
            repeats=args.repeats,
        )
        for backend, path, n_items, fn in specs
    ]
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "domain": args.domain,
        "problem": args.problem,
        "batch_size": args.batch_size,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "state_pool_sizes": {
            "pymimir": len(pymimir_pool),
            "pytyr": len(pytyr_pool),
        },
        "results": [asdict(result) for result in results],
    }
    print(json.dumps(payload, indent=2))
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
