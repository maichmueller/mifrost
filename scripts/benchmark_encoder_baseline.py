from __future__ import annotations

import argparse
import json
import re
import statistics
import subprocess
import sys
import time
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, TypeVar

import pymimir

ROOT = Path(__file__).resolve().parents[1]

mifrost = None
to_advanced_action = None
to_advanced_state = None
_T = TypeVar("_T")


def _configure_import_mode(import_mode: str) -> None:
    if import_mode == "source":
        src = ROOT / "src"
        if str(src) not in sys.path:
            sys.path.insert(0, str(src))
        return
    if import_mode != "installed":
        raise ValueError(f"Unsupported import mode: {import_mode!r}")


def _import_runtime(import_mode: str) -> None:
    global mifrost
    global to_advanced_action
    global to_advanced_state

    _configure_import_mode(import_mode)

    import mifrost as mifrost_module
    from mifrost.encoders.types import (
        to_advanced_action as to_advanced_action_fn,
        to_advanced_state as to_advanced_state_fn,
    )

    mifrost = mifrost_module
    to_advanced_action = to_advanced_action_fn
    to_advanced_state = to_advanced_state_fn


@dataclass
class BenchResult:
    suite: str
    scenario: str
    path: str
    n_items: int
    median_ms: float
    mean_ms: float
    min_ms: float
    max_ms: float
    stdev_ms: float
    context: dict[str, Any]


def _state_key(state: Any) -> str:
    return str(state)


def _load_problem(domain: str, problem: str) -> tuple[Any, Any]:
    domain_path = ROOT / "data" / "pddl" / domain / "domain.pddl"
    problem_path = ROOT / "data" / "pddl" / domain / f"{problem}.pddl"
    domain_obj = pymimir.Domain(domain_path)
    problem_obj = pymimir.Problem(domain_obj, problem_path, mode="grounded")
    return domain_obj, problem_obj


def _goal_literals(problem_obj: Any) -> list[Any]:
    return list(problem_obj.get_goal_condition().get_literals())


def _collect_state_pool(
    root: Any,
    *,
    max_states: int,
    max_branch: int,
) -> tuple[list[Any], list[tuple[Any, Any, Any]]]:
    states = [root]
    transitions: list[tuple[Any, Any, Any]] = []
    seen = {_state_key(root): root}
    transition_seen: set[tuple[str, str]] = set()
    queue: deque[Any] = deque([root])

    while queue and len(states) < max_states:
        state = queue.popleft()
        actions = list(state.generate_applicable_actions())[:max_branch]
        for action in actions:
            successor = action.apply(state)
            sk = _state_key(state)
            tk = _state_key(successor)
            if (sk, tk) not in transition_seen:
                transition_seen.add((sk, tk))
                transitions.append((state, action, successor))
            if tk not in seen:
                seen[tk] = successor
                states.append(successor)
                queue.append(successor)
                if len(states) >= max_states:
                    break

    return states, transitions


def _pick_cycle(items: list[_T], n: int) -> list[_T]:
    if not items:
        return []
    return [items[i % len(items)] for i in range(n)]


def _build_dag(root: Any, *, max_nodes: int, max_branch: int) -> Any:
    dag = mifrost.TransitionDAG(to_advanced_state(root))
    seen = {_state_key(root): root}
    edge_seen: set[tuple[str, str]] = set()
    queue: deque[Any] = deque([root])

    while queue and len(seen) < max_nodes:
        state = queue.popleft()
        actions = list(state.generate_applicable_actions())[:max_branch]
        for action in actions:
            successor = action.apply(state)
            sk = _state_key(state)
            tk = _state_key(successor)
            if (sk, tk) not in edge_seen:
                edge_seen.add((sk, tk))
                dag.register_transition(
                    to_advanced_state(state),
                    to_advanced_state(successor),
                    to_advanced_action(action),
                )
            if tk not in seen and len(seen) < max_nodes:
                seen[tk] = successor
                queue.append(successor)

    return dag


def _benchmark(
    fn: Callable[[], Any],
    *,
    warmup: int,
    repeats: int,
) -> list[float]:
    for _ in range(max(0, warmup)):
        fn()
    durations_ms: list[float] = []
    for _ in range(max(1, repeats)):
        t0 = time.perf_counter()
        out = fn()
        durations_ms.append((time.perf_counter() - t0) * 1000.0)
        _ = out
    return durations_ms


def _measure(
    *,
    suite: str,
    scenario: str,
    path: str,
    n_items: int,
    fn: Callable[[], Any],
    warmup: int,
    repeats: int,
    context: dict[str, Any],
) -> BenchResult:
    durations = _benchmark(fn, warmup=warmup, repeats=repeats)
    return BenchResult(
        suite=suite,
        scenario=scenario,
        path=path,
        n_items=n_items,
        median_ms=statistics.median(durations),
        mean_ms=statistics.fmean(durations),
        min_ms=min(durations),
        max_ms=max(durations),
        stdev_ms=statistics.pstdev(durations),
        context=context,
    )


_CPP_MEDIAN_RE = re.compile(
    r"^(BM_[A-Za-z0-9_]+_median)\s+([0-9.]+)\s+ns\s+([0-9.]+)\s+ns\b"
)


def _find_cpp_benchmark(explicit: Path | None) -> Path | None:
    if explicit is not None:
        return explicit
    candidates = [
        path
        for path in ROOT.glob("build*/src/mifrost_bench_relation_encoders")
        if path.is_file()
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _run_cpp_relation_benchmark(
    executable: Path,
    *,
    domain: str,
    problem: str,
    batch_size: int,
) -> dict[str, float]:
    proc = subprocess.run(
        [
            str(executable),
            "--benchmark_min_time=0.05s",
            "--benchmark_repetitions=5",
            "--benchmark_report_aggregates_only=true",
            "--domain",
            domain,
            "--problem",
            problem,
            "--batch_size",
            str(batch_size),
        ],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    metrics: dict[str, float] = {}
    for line in proc.stdout.splitlines():
        match = _CPP_MEDIAN_RE.match(line.strip())
        if not match:
            continue
        metrics[match.group(1)] = float(match.group(3))
    if not metrics:
        raise RuntimeError(
            "Failed to parse median rows from mifrost_bench_relation_encoders output"
        )
    return metrics


def _build_python_matrix(
    domain_obj: Any,
    problem_obj: Any,
    *,
    batch_size: int,
    horizon_roots: int,
    horizon_dag_size: int,
    max_states: int,
    max_branch: int,
    warmup: int,
    repeats: int,
) -> list[BenchResult]:
    goals = _goal_literals(problem_obj)
    root = problem_obj.get_initial_state()
    states_pool, transitions_pool = _collect_state_pool(
        root,
        max_states=max_states,
        max_branch=max_branch,
    )
    states = _pick_cycle(states_pool, batch_size)
    roots = _pick_cycle(states_pool, horizon_roots)
    dags = [
        _build_dag(state, max_nodes=horizon_dag_size, max_branch=max_branch)
        for state in roots
    ]
    transition_pairs = _pick_cycle(transitions_pool, batch_size)
    current_states = [entry[0] for entry in transition_pairs]
    successors = [entry[2] for entry in transition_pairs]

    results: list[BenchResult] = []
    context = {
        "batch_size": batch_size,
        "horizon_roots": horizon_roots,
        "horizon_dag_size": horizon_dag_size,
        "state_pool_size": len(states_pool),
        "transition_pool_size": len(transitions_pool),
    }

    hgraph = mifrost.HGraphEncoder(domain_obj)
    results.append(
        _measure(
            suite="python",
            scenario="state_goals",
            path="hgraph_batch_native",
            n_items=len(states),
            fn=lambda: hgraph.encode_batch(states, goals=goals),
            warmup=warmup,
            repeats=repeats,
            context=context,
        )
    )
    results.append(
        _measure(
            suite="python",
            scenario="state_goals",
            path="hgraph_single_native",
            n_items=len(states),
            fn=lambda: [hgraph.encode(state, goals=goals) for state in states],
            warmup=warmup,
            repeats=repeats,
            context=context,
        )
    )
    hgraph_no_names = mifrost.HGraphEncoder(domain_obj, export_node_names=False)
    results.append(
        _measure(
            suite="python",
            scenario="state_goals",
            path="hgraph_batch_native_no_names",
            n_items=len(states),
            fn=lambda: hgraph_no_names.encode_batch(states, goals=goals),
            warmup=warmup,
            repeats=repeats,
            context=context,
        )
    )

    flat = mifrost.FlatRelationEncoder(domain_obj)
    results.append(
        _measure(
            suite="python",
            scenario="state_goals",
            path="flat_batch_native",
            n_items=len(states),
            fn=lambda: flat.encode_batch(states, goals=goals),
            warmup=warmup,
            repeats=repeats,
            context=context,
        )
    )
    flat_no_names = mifrost.FlatRelationEncoder(domain_obj, export_node_names=False)
    results.append(
        _measure(
            suite="python",
            scenario="state_goals",
            path="flat_batch_native_no_names",
            n_items=len(states),
            fn=lambda: flat_no_names.encode_batch(states, goals=goals),
            warmup=warmup,
            repeats=repeats,
            context=context,
        )
    )

    color = mifrost.ColorEncoder(domain_obj)
    results.append(
        _measure(
            suite="python",
            scenario="state_goals",
            path="color_batch_native",
            n_items=len(states),
            fn=lambda: color.encode_batch(states, goals=goals),
            warmup=warmup,
            repeats=repeats,
            context=context,
        )
    )

    horizon = mifrost.HorizonEncoder(domain_obj)
    results.append(
        _measure(
            suite="python",
            scenario="horizon_full_goals",
            path="horizon_batch_native",
            n_items=len(roots),
            fn=lambda: horizon.encode_batch(roots, dags=dags, goals=goals),
            warmup=warmup,
            repeats=repeats,
            context=context,
        )
    )
    results.append(
        _measure(
            suite="python",
            scenario="horizon_full_goals",
            path="horizon_single_native",
            n_items=len(roots),
            fn=lambda: [
                horizon.encode(root_state, dag=dag, goals=goals)
                for root_state, dag in zip(roots, dags, strict=True)
            ],
            warmup=warmup,
            repeats=repeats,
            context=context,
        )
    )
    flat_horizon = mifrost.FlatHorizonEncoder(domain_obj)
    results.append(
        _measure(
            suite="python",
            scenario="horizon_full_goals",
            path="flat_horizon_batch_native",
            n_items=len(roots),
            fn=lambda: flat_horizon.encode_batch(roots, dags=dags, goals=goals),
            warmup=warmup,
            repeats=repeats,
            context=context,
        )
    )

    transition_full = mifrost.TransitionHGraphEncoder(domain_obj)
    results.append(
        _measure(
            suite="python",
            scenario="transition_full_goals",
            path="transition_batch_native",
            n_items=len(current_states),
            fn=lambda: transition_full.encode_batch(
                current_states,
                successors=successors,
                goals=goals,
            ),
            warmup=warmup,
            repeats=repeats,
            context=context,
        )
    )
    results.append(
        _measure(
            suite="python",
            scenario="transition_full_goals",
            path="transition_single_native",
            n_items=len(current_states),
            fn=lambda: [
                transition_full.encode(current, successor=successor, goals=goals)
                for current, successor in zip(current_states, successors, strict=True)
            ],
            warmup=warmup,
            repeats=repeats,
            context=context,
        )
    )
    transition_delta = mifrost.TransitionEffectsHGraphEncoder(domain_obj)
    results.append(
        _measure(
            suite="python",
            scenario="transition_delta_goals",
            path="transition_delta_batch_native",
            n_items=len(current_states),
            fn=lambda: transition_delta.encode_batch(
                current_states,
                successors=successors,
                goals=goals,
            ),
            warmup=warmup,
            repeats=repeats,
            context=context,
        )
    )
    flat_transition_full = mifrost.FlatTransitionEncoder(domain_obj)
    results.append(
        _measure(
            suite="python",
            scenario="transition_full_goals",
            path="flat_transition_batch_native",
            n_items=len(current_states),
            fn=lambda: flat_transition_full.encode_batch(
                current_states,
                successors=successors,
                goals=goals,
            ),
            warmup=warmup,
            repeats=repeats,
            context=context,
        )
    )
    flat_transition_delta = mifrost.FlatTransitionEffectsEncoder(domain_obj)
    results.append(
        _measure(
            suite="python",
            scenario="transition_delta_goals",
            path="flat_transition_delta_batch_native",
            n_items=len(current_states),
            fn=lambda: flat_transition_delta.encode_batch(
                current_states,
                successors=successors,
                goals=goals,
            ),
            warmup=warmup,
            repeats=repeats,
            context=context,
        )
    )

    conversion_inputs = {
        "hgraph_as_pyg": hgraph.encode_batch(states, goals=goals),
        "flat_as_pyg": flat.encode_batch(states, goals=goals),
        "horizon_as_pyg": horizon.encode_batch(roots, dags=dags, goals=goals),
        "transition_as_pyg": transition_full.encode_batch(
            current_states,
            successors=successors,
            goals=goals,
        ),
    }
    for path_name, encoding in conversion_inputs.items():
        results.append(
            _measure(
                suite="python",
                scenario="conversion",
                path=path_name,
                n_items=encoding.num_graphs,
                fn=lambda enc=encoding: enc.as_pyg(as_batch=True),
                warmup=warmup,
                repeats=repeats,
                context=context,
            )
        )

    return results


def _find_result(
    results: list[BenchResult],
    *,
    scenario: str,
    path: str,
) -> BenchResult:
    for result in results:
        if result.scenario == scenario and result.path == path:
            return result
    raise KeyError((scenario, path))


def _per_item_ms(result: BenchResult) -> float:
    return result.median_ms / max(1, result.n_items)


def _compute_ratio_report(
    python_results: list[BenchResult],
    cpp_blocks: dict[str, float] | None,
    cpp_gripper: dict[str, float] | None,
) -> dict[str, float]:
    ratios = {
        "hgraph_batch_vs_single_per_item": _per_item_ms(
            _find_result(
                python_results,
                scenario="state_goals",
                path="hgraph_single_native",
            )
        )
        / _per_item_ms(
            _find_result(
                python_results,
                scenario="state_goals",
                path="hgraph_batch_native",
            )
        ),
        "transition_batch_vs_single_per_item": _per_item_ms(
            _find_result(
                python_results,
                scenario="transition_full_goals",
                path="transition_single_native",
            )
        )
        / _per_item_ms(
            _find_result(
                python_results,
                scenario="transition_full_goals",
                path="transition_batch_native",
            )
        ),
        "horizon_batch_vs_single_per_item": _per_item_ms(
            _find_result(
                python_results,
                scenario="horizon_full_goals",
                path="horizon_single_native",
            )
        )
        / _per_item_ms(
            _find_result(
                python_results,
                scenario="horizon_full_goals",
                path="horizon_batch_native",
            )
        ),
        "flat_vs_hgraph_python_batch": _find_result(
            python_results,
            scenario="state_goals",
            path="hgraph_batch_native",
        ).median_ms
        / _find_result(
            python_results,
            scenario="state_goals",
            path="flat_batch_native",
        ).median_ms,
    }
    if cpp_blocks is not None:
        ratios["flat_vs_hgraph_cpp_blocks_batch"] = (
            cpp_blocks["BM_HGraphEncodeBatch_median"]
            / cpp_blocks["BM_FlatEncodeBatch_median"]
        )
    if cpp_gripper is not None:
        ratios["flat_vs_hgraph_cpp_gripper_batch"] = (
            cpp_gripper["BM_HGraphEncodeBatch_median"]
            / cpp_gripper["BM_FlatEncodeBatch_median"]
        )
    return ratios


def _check_thresholds(
    ratios: dict[str, float],
    *,
    min_cpp_blocks: float,
    min_cpp_gripper: float,
    min_hgraph_batch: float,
    min_transition_batch: float,
    min_horizon_batch: float,
) -> list[str]:
    failures: list[str] = []
    if "flat_vs_hgraph_cpp_blocks_batch" in ratios and (
        ratios["flat_vs_hgraph_cpp_blocks_batch"] < min_cpp_blocks
    ):
        failures.append(
            "flat_vs_hgraph_cpp_blocks_batch "
            f"{ratios['flat_vs_hgraph_cpp_blocks_batch']:.3f} < {min_cpp_blocks:.3f}"
        )
    if "flat_vs_hgraph_cpp_gripper_batch" in ratios and (
        ratios["flat_vs_hgraph_cpp_gripper_batch"] < min_cpp_gripper
    ):
        failures.append(
            "flat_vs_hgraph_cpp_gripper_batch "
            f"{ratios['flat_vs_hgraph_cpp_gripper_batch']:.3f} < {min_cpp_gripper:.3f}"
        )
    if ratios["hgraph_batch_vs_single_per_item"] < min_hgraph_batch:
        failures.append(
            "hgraph_batch_vs_single_per_item "
            f"{ratios['hgraph_batch_vs_single_per_item']:.3f} < {min_hgraph_batch:.3f}"
        )
    if ratios["transition_batch_vs_single_per_item"] < min_transition_batch:
        failures.append(
            "transition_batch_vs_single_per_item "
            f"{ratios['transition_batch_vs_single_per_item']:.3f} < {min_transition_batch:.3f}"
        )
    if ratios["horizon_batch_vs_single_per_item"] < min_horizon_batch:
        failures.append(
            "horizon_batch_vs_single_per_item "
            f"{ratios['horizon_batch_vs_single_per_item']:.3f} < {min_horizon_batch:.3f}"
        )
    return failures


def _print_python_results(results: list[BenchResult]) -> None:
    print(
        "suite".ljust(10)
        + "scenario".ljust(24)
        + "path".ljust(32)
        + "items".rjust(7)
        + "median_ms".rjust(12)
    )
    print("-" * 85)
    for result in results:
        print(
            result.suite.ljust(10)
            + result.scenario.ljust(24)
            + result.path.ljust(32)
            + str(result.n_items).rjust(7)
            + f"{result.median_ms:12.3f}"
        )


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Focused encoder baseline matrix with ratio gates."
    )
    parser.add_argument(
        "--import-mode",
        choices=("installed", "source"),
        default="installed",
    )
    parser.add_argument("--domain", default="blocks")
    parser.add_argument("--problem", default="smedium")
    parser.add_argument("--max-states", type=int, default=64)
    parser.add_argument("--max-branch", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--horizon-roots", type=int, default=8)
    parser.add_argument("--horizon-dag-size", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--cpp-bench",
        type=Path,
        default=None,
        help="Path to mifrost_bench_relation_encoders. Auto-detected when omitted.",
    )
    parser.add_argument("--skip-cpp", action="store_true")
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--min-flat-vs-hgraph-cpp-blocks", type=float, default=3.5)
    parser.add_argument("--min-flat-vs-hgraph-cpp-gripper", type=float, default=4.5)
    parser.add_argument("--min-hgraph-batch-vs-single", type=float, default=3.0)
    parser.add_argument("--min-transition-batch-vs-single", type=float, default=2.5)
    parser.add_argument("--min-horizon-batch-vs-single", type=float, default=1.25)
    args = parser.parse_args(argv)

    _import_runtime(args.import_mode)

    cpp_blocks: dict[str, float] | None = None
    cpp_gripper: dict[str, float] | None = None
    if not args.skip_cpp:
        executable = _find_cpp_benchmark(args.cpp_bench)
        if executable is None:
            raise SystemExit(
                "Could not find mifrost_bench_relation_encoders; pass --cpp-bench "
                "or use --skip-cpp"
            )
        cpp_blocks = _run_cpp_relation_benchmark(
            executable,
            domain="blocks",
            problem="smedium",
            batch_size=32,
        )
        cpp_gripper = _run_cpp_relation_benchmark(
            executable,
            domain="gripper",
            problem="gripper_b-5",
            batch_size=32,
        )

    domain_obj, problem_obj = _load_problem(args.domain, args.problem)
    python_results = _build_python_matrix(
        domain_obj,
        problem_obj,
        batch_size=args.batch_size,
        horizon_roots=args.horizon_roots,
        horizon_dag_size=args.horizon_dag_size,
        max_states=args.max_states,
        max_branch=args.max_branch,
        warmup=args.warmup,
        repeats=args.repeats,
    )
    ratios = _compute_ratio_report(python_results, cpp_blocks, cpp_gripper)
    failures = _check_thresholds(
        ratios,
        min_cpp_blocks=args.min_flat_vs_hgraph_cpp_blocks,
        min_cpp_gripper=args.min_flat_vs_hgraph_cpp_gripper,
        min_hgraph_batch=args.min_hgraph_batch_vs_single,
        min_transition_batch=args.min_transition_batch_vs_single,
        min_horizon_batch=args.min_horizon_batch_vs_single,
    )

    if cpp_blocks is not None:
        print(
            "C++ blocks batch speedup:",
            f"{ratios['flat_vs_hgraph_cpp_blocks_batch']:.3f}x",
        )
    if cpp_gripper is not None:
        print(
            "C++ gripper batch speedup:",
            f"{ratios['flat_vs_hgraph_cpp_gripper_batch']:.3f}x",
        )
    print("Python ratios:")
    for key, value in sorted(ratios.items()):
        if key.startswith("flat_vs_hgraph_cpp_"):
            continue
        print(f"  {key}: {value:.3f}x")
    _print_python_results(python_results)

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config": {
            "import_mode": args.import_mode,
            "domain": args.domain,
            "problem": args.problem,
            "max_states": args.max_states,
            "max_branch": args.max_branch,
            "batch_size": args.batch_size,
            "horizon_roots": args.horizon_roots,
            "horizon_dag_size": args.horizon_dag_size,
            "warmup": args.warmup,
            "repeats": args.repeats,
            "skip_cpp": args.skip_cpp,
        },
        "cpp_blocks": cpp_blocks,
        "cpp_gripper": cpp_gripper,
        "python_results": [asdict(result) for result in python_results],
        "ratios": ratios,
        "failures": failures,
    }
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Wrote JSON report: {args.output_json}")

    if failures:
        for failure in failures:
            print(f"PERF REGRESSION: {failure}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
