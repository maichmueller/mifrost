from __future__ import annotations

import argparse
import cProfile
import pstats
import sys
import time
from pathlib import Path

import pymimir
from torch_geometric.data import Batch


def _strip_scikit_build_editable() -> None:
    sys.meta_path = [
        finder
        for finder in sys.meta_path
        if finder.__class__.__module__ != "_mifrost_editable"
    ]


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
_strip_scikit_build_editable()

import mifrost


def load_problem(domain: str, problem: str):
    root = Path(__file__).resolve().parents[1]
    domain_path = root / "data" / "pddl" / domain / "domain.pddl"
    problem_path = root / "data" / "pddl" / domain / f"{problem}.pddl"
    domain_obj = pymimir.Domain(domain_path)
    problem_obj = pymimir.Problem(domain_obj, problem_path, mode="lifted")
    return domain_obj, problem_obj


def build_states(problem_obj: pymimir.Problem, count: int):
    root = problem_obj.get_initial_state()
    if count <= 1:
        return [root]
    # Try to gather unique successors, then repeat as needed.
    states = [root]
    for action in root.generate_applicable_actions():
        succ = action.apply(root)
        states.append(succ)
        if len(states) >= count:
            break
    while len(states) < count:
        states.append(states[-1])
    return states


def prepare_inputs(
    states, problem_obj, include_goals, include_actions, include_subgoals
):
    goals = None
    subgoal_layers = None
    if include_goals or include_subgoals:
        goals = list(problem_obj.get_goal_condition().get_literals())
    if include_subgoals and goals:
        subgoal_layers = [goals[:1]]
    actions = None
    if include_actions:
        actions = [state.generate_applicable_actions() for state in states]
    return goals, subgoal_layers, actions


def bench_encode_single(encoder, states, goals, subgoal_layers, actions, iterations):
    n = len(states)
    for i in range(iterations):
        idx = i % n
        encoder.encode(
            states[idx],
            goals=goals,
            actions=actions[idx] if actions is not None else None,
            subgoal_layers=subgoal_layers,
        )


def bench_encode_single_parts(
    encoder, states, goals, subgoal_layers, actions, iterations
):
    n = len(states)
    for i in range(iterations):
        idx = i % n
        encoder.encode_parts(
            states[idx],
            goals=goals,
            actions=actions[idx] if actions is not None else None,
            subgoal_layers=subgoal_layers,
        )


def bench_encode_batch(encoder, states, goals, subgoal_layers, actions):
    encoder.encode_batch(
        states,
        goals=goals,
        actions=actions,
        subgoal_layers=subgoal_layers,
    )


def bench_encode_stream(encoder, states, goals, subgoal_layers, actions):
    stream = encoder.stream()
    for idx, state in enumerate(states):
        stream.append(
            state,
            goals=goals,
            actions=actions[idx] if actions is not None else None,
            subgoal_layers=subgoal_layers,
        )
    stream.flush(as_batch=True)


def bench_encode_batch_no_metadata(encoder, states, goals, subgoal_layers, actions):
    encoder.encode_batch(
        states,
        goals=goals,
        actions=actions,
        subgoal_layers=subgoal_layers,
        include_metadata=False,
    )


def bench_encode_batch_parts(encoder, states, goals, subgoal_layers, actions):
    encoder.encode_batch_parts(
        states,
        goals=goals,
        actions=actions,
        subgoal_layers=subgoal_layers,
    )


def timed(label, fn, count=None):
    start = time.perf_counter()
    fn()
    elapsed = time.perf_counter() - start
    if count:
        per = elapsed / count
        print(f"{label}: {elapsed:.4f}s total ({per:.6f}s per item)")
    else:
        print(f"{label}: {elapsed:.4f}s")


def run_profile(profile_kind, label, fn, count=None):
    if profile_kind == "none":
        timed(label, fn, count=count)
        return
    if profile_kind == "pyinstrument":
        try:
            from pyinstrument import Profiler
        except ImportError:
            raise SystemExit(
                "pyinstrument not installed; install it or use --profile=cprofile"
            )
        profiler = Profiler()
        profiler.start()
        fn()
        profiler.stop()
        print(profiler.output_text(unicode=True, color=True))
        return
    profiler = cProfile.Profile()
    profiler.enable()
    fn()
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats("cumtime")
    print(f"== {label} (top 25 by cumulative time) ==")
    stats.print_stats(25)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", default="blocks")
    parser.add_argument("--problem", default="probBLOCKS-4-0")
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--stream-size", type=int, default=16)
    parser.add_argument(
        "--profile", choices=("none", "cprofile", "pyinstrument"), default="none"
    )
    parser.add_argument("--include-goals", action="store_true")
    parser.add_argument("--include-actions", action="store_true")
    parser.add_argument("--include-subgoals", action="store_true")
    args = parser.parse_args(argv)

    domain_obj, problem_obj = load_problem(args.domain, args.problem)
    states_single = build_states(problem_obj, max(1, args.iterations))
    states_batch = build_states(problem_obj, max(1, args.batch_size))
    states_stream = build_states(problem_obj, max(1, args.stream_size))

    goals, subgoal_layers, actions_single = prepare_inputs(
        states_single,
        problem_obj,
        args.include_goals,
        args.include_actions,
        args.include_subgoals,
    )
    _, _, actions_batch = prepare_inputs(
        states_batch,
        problem_obj,
        args.include_goals,
        args.include_actions,
        args.include_subgoals,
    )
    _, _, actions_stream = prepare_inputs(
        states_stream,
        problem_obj,
        args.include_goals,
        args.include_actions,
        args.include_subgoals,
    )

    encoder = mifrost.HGraphEncoder(domain_obj, ignore_actions=not args.include_actions)

    run_profile(
        args.profile,
        "encode_single",
        lambda: bench_encode_single(
            encoder,
            states_single,
            goals,
            subgoal_layers,
            actions_single,
            args.iterations,
        ),
        count=args.iterations,
    )
    run_profile(
        args.profile,
        "encode_single_parts",
        lambda: bench_encode_single_parts(
            encoder,
            states_single,
            goals,
            subgoal_layers,
            actions_single,
            args.iterations,
        ),
        count=args.iterations,
    )
    run_profile(
        args.profile,
        "encode_batch",
        lambda: bench_encode_batch(
            encoder,
            states_batch,
            goals,
            subgoal_layers,
            actions_batch,
        ),
        count=len(states_batch),
    )
    run_profile(
        args.profile,
        "encode_stream",
        lambda: bench_encode_stream(
            encoder,
            states_stream,
            goals,
            subgoal_layers,
            actions_stream,
        ),
        count=len(states_stream),
    )
    run_profile(
        args.profile,
        "encode_batch_parts",
        lambda: bench_encode_batch_parts(
            encoder,
            states_batch,
            goals,
            subgoal_layers,
            actions_batch,
        ),
        count=len(states_batch),
    )
    run_profile(
        args.profile,
        "encode_batch_no_metadata",
        lambda: bench_encode_batch_no_metadata(
            encoder,
            states_batch,
            goals,
            subgoal_layers,
            actions_batch,
        ),
        count=len(states_batch),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
