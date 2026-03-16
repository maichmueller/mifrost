from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence, TypeVar

import pymimir

ROOT = Path(__file__).resolve().parents[1]

mifrost = None
HGraphEncoder = None
HorizonEncoder = None
TransitionEffectsHGraphEncoder = None
TransitionHGraphEncoder = None
HGRAPH_BENCH_SPEC = None
TRANSITION_LANE_SPEC = None

StateT = Any
ActionT = Any
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
    global HGraphEncoder
    global HorizonEncoder
    global TransitionEffectsHGraphEncoder
    global TransitionHGraphEncoder
    global HGRAPH_BENCH_SPEC
    global TRANSITION_LANE_SPEC

    _configure_import_mode(import_mode)

    import mifrost as mifrost_module
    from mifrost.encoders import (
        HGraphEncoder as HGraphEncoderCls,
        HorizonEncoder as HorizonEncoderCls,
        TransitionEffectsHGraphEncoder as TransitionEffectsHGraphEncoderCls,
        TransitionHGraphEncoder as TransitionHGraphEncoderCls,
    )
    from mifrost.encoders._lane_specs import (
        HGRAPH_BENCH_SPEC as HGraphBenchSpec,
        TRANSITION_LANE_SPEC as TransitionLaneSpec,
    )

    mifrost = mifrost_module
    HGraphEncoder = HGraphEncoderCls
    HorizonEncoder = HorizonEncoderCls
    TransitionEffectsHGraphEncoder = TransitionEffectsHGraphEncoderCls
    TransitionHGraphEncoder = TransitionHGraphEncoderCls
    HGRAPH_BENCH_SPEC = HGraphBenchSpec
    TRANSITION_LANE_SPEC = TransitionLaneSpec


@dataclass
class BenchSummary:
    suite: str
    scenario: str
    path: str
    n_items: int
    repeats: int
    warmup: int
    total_s_mean: float
    total_s_median: float
    total_s_min: float
    total_s_max: float
    total_s_stdev: float
    per_item_ms_median: float
    items_per_s_median: float
    context: dict[str, Any]


def _parse_int_list(value: str) -> list[int]:
    return [int(v.strip()) for v in value.split(",") if v.strip()]


def _parse_str_list(value: str) -> list[str]:
    return [v.strip() for v in value.split(",") if v.strip()]


def _parse_bool(value: str) -> bool:
    value = value.strip().lower()
    if value in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise ValueError(f"invalid bool value: {value}")


def _parse_bool_list(value: str) -> list[bool]:
    return [_parse_bool(v) for v in value.split(",") if v.strip()]


def _advanced_state(state: StateT) -> Any:
    return getattr(state, "_advanced_state", state)


def _advanced_action(action: ActionT) -> Any:
    return getattr(action, "_advanced_ground_action", action)


def _state_key(state: StateT) -> str:
    return str(state)


def _load_problem(domain: str, problem: str, mode: str) -> tuple[Any, Any]:
    domain_path = ROOT / "data" / "pddl" / domain / "domain.pddl"
    problem_path = ROOT / "data" / "pddl" / domain / f"{problem}.pddl"
    domain_obj = pymimir.Domain(domain_path)
    problem_obj = pymimir.Problem(domain_obj, problem_path, mode=mode)
    return domain_obj, problem_obj


def _collect_state_pool(
    root: StateT,
    *,
    max_states: int,
    max_branch: int,
) -> tuple[list[StateT], list[tuple[StateT, ActionT, StateT]]]:
    states: list[StateT] = [root]
    transitions: list[tuple[StateT, ActionT, StateT]] = []

    seen: dict[str, StateT] = {_state_key(root): root}
    transition_seen: set[tuple[str, str]] = set()
    queue: deque[StateT] = deque([root])

    while queue and len(states) < max_states:
        state = queue.popleft()
        actions = list(state.generate_applicable_actions())
        if max_branch > 0:
            actions = actions[:max_branch]
        for action in actions:
            successor = action.apply(state)
            sk = _state_key(state)
            tk = _state_key(successor)
            pair = (sk, tk)
            if pair not in transition_seen:
                transition_seen.add(pair)
                transitions.append((state, action, successor))
            if tk not in seen:
                seen[tk] = successor
                states.append(successor)
                queue.append(successor)
                if len(states) >= max_states:
                    break

    return states, transitions


def _pick_cycle(items: Sequence[_T], n: int) -> list[_T]:
    if n <= 0:
        return []
    if not items:
        return []
    if n <= len(items):
        return list(items[:n])
    out: list[_T] = []
    i = 0
    while len(out) < n:
        out.append(items[i % len(items)])
        i += 1
    return out


def _goal_literals(problem_obj: Any) -> list[Any]:
    return list(problem_obj.get_goal_condition().get_literals())


def _subgoal_layers_from_goals(goals: Sequence[Any], width: int = 2) -> list[list[Any]]:
    if not goals:
        return []
    return [list(goals[: min(width, len(goals))])]


def _actions_for_states(
    states: Sequence[StateT], max_actions_per_state: int
) -> list[list[Any]]:
    actions_by_state: list[list[Any]] = []
    for state in states:
        actions = list(state.generate_applicable_actions())
        if max_actions_per_state > 0:
            actions = actions[:max_actions_per_state]
        actions_by_state.append(actions)
    return actions_by_state


def _build_dag(
    root: StateT,
    *,
    max_nodes: int,
    max_branch: int,
    include_actions: bool,
) -> tuple[Any, int, int]:
    dag = mifrost.TransitionDAG(_advanced_state(root))
    queue: deque[StateT] = deque([root])
    seen: dict[str, StateT] = {_state_key(root): root}
    edge_seen: set[tuple[str, str]] = set()

    while queue and len(seen) < max_nodes:
        state = queue.popleft()
        actions = list(state.generate_applicable_actions())
        if max_branch > 0:
            actions = actions[:max_branch]
        for action in actions:
            successor = action.apply(state)
            sk = _state_key(state)
            tk = _state_key(successor)
            edge_key = (sk, tk)
            if edge_key not in edge_seen:
                edge_seen.add(edge_key)
                if include_actions:
                    dag.register_transition(
                        _advanced_state(state),
                        _advanced_state(successor),
                        _advanced_action(action),
                    )
                else:
                    dag.register_transition(
                        _advanced_state(state),
                        _advanced_state(successor),
                    )
            if tk not in seen and len(seen) < max_nodes:
                seen[tk] = successor
                queue.append(successor)

    return dag, len(dag.nodes()), len(dag.transitions())


def _benchmark(
    fn: Callable[[], None],
    *,
    warmup: int,
    repeats: int,
) -> list[float]:
    for _ in range(max(0, warmup)):
        fn()
    durations: list[float] = []
    for _ in range(max(1, repeats)):
        start = time.perf_counter()
        fn()
        durations.append(time.perf_counter() - start)
    return durations


def _benchmark_prepared(
    prepare: Callable[[], Any],
    action: Callable[[Any], None],
    *,
    warmup: int,
    repeats: int,
) -> list[float]:
    for _ in range(max(0, warmup)):
        prepared = prepare()
        action(prepared)
    durations: list[float] = []
    for _ in range(max(1, repeats)):
        prepared = prepare()
        start = time.perf_counter()
        action(prepared)
        durations.append(time.perf_counter() - start)
    return durations


def _summarize(
    *,
    suite: str,
    scenario: str,
    path: str,
    durations: list[float],
    n_items: int,
    repeats: int,
    warmup: int,
    context: dict[str, Any],
) -> BenchSummary:
    mean_s = statistics.fmean(durations)
    median_s = statistics.median(durations)
    stdev_s = statistics.stdev(durations) if len(durations) > 1 else 0.0
    per_item_ms = (median_s / max(1, n_items)) * 1000.0
    items_per_s = n_items / median_s if median_s > 0 else float("inf")
    return BenchSummary(
        suite=suite,
        scenario=scenario,
        path=path,
        n_items=n_items,
        repeats=repeats,
        warmup=warmup,
        total_s_mean=mean_s,
        total_s_median=median_s,
        total_s_min=min(durations),
        total_s_max=max(durations),
        total_s_stdev=stdev_s,
        per_item_ms_median=per_item_ms,
        items_per_s_median=items_per_s,
        context=context,
    )


def _run_hgraph_suite(
    *,
    domain_obj: Any,
    problem_obj: Any,
    states_pool: Sequence[StateT],
    batch_sizes: Sequence[int],
    include_lgan_values: Sequence[bool],
    max_actions_per_state: int,
    benchmark_pyg: bool,
    warmup: int,
    repeats: int,
) -> list[BenchSummary]:
    goals = _goal_literals(problem_obj)
    subgoal_layers = _subgoal_layers_from_goals(goals)
    results: list[BenchSummary] = []

    scenarios = [
        ("state_only", False, False, False),
        ("goals_only", True, False, False),
        ("goals_actions", True, True, False),
        ("goals_subgoals_actions", True, True, True),
    ]

    for include_lgan in include_lgan_values:
        for batch_size in batch_sizes:
            states = _pick_cycle(states_pool, batch_size)
            actions_by_state = _actions_for_states(states, max_actions_per_state)

            for (
                scenario_name,
                include_goals,
                include_actions,
                include_subgoals,
            ) in scenarios:
                if include_lgan and not HGRAPH_BENCH_SPEC.allows_lgan(
                    include_actions=include_actions
                ):
                    continue
                encoder = HGraphEncoder(
                    domain_obj,
                    ignore_actions=not include_actions,
                    include_lgan_edges=include_lgan,
                    support_literals=include_goals or include_subgoals,
                )

                goals_arg = goals if include_goals else None
                subgoals_arg = subgoal_layers if include_subgoals else None
                actions_arg = actions_by_state if include_actions else None

                def _single_native() -> None:
                    for idx, state in enumerate(states):
                        encoder.encode(
                            state,
                            goals=goals_arg,
                            actions=actions_by_state[idx] if include_actions else None,
                            subgoal_layers=subgoals_arg,
                        )

                def _batch_native() -> None:
                    encoder.encode_batch(
                        states,
                        goals=goals_arg,
                        actions=actions_arg,
                        subgoal_layers=subgoals_arg,
                    )

                def _stream_native() -> None:
                    stream = encoder.stream()
                    for idx, state in enumerate(states):
                        stream.append(
                            state,
                            goals=goals_arg,
                            actions=actions_by_state[idx] if include_actions else None,
                            subgoal_layers=subgoals_arg,
                        )
                    stream.flush()

                def _prepare_stream() -> Any:
                    stream = encoder.stream()
                    for idx, state in enumerate(states):
                        stream.append(
                            state,
                            goals=goals_arg,
                            actions=actions_by_state[idx] if include_actions else None,
                            subgoal_layers=subgoals_arg,
                        )
                    return stream

                context = {
                    "batch_size": batch_size,
                    "include_lgan": include_lgan,
                    "include_goals": include_goals,
                    "include_actions": include_actions,
                    "include_subgoals": include_subgoals,
                }

                for path_name, fn in [
                    ("single_native", _single_native),
                    ("batch_native", _batch_native),
                    ("stream_native", _stream_native),
                ]:
                    durations = _benchmark(fn, warmup=warmup, repeats=repeats)
                    results.append(
                        _summarize(
                            suite="hgraph",
                            scenario=scenario_name,
                            path=path_name,
                            durations=durations,
                            n_items=len(states),
                            repeats=repeats,
                            warmup=warmup,
                            context=context,
                        )
                    )

                durations = _benchmark_prepared(
                    _prepare_stream,
                    lambda stream: stream.flush(),
                    warmup=warmup,
                    repeats=repeats,
                )
                results.append(
                    _summarize(
                        suite="hgraph",
                        scenario=scenario_name,
                        path="stream_flush_native",
                        durations=durations,
                        n_items=len(states),
                        repeats=repeats,
                        warmup=warmup,
                        context=context,
                    )
                )

                durations = _benchmark_prepared(
                    lambda: encoder.encode_batch(
                        states,
                        goals=goals_arg,
                        actions=actions_arg,
                        subgoal_layers=subgoals_arg,
                    ),
                    lambda encoding: encoding.as_pyg(as_batch=True),
                    warmup=warmup,
                    repeats=repeats,
                )
                results.append(
                    _summarize(
                        suite="hgraph",
                        scenario=scenario_name,
                        path="batch_as_pyg",
                        durations=durations,
                        n_items=len(states),
                        repeats=repeats,
                        warmup=warmup,
                        context=context,
                    )
                )

                if benchmark_pyg:

                    def _single_pyg() -> None:
                        for idx, state in enumerate(states):
                            encoder.encode_pyg(
                                state,
                                goals=goals_arg,
                                actions=(
                                    actions_by_state[idx] if include_actions else None
                                ),
                                subgoal_layers=subgoals_arg,
                            )

                    def _batch_pyg() -> None:
                        encoder.encode_batch_pyg(
                            states,
                            goals=goals_arg,
                            actions=actions_arg,
                            subgoal_layers=subgoals_arg,
                        )

                    def _stream_pyg() -> None:
                        stream = encoder.stream()
                        for idx, state in enumerate(states):
                            stream.append(
                                state,
                                goals=goals_arg,
                                actions=(
                                    actions_by_state[idx] if include_actions else None
                                ),
                                subgoal_layers=subgoals_arg,
                            )
                        stream.flush_pyg(as_batch=True)

                    for path_name, fn in [
                        ("single_pyg", _single_pyg),
                        ("batch_pyg", _batch_pyg),
                        ("stream_pyg", _stream_pyg),
                    ]:
                        durations = _benchmark(fn, warmup=warmup, repeats=repeats)
                        results.append(
                            _summarize(
                                suite="hgraph",
                                scenario=scenario_name,
                                path=path_name,
                                durations=durations,
                                n_items=len(states),
                                repeats=repeats,
                                warmup=warmup,
                                context=context,
                            )
                        )

                    durations = _benchmark_prepared(
                        _prepare_stream,
                        lambda stream: stream.flush_pyg(as_batch=True),
                        warmup=warmup,
                        repeats=repeats,
                    )
                    results.append(
                        _summarize(
                            suite="hgraph",
                            scenario=scenario_name,
                            path="stream_flush_pyg",
                            durations=durations,
                            n_items=len(states),
                            repeats=repeats,
                            warmup=warmup,
                            context=context,
                        )
                    )

    return results


def _run_transition_suite(
    *,
    domain_obj: Any,
    problem_obj: Any,
    transitions_pool: Sequence[tuple[StateT, ActionT, StateT]],
    batch_sizes: Sequence[int],
    include_lgan_values: Sequence[bool],
    benchmark_pyg: bool,
    warmup: int,
    repeats: int,
) -> list[BenchSummary]:
    goals = _goal_literals(problem_obj)
    results: list[BenchSummary] = []

    transition_pairs = [
        (current, successor) for current, _, successor in transitions_pool
    ]
    if not transition_pairs:
        return results

    scenario_defs = [
        ("full_goals", TransitionHGraphEncoder, True),
        ("full_no_goals", TransitionHGraphEncoder, False),
        ("delta_goals", TransitionEffectsHGraphEncoder, True),
        ("delta_no_goals", TransitionEffectsHGraphEncoder, False),
    ]

    for include_lgan in include_lgan_values:
        for batch_size in batch_sizes:
            if include_lgan and not TRANSITION_LANE_SPEC.allows_lgan(
                include_actions=False
            ):
                continue
            pairs = _pick_cycle(transition_pairs, batch_size)
            current_states = [pair[0] for pair in pairs]
            successor_states = [pair[1] for pair in pairs]

            for scenario_name, encoder_cls, include_goals in scenario_defs:
                encoder = encoder_cls(domain_obj, include_lgan_edges=include_lgan)
                goals_arg = goals if include_goals else None

                def _single_native() -> None:
                    for current, successor in zip(current_states, successor_states):
                        encoder.encode(current, successor=successor, goals=goals_arg)

                def _batch_native() -> None:
                    encoder.encode_batch(
                        current_states,
                        successors=successor_states,
                        goals=goals_arg,
                    )

                def _stream_native() -> None:
                    stream = encoder.stream()
                    for current, successor in zip(current_states, successor_states):
                        stream.append(current, successor, goals=goals_arg)
                    stream.flush()

                def _prepare_stream() -> Any:
                    stream = encoder.stream()
                    for current, successor in zip(current_states, successor_states):
                        stream.append(current, successor, goals=goals_arg)
                    return stream

                context = {
                    "batch_size": batch_size,
                    "include_lgan": include_lgan,
                    "include_goals": include_goals,
                }

                for path_name, fn in [
                    ("single_native", _single_native),
                    ("batch_native", _batch_native),
                    ("stream_native", _stream_native),
                ]:
                    durations = _benchmark(fn, warmup=warmup, repeats=repeats)
                    results.append(
                        _summarize(
                            suite="transition",
                            scenario=scenario_name,
                            path=path_name,
                            durations=durations,
                            n_items=len(current_states),
                            repeats=repeats,
                            warmup=warmup,
                            context=context,
                        )
                    )

                durations = _benchmark_prepared(
                    _prepare_stream,
                    lambda stream: stream.flush(),
                    warmup=warmup,
                    repeats=repeats,
                )
                results.append(
                    _summarize(
                        suite="transition",
                        scenario=scenario_name,
                        path="stream_flush_native",
                        durations=durations,
                        n_items=len(current_states),
                        repeats=repeats,
                        warmup=warmup,
                        context=context,
                    )
                )

                durations = _benchmark_prepared(
                    lambda: encoder.encode_batch(
                        current_states,
                        successors=successor_states,
                        goals=goals_arg,
                    ),
                    lambda encoding: encoding.as_pyg(as_batch=True),
                    warmup=warmup,
                    repeats=repeats,
                )
                results.append(
                    _summarize(
                        suite="transition",
                        scenario=scenario_name,
                        path="batch_as_pyg",
                        durations=durations,
                        n_items=len(current_states),
                        repeats=repeats,
                        warmup=warmup,
                        context=context,
                    )
                )

                if benchmark_pyg:

                    def _single_pyg() -> None:
                        for current, successor in zip(current_states, successor_states):
                            encoder.encode_pyg(
                                current,
                                successor=successor,
                                goals=goals_arg,
                            )

                    def _batch_pyg() -> None:
                        encoder.encode_batch_pyg(
                            current_states,
                            successors=successor_states,
                            goals=goals_arg,
                        )

                    def _stream_pyg() -> None:
                        stream = encoder.stream()
                        for current, successor in zip(current_states, successor_states):
                            stream.append(current, successor, goals=goals_arg)
                        stream.flush_pyg(as_batch=True)

                    for path_name, fn in [
                        ("single_pyg", _single_pyg),
                        ("batch_pyg", _batch_pyg),
                        ("stream_pyg", _stream_pyg),
                    ]:
                        durations = _benchmark(fn, warmup=warmup, repeats=repeats)
                        results.append(
                            _summarize(
                                suite="transition",
                                scenario=scenario_name,
                                path=path_name,
                                durations=durations,
                                n_items=len(current_states),
                                repeats=repeats,
                                warmup=warmup,
                                context=context,
                            )
                        )

                    durations = _benchmark_prepared(
                        _prepare_stream,
                        lambda stream: stream.flush_pyg(as_batch=True),
                        warmup=warmup,
                        repeats=repeats,
                    )
                    results.append(
                        _summarize(
                            suite="transition",
                            scenario=scenario_name,
                            path="stream_flush_pyg",
                            durations=durations,
                            n_items=len(current_states),
                            repeats=repeats,
                            warmup=warmup,
                            context=context,
                        )
                    )

    return results


def _run_horizon_suite(
    *,
    domain_obj: Any,
    roots_pool: Sequence[StateT],
    dag_sizes: Sequence[int],
    horizon_batch_sizes: Sequence[int],
    include_lgan_values: Sequence[bool],
    max_branch: int,
    benchmark_pyg: bool,
    warmup: int,
    repeats: int,
) -> list[BenchSummary]:
    results: list[BenchSummary] = []
    if not roots_pool:
        return results

    scenario_defs = [
        ("full_goals_actions", mifrost.HorizonEncoderMode.full, True, True),
        ("full_goals_only", mifrost.HorizonEncoderMode.full, True, False),
        ("delta_goals_only", mifrost.HorizonEncoderMode.delta, True, False),
        ("action_goals_actions", mifrost.HorizonEncoderMode.action, True, True),
    ]

    for include_lgan in include_lgan_values:
        for dag_size in dag_sizes:
            for n_roots in horizon_batch_sizes:
                roots = _pick_cycle(roots_pool, n_roots)
                dags: list[Any] = []
                dag_nodes: list[int] = []
                dag_edges: list[int] = []
                for root in roots:
                    dag, n_nodes, n_edges = _build_dag(
                        root,
                        max_nodes=dag_size,
                        max_branch=max_branch,
                        include_actions=True,
                    )
                    dags.append(dag)
                    dag_nodes.append(n_nodes)
                    dag_edges.append(n_edges)

                for (
                    scenario_name,
                    mode,
                    include_goals,
                    include_actions,
                ) in scenario_defs:
                    encoder = HorizonEncoder(
                        domain_obj,
                        transition_mode=mode,
                        include_lgan_edges=include_lgan,
                        ignore_actions=not include_actions,
                    )
                    goals_arg = (
                        _goal_literals(roots[0].get_problem())
                        if include_goals
                        else None
                    )

                    def _single_native() -> None:
                        for root, dag in zip(roots, dags):
                            encoder.encode(root, dag=dag, goals=goals_arg)

                    def _batch_native() -> None:
                        encoder.encode_batch(roots, dags=dags, goals=goals_arg)

                    def _stream_native() -> None:
                        stream = encoder.stream()
                        for root, dag in zip(roots, dags):
                            stream.append(root, dag=dag, goals=goals_arg)
                        stream.flush()

                    def _prepare_stream() -> Any:
                        stream = encoder.stream()
                        for root, dag in zip(roots, dags):
                            stream.append(root, dag=dag, goals=goals_arg)
                        return stream

                    context = {
                        "n_roots": n_roots,
                        "requested_dag_size": dag_size,
                        "avg_dag_nodes": (
                            statistics.fmean(dag_nodes) if dag_nodes else 0.0
                        ),
                        "avg_dag_edges": (
                            statistics.fmean(dag_edges) if dag_edges else 0.0
                        ),
                        "include_lgan": include_lgan,
                        "mode": str(mode),
                        "include_goals": include_goals,
                        "include_actions": include_actions,
                    }

                    for path_name, fn in [
                        ("single_native", _single_native),
                        ("batch_native", _batch_native),
                        ("stream_native", _stream_native),
                    ]:
                        durations = _benchmark(fn, warmup=warmup, repeats=repeats)
                        results.append(
                            _summarize(
                                suite="horizon",
                                scenario=scenario_name,
                                path=path_name,
                                durations=durations,
                                n_items=len(roots),
                                repeats=repeats,
                                warmup=warmup,
                                context=context,
                            )
                        )

                    durations = _benchmark_prepared(
                        _prepare_stream,
                        lambda stream: stream.flush(),
                        warmup=warmup,
                        repeats=repeats,
                    )
                    results.append(
                        _summarize(
                            suite="horizon",
                            scenario=scenario_name,
                            path="stream_flush_native",
                            durations=durations,
                            n_items=len(roots),
                            repeats=repeats,
                            warmup=warmup,
                            context=context,
                        )
                    )

                    durations = _benchmark_prepared(
                        lambda: encoder.encode_batch(roots, dags=dags, goals=goals_arg),
                        lambda encoding: encoding.as_pyg(as_batch=True),
                        warmup=warmup,
                        repeats=repeats,
                    )
                    results.append(
                        _summarize(
                            suite="horizon",
                            scenario=scenario_name,
                            path="batch_as_pyg",
                            durations=durations,
                            n_items=len(roots),
                            repeats=repeats,
                            warmup=warmup,
                            context=context,
                        )
                    )

                    if benchmark_pyg:

                        def _single_pyg() -> None:
                            for root, dag in zip(roots, dags):
                                encoder.encode_pyg(root, dag=dag, goals=goals_arg)

                        def _batch_pyg() -> None:
                            encoder.encode_batch_pyg(roots, dags=dags, goals=goals_arg)

                        def _stream_pyg() -> None:
                            stream = encoder.stream()
                            for root, dag in zip(roots, dags):
                                stream.append(root, dag=dag, goals=goals_arg)
                            stream.flush_pyg(as_batch=True)

                        for path_name, fn in [
                            ("single_pyg", _single_pyg),
                            ("batch_pyg", _batch_pyg),
                            ("stream_pyg", _stream_pyg),
                        ]:
                            durations = _benchmark(fn, warmup=warmup, repeats=repeats)
                            results.append(
                                _summarize(
                                    suite="horizon",
                                    scenario=scenario_name,
                                    path=path_name,
                                    durations=durations,
                                    n_items=len(roots),
                                    repeats=repeats,
                                    warmup=warmup,
                                    context=context,
                                )
                            )

                        durations = _benchmark_prepared(
                            _prepare_stream,
                            lambda stream: stream.flush_pyg(as_batch=True),
                            warmup=warmup,
                            repeats=repeats,
                        )
                        results.append(
                            _summarize(
                                suite="horizon",
                                scenario=scenario_name,
                                path="stream_flush_pyg",
                                durations=durations,
                                n_items=len(roots),
                                repeats=repeats,
                                warmup=warmup,
                                context=context,
                            )
                        )

    return results


def _print_table(results: Sequence[BenchSummary]) -> None:
    path_width = 22
    header = (
        "suite".ljust(11)
        + "scenario".ljust(27)
        + "path".ljust(path_width)
        + "lgan".ljust(7)
        + "dag".rjust(6)
        + "items".rjust(7)
        + "median ms/item".rjust(16)
        + "items/s".rjust(12)
    )
    print(header)
    print("-" * len(header))
    for row in sorted(
        results,
        key=lambda r: (
            r.suite,
            r.scenario,
            r.path,
            int(bool(r.context.get("include_lgan", False))),
            int(r.context.get("requested_dag_size", 0)),
            r.context.get("batch_size", r.context.get("n_roots", 0)),
        ),
    ):
        include_lgan = (
            "true" if bool(row.context.get("include_lgan", False)) else "false"
        )
        dag_size = row.context.get("requested_dag_size")
        dag_cell = "-" if dag_size is None else str(int(dag_size))
        print(
            row.suite.ljust(11)
            + row.scenario.ljust(27)
            + row.path.ljust(path_width)
            + include_lgan.ljust(7)
            + dag_cell.rjust(6)
            + str(row.n_items).rjust(7)
            + f"{row.per_item_ms_median:16.4f}"
            + f"{row.items_per_s_median:12.2f}"
        )


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "End-to-end encoder benchmarking suite with diverse state batches, "
            "input combinations, and horizon DAG scaling."
        )
    )
    parser.add_argument("--domain", default="blocks")
    parser.add_argument("--problem", default="probBLOCKS-8-1")
    parser.add_argument("--mode", choices=("grounded", "lifted"), default="grounded")
    parser.add_argument(
        "--import-mode",
        choices=("installed", "source"),
        default="installed",
    )
    parser.add_argument("--max-states", type=int, default=256)
    parser.add_argument("--max-branch", type=int, default=4)
    parser.add_argument("--max-actions-per-state", type=int, default=4)
    parser.add_argument("--batch-sizes", default="8,32,128")
    parser.add_argument("--horizon-dag-sizes", default="8,32,64")
    parser.add_argument("--horizon-batch-sizes", default="4,16")
    parser.add_argument("--horizon-root-count", type=int, default=32)
    parser.add_argument("--include-lgan-values", default="false,true")
    parser.add_argument(
        "--families",
        default="hgraph,transition,horizon",
        help="Comma-separated: hgraph,transition,horizon",
    )
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--benchmark-pyg", action="store_true")
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args(argv)

    families = set(_parse_str_list(args.families))
    include_lgan_values = _parse_bool_list(args.include_lgan_values)
    batch_sizes = _parse_int_list(args.batch_sizes)
    horizon_dag_sizes = _parse_int_list(args.horizon_dag_sizes)
    horizon_batch_sizes = _parse_int_list(args.horizon_batch_sizes)

    _import_runtime(args.import_mode)
    domain_obj, problem_obj = _load_problem(args.domain, args.problem, args.mode)
    root = problem_obj.get_initial_state()
    states_pool, transitions_pool = _collect_state_pool(
        root, max_states=args.max_states, max_branch=args.max_branch
    )
    roots_pool = _pick_cycle(states_pool, args.horizon_root_count)

    results: list[BenchSummary] = []
    if "hgraph" in families:
        results.extend(
            _run_hgraph_suite(
                domain_obj=domain_obj,
                problem_obj=problem_obj,
                states_pool=states_pool,
                batch_sizes=batch_sizes,
                include_lgan_values=include_lgan_values,
                max_actions_per_state=args.max_actions_per_state,
                benchmark_pyg=args.benchmark_pyg,
                warmup=args.warmup,
                repeats=args.repeats,
            )
        )
    if "transition" in families:
        results.extend(
            _run_transition_suite(
                domain_obj=domain_obj,
                problem_obj=problem_obj,
                transitions_pool=transitions_pool,
                batch_sizes=batch_sizes,
                include_lgan_values=include_lgan_values,
                benchmark_pyg=args.benchmark_pyg,
                warmup=args.warmup,
                repeats=args.repeats,
            )
        )
    if "horizon" in families:
        results.extend(
            _run_horizon_suite(
                domain_obj=domain_obj,
                roots_pool=roots_pool,
                dag_sizes=horizon_dag_sizes,
                horizon_batch_sizes=horizon_batch_sizes,
                include_lgan_values=include_lgan_values,
                max_branch=args.max_branch,
                benchmark_pyg=args.benchmark_pyg,
                warmup=args.warmup,
                repeats=args.repeats,
            )
        )

    print(
        f"Collected {len(states_pool)} states and {len(transitions_pool)} unique transitions "
        f"from domain={args.domain} problem={args.problem}."
    )
    print(f"Produced {len(results)} benchmark rows.")
    _print_table(results)

    if args.output_json is not None:
        payload = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "config": {
                "domain": args.domain,
                "problem": args.problem,
                "mode": args.mode,
                "import_mode": args.import_mode,
                "max_states": args.max_states,
                "max_branch": args.max_branch,
                "max_actions_per_state": args.max_actions_per_state,
                "batch_sizes": batch_sizes,
                "horizon_dag_sizes": horizon_dag_sizes,
                "horizon_batch_sizes": horizon_batch_sizes,
                "horizon_root_count": args.horizon_root_count,
                "include_lgan_values": include_lgan_values,
                "families": sorted(families),
                "benchmark_pyg": args.benchmark_pyg,
                "warmup": args.warmup,
                "repeats": args.repeats,
            },
            "results": [asdict(row) for row in results],
        }
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2))
        print(f"Wrote JSON report: {args.output_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
