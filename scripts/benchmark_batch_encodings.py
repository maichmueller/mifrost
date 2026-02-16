from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import torch


@dataclass
class ScenarioResult:
    scenario: str
    batch_size: int
    repeats: int
    warmup: int
    mean_ms: float
    median_ms: float
    stdev_ms: float
    min_ms: float
    max_ms: float


def _bench(
    fn: Callable[[], Any],
    *,
    warmup: int,
    repeats: int,
    timer_ns: Callable[[], int],
    inner_iters: int,
) -> list[float]:
    inner_iters = max(1, int(inner_iters))

    for _ in range(max(0, warmup)):
        for __ in range(inner_iters):
            fn()

    durations_ms: list[float] = []
    for _ in range(max(1, repeats)):
        t0 = timer_ns()
        last = None
        for __ in range(inner_iters):
            last = fn()
        _ = last  # keep the result alive until after timing ends
        durations_ms.append(((timer_ns() - t0) / 1e6) / inner_iters)
    return durations_ms


def _summary(
    *,
    scenario: str,
    durations_ms: list[float],
    batch_size: int,
    repeats: int,
    warmup: int,
) -> ScenarioResult:
    return ScenarioResult(
        scenario=scenario,
        batch_size=batch_size,
        repeats=repeats,
        warmup=warmup,
        mean_ms=statistics.fmean(durations_ms),
        median_ms=statistics.median(durations_ms),
        stdev_ms=statistics.pstdev(durations_ms),
        min_ms=min(durations_ms),
        max_ms=max(durations_ms),
    )


def _fill_graph(builder: Any, seed: int, *, node_count: int, feature_dim: int) -> None:
    x = torch.arange(node_count * feature_dim, dtype=torch.float32).reshape(
        node_count, feature_dim
    ) + float(seed)
    builder.add_node_features("atom", "x", x)

    if node_count >= 2:
        src = torch.arange(node_count - 1, dtype=torch.int64)
        dst = src + 1
        builder.add_edges("atom", "rel", "atom", src, dst)


def _make_plain_encoding(
    mifrost_module: Any, seed: int, *, node_count: int, feature_dim: int
) -> Any:
    b = mifrost_module.BatchBuilder()
    b.set_graph_kind("hetero")
    _fill_graph(b, seed, node_count=node_count, feature_dim=feature_dim)
    b.next_graph()
    return b.build()


def _make_typed_graph_field_encoding(
    mifrost_module: Any, seed: int, *, node_count: int, feature_dim: int
) -> Any:
    b = mifrost_module.BatchBuilder()
    b.set_graph_kind("hetero")
    b.register_graph_field(
        "target_indices",
        {
            "dtype": "i64",
            "mode": "ragged_cat",
            "dim": 1,
            "inc": {"kind": "none"},
        },
    )
    b.register_graph_field(
        "problem_id",
        {
            "dtype": "i64",
            "mode": "const",
            "dim": 1,
            "inc": {"kind": "none"},
        },
    )

    _fill_graph(b, seed, node_count=node_count, feature_dim=feature_dim)
    b.set_graph_field("target_indices", [seed, seed + 1, seed + 2])
    b.set_graph_field("problem_id", 7)
    b.next_graph()
    return b.build()


def _supports_pyobj_attrs(encoding: Any) -> bool:
    return hasattr(encoding, "__dict__")


def _make_pyobj_encoding(
    mifrost_module: Any, seed: int, *, node_count: int, feature_dim: int
) -> Any:
    enc = _make_plain_encoding(
        mifrost_module, seed, node_count=node_count, feature_dim=feature_dim
    )
    enc.reward_signature = torch.tensor([1, 2, 3, 4], dtype=torch.int64)
    enc.targets = [seed, seed + 1, seed + 2]
    return enc


def _print_table(results: list[ScenarioResult], *, gate_stat: str) -> None:
    if not results:
        return

    baseline = next((r for r in results if r.scenario == "plain"), results[0])
    if gate_stat == "median":
        baseline_gate_ms = baseline.median_ms
    else:
        baseline_gate_ms = baseline.mean_ms

    print(
        "scenario                mean_ms   median_ms  stdev_ms  "
        f"overhead_vs_plain({gate_stat})"
    )
    for r in results:
        gate_ms = r.median_ms if gate_stat == "median" else r.mean_ms
        overhead = gate_ms / baseline_gate_ms if baseline_gate_ms > 0 else float("inf")
        print(
            f"{r.scenario:<22} {r.mean_ms:>8.4f}  {r.median_ms:>9.4f}  {r.stdev_ms:>8.4f}  {overhead:>16.3f}"
        )


def _import_mifrost(import_mode: str) -> Any:
    if import_mode == "source":
        root = Path(__file__).resolve().parents[1]
        src = root / "src"
        if str(src) not in sys.path:
            sys.path.insert(0, str(src))
    elif import_mode != "installed":
        raise ValueError(f"Unsupported import mode: {import_mode!r}")

    import mifrost

    return mifrost


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Microbenchmark for mifrost.batch_encodings across plain, typed, and pyobj collation paths."
    )
    parser.add_argument(
        "--clock",
        choices=("wall", "cpu"),
        default="wall",
        help=(
            "Timer source. 'wall' uses perf_counter (subject to noisy neighbors). "
            "'cpu' uses process_time (reduces scheduler interference; best when torch is single-threaded)."
        ),
    )
    parser.add_argument(
        "--gate-stat",
        choices=("mean", "median"),
        default="mean",
        help=(
            "Statistic used for perf gating thresholds. "
            "'median' is more robust to sporadic host interference than 'mean'."
        ),
    )
    parser.add_argument(
        "--inner-iters",
        type=int,
        default=1,
        help=(
            "Number of consecutive batch_encodings() calls per timed iteration. "
            "Reported durations are normalized back to per-call ms. "
            "Increasing this reduces CI noise."
        ),
    )
    parser.add_argument(
        "--scenarios",
        default="plain,typed,pyobj",
        help="Comma-separated scenarios from: plain,typed,pyobj",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--node-count", type=int, default=32)
    parser.add_argument("--feature-dim", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=400)
    parser.add_argument("--json-out", type=str, default="")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--import-mode",
        choices=("installed", "source"),
        default="installed",
        help="How to resolve the mifrost import path.",
    )
    parser.add_argument(
        "--max-overhead-typed",
        type=float,
        default=None,
        help="Fail if typed.(gate_stat) / plain.(gate_stat) exceeds this value.",
    )
    parser.add_argument(
        "--max-overhead-pyobj",
        type=float,
        default=None,
        help="Fail if pyobj.(gate_stat) / plain.(gate_stat) exceeds this value.",
    )
    parser.add_argument(
        "--max-plain-mean-ms",
        type=float,
        default=None,
        help="Fail if plain.(gate_stat) exceeds this value.",
    )
    args = parser.parse_args()
    mifrost_module = _import_mifrost(args.import_mode)

    # Keep the benchmark as single-threaded as possible; otherwise CPU-time based
    # measurement can become misleading and wall-time becomes much noisier across
    # CI runners.
    try:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except Exception:
        # Older torch builds may not support all knobs; don't fail the benchmark.
        pass

    if args.clock == "wall":
        timer_ns: Callable[[], int] = time.perf_counter_ns
    else:
        timer_ns = time.process_time_ns

    requested = [s.strip().lower() for s in args.scenarios.split(",") if s.strip()]
    valid = {"plain", "typed", "pyobj"}
    unknown = [s for s in requested if s not in valid]
    if unknown:
        raise ValueError(f"Unknown scenarios requested: {unknown}")

    enc_plain = [
        _make_plain_encoding(
            mifrost_module, i, node_count=args.node_count, feature_dim=args.feature_dim
        )
        for i in range(args.batch_size)
    ]
    enc_typed = [
        _make_typed_graph_field_encoding(
            mifrost_module, i, node_count=args.node_count, feature_dim=args.feature_dim
        )
        for i in range(args.batch_size)
    ]

    pyobj_supported = _supports_pyobj_attrs(enc_plain[0])
    enc_pyobj = (
        [
            _make_pyobj_encoding(
                mifrost_module,
                i,
                node_count=args.node_count,
                feature_dim=args.feature_dim,
            )
            for i in range(args.batch_size)
        ]
        if pyobj_supported
        else []
    )
    pyobj_specs = {
        "reward_signature": {"dtype": "pyobj", "mode": "const"},
        "targets": {"dtype": "pyobj", "mode": "ragged_cat"},
    }

    fns: dict[str, Callable[[], Any]] = {
        "plain": lambda: mifrost_module.batch_encodings(enc_plain),
        "typed": lambda: mifrost_module.batch_encodings(enc_typed),
    }
    if pyobj_supported:
        fns["pyobj"] = lambda: mifrost_module.batch_encodings(
            enc_pyobj, graph_field_specs=pyobj_specs
        )

    results: list[ScenarioResult] = []
    skipped: list[str] = []

    for scenario in requested:
        fn = fns.get(scenario)
        if fn is None:
            skipped.append(scenario)
            continue
        durations_ms = _bench(
            fn,
            warmup=args.warmup,
            repeats=args.repeats,
            timer_ns=timer_ns,
            inner_iters=args.inner_iters,
        )
        results.append(
            _summary(
                scenario=scenario,
                durations_ms=durations_ms,
                batch_size=args.batch_size,
                repeats=args.repeats,
                warmup=args.warmup,
            )
        )

    payload = {
        "results": [asdict(r) for r in results],
        "skipped": skipped,
        "context": {
            "batch_size": args.batch_size,
            "node_count": args.node_count,
            "feature_dim": args.feature_dim,
            "warmup": args.warmup,
            "repeats": args.repeats,
            "pyobj_supported": pyobj_supported,
            "clock": args.clock,
            "gate_stat": args.gate_stat,
            "inner_iters": args.inner_iters,
        },
    }

    if not args.quiet:
        _print_table(results, gate_stat=args.gate_stat)
        if skipped:
            print(f"skipped: {', '.join(skipped)}")
        print(json.dumps(payload, indent=2))

    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8"
        )

    by_name = {r.scenario: r for r in results}
    plain = by_name.get("plain")
    failures: list[str] = []

    if plain is not None:
        plain_gate_ms = plain.mean_ms if args.gate_stat == "mean" else plain.median_ms
        if (
            args.max_plain_mean_ms is not None
            and plain_gate_ms > args.max_plain_mean_ms
        ):
            failures.append(
                f"plain {args.gate_stat} {plain_gate_ms:.4f}ms > max {args.max_plain_mean_ms:.4f}ms"
            )

        typed = by_name.get("typed")
        if typed is not None and args.max_overhead_typed is not None:
            typed_gate_ms = (
                typed.mean_ms if args.gate_stat == "mean" else typed.median_ms
            )
            typed_overhead = (
                typed_gate_ms / plain_gate_ms if plain_gate_ms > 0 else float("inf")
            )
            if typed_overhead > args.max_overhead_typed:
                failures.append(
                    f"typed overhead {typed_overhead:.3f} > max {args.max_overhead_typed:.3f}"
                )

        pyobj = by_name.get("pyobj")
        if pyobj is not None and args.max_overhead_pyobj is not None:
            pyobj_gate_ms = (
                pyobj.mean_ms if args.gate_stat == "mean" else pyobj.median_ms
            )
            pyobj_overhead = (
                pyobj_gate_ms / plain_gate_ms if plain_gate_ms > 0 else float("inf")
            )
            if pyobj_overhead > args.max_overhead_pyobj:
                failures.append(
                    f"pyobj overhead {pyobj_overhead:.3f} > max {args.max_overhead_pyobj:.3f}"
                )

    if failures:
        for line in failures:
            print(f"PERF REGRESSION: {line}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
