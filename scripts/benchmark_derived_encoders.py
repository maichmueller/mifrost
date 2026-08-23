"""Encoding-speed benchmarks for the derived-graph family and custom toolkit.

Times the native derived-graph facades (full Python ``encode_batch_pyg`` path
versus raw native engine time via ``encoder._runtime.encode_batch``), the
pure-Python `ObjectFeatureEncoder`, a toolkit-driven GraphWriter/NodeTable/
EdgeSink microbenchmark, and raw `StateView.state_facts` throughput on the
``blocks`` smedium and large problems with deterministic state batches.

Also captures ``dumps()`` parity snapshots (hex) so before/after optimization
runs can be compared byte-for-byte.
"""

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
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import pymimir  # noqa: E402

from mifrost.encoders.custom.state_view import StateView  # noqa: E402
from mifrost.encoders.custom.writer import GraphWriter  # noqa: E402
from mifrost.encoders.derived import (  # noqa: E402
    AtomLineGraphEncoder,
    HypergraphIncidenceEncoder,
    ObjectGraphEncoder,
    StarGraphEncoder,
    TransformerBiasEncoder,
    TupleTensorEncoder,
)
from mifrost.encoders.object_feature import ObjectFeatureEncoder  # noqa: E402

PROBLEMS = ("smedium", "large")
BATCH_SIZES = (1, 32, 256)

NATIVE_FACADES = (
    ("StarGraphEncoder", StarGraphEncoder, {}),
    ("ObjectGraphEncoder_clique", ObjectGraphEncoder, {"atom_expansion": "clique"}),
    ("AtomLineGraphEncoder", AtomLineGraphEncoder, {}),
    ("HypergraphIncidenceEncoder", HypergraphIncidenceEncoder, {}),
    ("TupleTensorEncoder", TupleTensorEncoder, {}),
    ("TransformerBiasEncoder", TransformerBiasEncoder, {}),
)


@dataclass
class BenchRow:
    problem: str
    suite: str
    path: str
    batch_size: int
    repeats: int
    warmup: int
    mean_ms: float
    stdev_ms: float
    min_ms: float


def _collect_state_pool(
    root: Any,
    *,
    max_states: int,
    max_branch: int,
) -> list[Any]:
    states: list[Any] = [root]
    seen: set[str] = {str(root)}
    queue: deque[Any] = deque([root])
    while queue and len(states) < max_states:
        state = queue.popleft()
        actions = list(state.generate_applicable_actions())[:max_branch]
        for action in actions:
            successor = action.apply(state)
            key = str(successor)
            if key not in seen:
                seen.add(key)
                states.append(successor)
                queue.append(successor)
                if len(states) >= max_states:
                    break
    return states


def _pick_cycle(items: list[Any], n: int) -> list[Any]:
    if n <= len(items):
        return list(items[:n])
    out: list[Any] = []
    i = 0
    while len(out) < n:
        out.append(items[i % len(items)])
        i += 1
    return out


def _benchmark(
    fn: Callable[[], Any],
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
        durations.append((time.perf_counter() - start) * 1000.0)
    return durations


def _row(
    durations: list[float],
    *,
    problem: str,
    suite: str,
    path: str,
    batch_size: int,
    warmup: int,
    repeats: int,
) -> BenchRow:
    return BenchRow(
        problem=problem,
        suite=suite,
        path=path,
        batch_size=batch_size,
        repeats=repeats,
        warmup=warmup,
        mean_ms=statistics.fmean(durations),
        stdev_ms=statistics.stdev(durations) if len(durations) > 1 else 0.0,
        min_ms=min(durations),
    )


def _toolkit_microbench(view: StateView, n_nodes: int, n_edges: int) -> Any:
    """GraphWriter + NodeTable + EdgeSink alone: add nodes/edges then finish."""
    writer = GraphWriter(view)
    predicates = writer.vocabulary("predicates")
    node_ids = [
        writer.add_node(("object", f"o{index}"), role="object", channels=(index,))
        for index in range(n_nodes)
    ]
    edges = writer.edges
    kinds = ["arg_fwd", "arg_bwd"]
    for index in range(n_edges):
        src = node_ids[index % n_nodes]
        dst = node_ids[(index + 1) % n_nodes]
        kind = kinds[index & 1]
        edges.add(src, dst, kind, index, index)
        predicates.id_for(kind)
    return writer.finish()


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--batch-sizes", default="1,32,256")
    parser.add_argument("--problems", default="smedium,large")
    parser.add_argument("--max-states", type=int, default=256)
    parser.add_argument("--max-branch", type=int, default=4)
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args(argv)

    batch_sizes = [int(v) for v in args.batch_sizes.split(",") if v.strip()]
    problems = [v.strip() for v in args.problems.split(",") if v.strip()]

    results: list[BenchRow] = []
    dumps_snapshot: dict[str, str] = {}

    for problem in problems:
        domain_obj = pymimir.Domain(ROOT / "data" / "pddl" / "blocks" / "domain.pddl")
        problem_obj = pymimir.Problem(
            domain_obj,
            ROOT / "data" / "pddl" / "blocks" / f"{problem}.pddl",
            mode="grounded",
        )
        root = problem_obj.get_initial_state()
        pool = _collect_state_pool(
            root, max_states=args.max_states, max_branch=args.max_branch
        )
        view = None

        for facade_name, facade_cls, kwargs in NATIVE_FACADES:
            encoder = facade_cls(problem_obj, **kwargs)

            # Parity snapshot once per encoder/problem (batch of two).
            snapshot_key = f"native/{facade_name}/{problem}"
            dumps_snapshot[snapshot_key] = (
                encoder.encode_batch([root, root]).dumps().hex()
            )

            for batch_size in batch_sizes:
                states = _pick_cycle(pool, batch_size)

                def _pyg() -> None:
                    encoder.encode_batch_pyg(states)

                def _native() -> None:
                    encoder._runtime.encode_batch(states)

                for path_name, fn in (
                    ("encode_batch_pyg", _pyg),
                    ("runtime_encode_batch", _native),
                ):
                    durations = _benchmark(fn, warmup=args.warmup, repeats=args.repeats)
                    results.append(
                        _row(
                            durations,
                            problem=problem,
                            suite=facade_name,
                            path=path_name,
                            batch_size=batch_size,
                            warmup=args.warmup,
                            repeats=args.repeats,
                        )
                    )

        feature_encoder = ObjectFeatureEncoder(problem_obj)
        view = feature_encoder.view
        dumps_snapshot[f"python/ObjectFeatureEncoder/{problem}"] = (
            feature_encoder.encode_batch([root, root]).dumps().hex()
        )

        for batch_size in batch_sizes:
            states = _pick_cycle(pool, batch_size)

            def _feature_pyg() -> None:
                feature_encoder.encode_batch_pyg(states)

            durations = _benchmark(
                _feature_pyg, warmup=args.warmup, repeats=args.repeats
            )
            results.append(
                _row(
                    durations,
                    problem=problem,
                    suite="ObjectFeatureEncoder_all_flags",
                    path="encode_batch_pyg",
                    batch_size=batch_size,
                    warmup=args.warmup,
                    repeats=args.repeats,
                )
            )

            def _state_facts() -> None:
                for state in states:
                    view.state_facts(state)

            durations = _benchmark(
                _state_facts, warmup=args.warmup, repeats=args.repeats
            )
            results.append(
                _row(
                    durations,
                    problem=problem,
                    suite="StateView.state_facts",
                    path="state_facts_loop",
                    batch_size=batch_size,
                    warmup=args.warmup,
                    repeats=args.repeats,
                )
            )

        # Toolkit microbenchmark: pure NodeTable/EdgeSink/Vocabulary/finish work.
        for batch_size in batch_sizes:

            def _toolkit() -> None:
                for _ in range(batch_size):
                    _toolkit_microbench(
                        view,
                        n_nodes=len(view.objects),
                        n_edges=64,
                    )

            durations = _benchmark(_toolkit, warmup=args.warmup, repeats=args.repeats)
            results.append(
                _row(
                    durations,
                    problem=problem,
                    suite="ToolkitGraphWriter_micro",
                    path="add_node_add_edge_finish",
                    batch_size=batch_size,
                    warmup=args.warmup,
                    repeats=args.repeats,
                )
            )

    header = (
        "problem".ljust(9)
        + "suite".ljust(30)
        + "path".ljust(22)
        + "batch".rjust(6)
        + "mean ms".rjust(12)
        + "stdev".rjust(10)
        + "min".rjust(10)
    )
    print(header)
    print("-" * len(header))
    for row in results:
        print(
            row.problem.ljust(9)
            + row.suite.ljust(30)
            + row.path.ljust(22)
            + str(row.batch_size).rjust(6)
            + f"{row.mean_ms:12.3f}"
            + f"{row.stdev_ms:10.3f}"
            + f"{row.min_ms:10.3f}"
        )

    if args.output_json is not None:
        payload = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "config": {
                "problems": problems,
                "batch_sizes": batch_sizes,
                "max_states": args.max_states,
                "max_branch": args.max_branch,
                "warmup": args.warmup,
                "repeats": args.repeats,
            },
            "results": [asdict(row) for row in results],
            "dumps_snapshot_hex": dumps_snapshot,
        }
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2))
        print(f"Wrote JSON report: {args.output_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
