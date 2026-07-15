from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
from typing import Any

import mifrost

from benchmark_flat_backends import (
    _cycle,
    _measure,
    _pymimir_inputs,
    _pytyr_inputs,
    _stream_batch,
)


def _backend_specs(
    backend: str,
    encoder: Any,
    states: list[Any],
    goals: list[Any],
) -> list[tuple[str, str, int, Any]]:
    size = len(states)
    return [
        (
            backend,
            "batch_default",
            size,
            lambda: encoder.encode_batch(states),
        ),
        (
            backend,
            "batch_explicit_goals",
            size,
            lambda: encoder.encode_batch(states, goals=goals),
        ),
        (
            backend,
            "single_loop",
            size,
            lambda: [encoder.encode(state) for state in states],
        ),
        (
            backend,
            "stream_end_to_end",
            size,
            lambda: _stream_batch(encoder, states),
        ),
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", default="blocks")
    parser.add_argument("--problem", default="smedium")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-states", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--repeats", type=int, default=200)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--pymimir-only", action="store_true")
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()

    pymimir_domain, pymimir_problem, pymimir_pool = _pymimir_inputs(
        args.domain, args.problem, args.max_states
    )
    pymimir_states = _cycle(pymimir_pool, args.batch_size)
    pymimir_goals = list(pymimir_problem.get_goal_condition().get_literals())
    pymimir_encoder = mifrost.HGraphEncoder(pymimir_domain)
    specs_by_backend = {
        "pymimir": _backend_specs(
            "pymimir", pymimir_encoder, pymimir_states, pymimir_goals
        )
    }
    state_pool_sizes = {"pymimir": len(pymimir_pool)}

    if not args.pymimir_only:
        pytyr_task, pytyr_reader, pytyr_pool = _pytyr_inputs(
            args.domain, args.problem, args.max_states
        )
        pytyr_states = _cycle(pytyr_pool, args.batch_size)
        pytyr_goals = list(pytyr_reader.problem_snapshot().goals)
        pytyr_encoder = mifrost.HGraphEncoder(pytyr_task)
        specs_by_backend["pytyr"] = _backend_specs(
            "pytyr", pytyr_encoder, pytyr_states, pytyr_goals
        )
        state_pool_sizes["pytyr"] = len(pytyr_pool)

    rounds = []
    backend_order = list(specs_by_backend)
    for round_index in range(args.rounds):
        order = backend_order if round_index % 2 == 0 else list(reversed(backend_order))
        results = []
        for backend in order:
            specs = specs_by_backend[backend]
            path_order = specs if round_index % 2 == 0 else list(reversed(specs))
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
                "round": round_index + 1,
                "backend_order": order,
                "results": [asdict(result) for result in results],
            }
        )

    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "mifrost_version": getattr(mifrost, "__version__", "unknown"),
        "mifrost_module": str(Path(mifrost.__file__).resolve()),
        "domain": args.domain,
        "problem": args.problem,
        "batch_size": args.batch_size,
        "max_states": args.max_states,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "round_count": args.rounds,
        "state_pool_sizes": state_pool_sizes,
        "rounds": rounds,
    }
    report = json.dumps(payload, indent=2)
    print(report)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(report + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
