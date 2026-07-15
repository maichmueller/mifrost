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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", default="blocks")
    parser.add_argument("--problem", default="smedium")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-states", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--pymimir-only", action="store_true")
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()

    pymimir_domain, pymimir_problem, pymimir_pool = _pymimir_inputs(
        args.domain, args.problem, args.max_states
    )
    pymimir_states = _cycle(pymimir_pool, args.batch_size)
    pymimir_goals = list(pymimir_problem.get_goal_condition().get_literals())
    pymimir_encoder = mifrost.ColorEncoder(pymimir_domain)

    specs: list[tuple[str, str, int, Any]] = [
        (
            "pymimir",
            "batch_default",
            args.batch_size,
            lambda: pymimir_encoder.encode_batch(pymimir_states),
        ),
        (
            "pymimir",
            "batch_explicit_goals",
            args.batch_size,
            lambda: pymimir_encoder.encode_batch(pymimir_states, goals=pymimir_goals),
        ),
        (
            "pymimir",
            "single_loop",
            args.batch_size,
            lambda: [pymimir_encoder.encode(state) for state in pymimir_states],
        ),
        (
            "pymimir",
            "stream_end_to_end",
            args.batch_size,
            lambda: _stream_batch(pymimir_encoder, pymimir_states),
        ),
    ]
    state_pool_sizes = {"pymimir": len(pymimir_pool)}
    if not args.pymimir_only:
        pytyr_task, pytyr_reader, pytyr_pool = _pytyr_inputs(
            args.domain, args.problem, args.max_states
        )
        pytyr_states = _cycle(pytyr_pool, args.batch_size)
        pytyr_goals = list(pytyr_reader.problem_snapshot().goals)
        pytyr_encoder = mifrost.ColorEncoder(pytyr_task)
        specs.extend(
            [
                (
                    "pytyr",
                    "batch_default",
                    args.batch_size,
                    lambda: pytyr_encoder.encode_batch(pytyr_states),
                ),
                (
                    "pytyr",
                    "batch_explicit_goals",
                    args.batch_size,
                    lambda: pytyr_encoder.encode_batch(pytyr_states, goals=pytyr_goals),
                ),
                (
                    "pytyr",
                    "single_loop",
                    args.batch_size,
                    lambda: [pytyr_encoder.encode(state) for state in pytyr_states],
                ),
                (
                    "pytyr",
                    "stream_end_to_end",
                    args.batch_size,
                    lambda: _stream_batch(pytyr_encoder, pytyr_states),
                ),
            ]
        )
        state_pool_sizes["pytyr"] = len(pytyr_pool)

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
        "state_pool_sizes": state_pool_sizes,
        "results": [asdict(result) for result in results],
    }
    report = json.dumps(payload, indent=2)
    print(report)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(report + "\n")


if __name__ == "__main__":
    main()
