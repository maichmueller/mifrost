from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[2]


def _load_script_module(name: str, relative_path: str):
    path = ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_benchmark_encoder_suite_skips_invalid_hgraph_lgan_scenarios(monkeypatch):
    module = _load_script_module(
        "benchmark_encoder_suite_test",
        "scripts/benchmark_encoder_suite.py",
    )

    class FakeGoalCondition:
        def get_literals(self):
            return ["goal0"]

    class FakeProblem:
        def get_goal_condition(self):
            return FakeGoalCondition()

    class FakeAction:
        def apply(self, state):
            return state

    class FakeState:
        def generate_applicable_actions(self):
            return [FakeAction()]

    class FakeStream:
        def append(self, *args, **kwargs):
            return None

        def flush(self):
            return None

        def flush_pyg(self, *, as_batch: bool):
            assert as_batch is True
            return None

    class FakeEncoding:
        def as_pyg(self, *, as_batch: bool):
            assert as_batch is True
            return None

    class FakeEncoder:
        def __init__(self, *args, **kwargs):
            self.kwargs = kwargs

        def encode(self, *args, **kwargs):
            return None

        def encode_batch(self, *args, **kwargs):
            return FakeEncoding()

        def stream(self):
            return FakeStream()

        def encode_pyg(self, *args, **kwargs):
            return None

        def encode_batch_pyg(self, *args, **kwargs):
            return None

    monkeypatch.setattr(module, "HGraphEncoder", FakeEncoder)
    monkeypatch.setattr(
        module,
        "HGRAPH_BENCH_SPEC",
        SimpleNamespace(allows_lgan=lambda *, include_actions: include_actions),
    )

    results = module._run_hgraph_suite(
        domain_obj=object(),
        problem_obj=FakeProblem(),
        states_pool=[FakeState()],
        batch_sizes=[1],
        include_lgan_values=[True],
        max_actions_per_state=1,
        benchmark_pyg=False,
        warmup=0,
        repeats=1,
    )

    assert results
    assert {(row.scenario, row.context["include_actions"]) for row in results} == {
        ("goals_actions", True),
        ("goals_subgoals_actions", True),
    }
    assert all(
        not (row.context["include_lgan"] and not row.context["include_actions"])
        for row in results
    )


def test_benchmark_encoder_suite_skips_invalid_transition_lgan_scenarios(
    monkeypatch,
):
    module = _load_script_module(
        "benchmark_encoder_suite_transition_test",
        "scripts/benchmark_encoder_suite.py",
    )

    class FakeGoalCondition:
        def get_literals(self):
            return ["goal0"]

    class FakeProblem:
        def get_goal_condition(self):
            return FakeGoalCondition()

    class FakeStream:
        def append(self, *args, **kwargs):
            return None

        def flush(self):
            return None

        def flush_pyg(self, *, as_batch: bool):
            assert as_batch is True
            return None

    class FakeEncoding:
        def as_pyg(self, *, as_batch: bool):
            assert as_batch is True
            return None

    class FakeEncoder:
        def __init__(self, *args, **kwargs):
            self.kwargs = kwargs

        def encode(self, *args, **kwargs):
            return None

        def encode_batch(self, *args, **kwargs):
            return FakeEncoding()

        def stream(self):
            return FakeStream()

        def encode_pyg(self, *args, **kwargs):
            return None

        def encode_batch_pyg(self, *args, **kwargs):
            return None

    monkeypatch.setattr(module, "TransitionHGraphEncoder", FakeEncoder)
    monkeypatch.setattr(module, "TransitionEffectsHGraphEncoder", FakeEncoder)
    monkeypatch.setattr(
        module,
        "TRANSITION_LANE_SPEC",
        SimpleNamespace(allows_lgan=lambda *, include_actions: False),
    )

    results = module._run_transition_suite(
        domain_obj=object(),
        problem_obj=FakeProblem(),
        transitions_pool=[("s0", "a0", "s1")],
        batch_sizes=[1],
        include_lgan_values=[True],
        benchmark_pyg=False,
        warmup=0,
        repeats=1,
    )

    assert results == []


def test_benchmark_encoder_baseline_uses_per_item_ratios():
    module = _load_script_module(
        "benchmark_encoder_baseline_test",
        "scripts/benchmark_encoder_baseline.py",
    )

    results = [
        module.BenchResult(
            suite="python",
            scenario="state_goals",
            path="hgraph_single_native",
            n_items=8,
            median_ms=32.0,
            mean_ms=32.0,
            min_ms=32.0,
            max_ms=32.0,
            stdev_ms=0.0,
            context={},
        ),
        module.BenchResult(
            suite="python",
            scenario="state_goals",
            path="hgraph_batch_native",
            n_items=32,
            median_ms=8.0,
            mean_ms=8.0,
            min_ms=8.0,
            max_ms=8.0,
            stdev_ms=0.0,
            context={},
        ),
        module.BenchResult(
            suite="python",
            scenario="state_goals",
            path="flat_batch_native",
            n_items=32,
            median_ms=4.0,
            mean_ms=4.0,
            min_ms=4.0,
            max_ms=4.0,
            stdev_ms=0.0,
            context={},
        ),
        module.BenchResult(
            suite="python",
            scenario="transition_full_goals",
            path="transition_single_native",
            n_items=4,
            median_ms=16.0,
            mean_ms=16.0,
            min_ms=16.0,
            max_ms=16.0,
            stdev_ms=0.0,
            context={},
        ),
        module.BenchResult(
            suite="python",
            scenario="transition_full_goals",
            path="transition_batch_native",
            n_items=16,
            median_ms=8.0,
            mean_ms=8.0,
            min_ms=8.0,
            max_ms=8.0,
            stdev_ms=0.0,
            context={},
        ),
        module.BenchResult(
            suite="python",
            scenario="horizon_full_goals",
            path="horizon_single_native",
            n_items=4,
            median_ms=10.0,
            mean_ms=10.0,
            min_ms=10.0,
            max_ms=10.0,
            stdev_ms=0.0,
            context={},
        ),
        module.BenchResult(
            suite="python",
            scenario="horizon_full_goals",
            path="horizon_batch_native",
            n_items=8,
            median_ms=4.0,
            mean_ms=4.0,
            min_ms=4.0,
            max_ms=4.0,
            stdev_ms=0.0,
            context={},
        ),
    ]

    ratios = module._compute_ratio_report(results, None, None)

    assert ratios["hgraph_batch_vs_single_per_item"] == pytest.approx(16.0)
    assert ratios["transition_batch_vs_single_per_item"] == pytest.approx(8.0)
    assert ratios["horizon_batch_vs_single_per_item"] == pytest.approx(5.0)
    assert ratios["flat_vs_hgraph_python_batch"] == pytest.approx(2.0)
