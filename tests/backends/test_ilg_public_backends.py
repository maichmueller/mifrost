from __future__ import annotations

from collections import Counter
from pathlib import Path
import subprocess
import sys
from typing import Any

import pytest
import torch

import mifrost
from mifrost.encoders.types import BatchParam

from tests.encoding.test_semantic_hgraph_encoder import _named_edges

from . import isolated_subprocess_package_parent
from .test_semantic_parity import _backend_pair
from .test_transition_public_backends import _aligned_transition


ROOT = Path(__file__).resolve().parents[2]


def _assert_ilg_parity(actual: Any, expected: Any) -> None:
    actual_data = actual.as_pyg()
    expected_data = expected.as_pyg()
    assert set(actual_data.node_types) == set(expected_data.node_types)
    assert set(actual_data.edge_types) == set(expected_data.edge_types)
    assert Counter(actual_data.object_names) == Counter(expected_data.object_names)

    for node_type in expected_data.node_types:
        assert actual_data[node_type].num_nodes == expected_data[node_type].num_nodes
        if node_type == "action":
            assert torch.equal(actual_data[node_type].x, expected_data[node_type].x)
            continue
        actual_rows = Counter(
            (str(name), tuple(row))
            for name, row in zip(
                actual_data[node_type].node_names,
                actual_data[node_type].x.tolist(),
                strict=True,
            )
        )
        expected_rows = Counter(
            (str(name), tuple(row))
            for name, row in zip(
                expected_data[node_type].node_names,
                expected_data[node_type].x.tolist(),
                strict=True,
            )
        )
        assert actual_rows == expected_rows

    for edge_type in expected_data.edge_types:
        source_type, relation, target_type = edge_type
        if "action" not in {source_type, target_type}:
            assert _named_edges(actual_data, edge_type) == _named_edges(
                expected_data, edge_type
            )
            continue
        actual_symbols = list(actual_data["_symbol_"].node_names)
        expected_symbols = list(expected_data["_symbol_"].node_names)

        def action_edges(data: Any, symbols: list[str]) -> Counter[Any]:
            return Counter(
                (
                    relation,
                    str(symbols[int(source if source_type == "_symbol_" else target)]),
                    int(target if target_type == "action" else source),
                )
                for source, target in data[edge_type].edge_index.t().tolist()
            )

        assert action_edges(actual_data, actual_symbols) == action_edges(
            expected_data, expected_symbols
        )


@pytest.mark.parametrize(
    "config",
    [
        {},
        {"include_lgan_edges": True},
        {"add_nullary_predicates": True},
    ],
)
def test_public_ilg_backends_match_state_configurations(
    config: dict[str, Any],
) -> None:
    _pymimir_reader, problem, pytyr_reader, pytyr_search = _backend_pair()
    pymimir_encoder = mifrost.ILGEncoder(problem.get_domain(), **config)
    pytyr_encoder = mifrost.ILGEncoder(pytyr_reader._planning_task, **config)

    assert pymimir_encoder.backend == "pymimir"
    assert pytyr_encoder.backend == "pytyr"
    _assert_ilg_parity(
        pytyr_encoder.encode(pytyr_search.initial_node().get_state()),
        pymimir_encoder.encode(problem.get_initial_state()),
    )


def test_public_ilg_backends_match_goals_actions_subgoals_and_lgan() -> None:
    (
        _pymimir_reader,
        problem,
        pymimir_state,
        pymimir_action,
        _pymimir_successor,
        pytyr_reader,
        pytyr_state,
        pytyr_action,
        _pytyr_successor,
    ) = _aligned_transition()
    pymimir_goals = list(problem.get_goal_condition().get_literals())
    pytyr_goals = list(pytyr_reader.problem_snapshot().goals)
    pymimir_encoder = mifrost.ILGEncoder(problem.get_domain(), include_lgan_edges=True)
    pytyr_encoder = mifrost.ILGEncoder(
        pytyr_reader._planning_task, include_lgan_edges=True
    )

    _assert_ilg_parity(
        pytyr_encoder.encode(
            pytyr_state,
            goals=pytyr_goals,
            actions=[pytyr_action],
            subgoal_layers=[pytyr_goals],
        ),
        pymimir_encoder.encode(
            pymimir_state,
            goals=pymimir_goals,
            actions=[pymimir_action],
            subgoal_layers=[pymimir_goals],
        ),
    )


def test_public_ilg_pytyr_batch_stream_and_cross_backend_batching() -> None:
    _pymimir_reader, problem, pytyr_reader, pytyr_search = _backend_pair()
    pymimir_state = problem.get_initial_state()
    pytyr_root = pytyr_search.initial_node()
    pytyr_state = pytyr_root.get_state()
    pytyr_action = pytyr_search.action(pytyr_search.successors(pytyr_root)[0])
    pytyr_goals = list(pytyr_reader.problem_snapshot().goals)
    pymimir_encoder = mifrost.ILGEncoder(problem.get_domain())
    pytyr_encoder = mifrost.ILGEncoder(pytyr_reader._planning_task)

    batch = pytyr_encoder.encode_batch(
        [pytyr_state, pytyr_state],
        goals=BatchParam.separate([pytyr_goals, []]),
        actions=BatchParam.shared([pytyr_action]),
        subgoal_layers=BatchParam.shared([pytyr_goals]),
    )
    assert batch.num_graphs == 2

    stream = pytyr_encoder.stream()
    assert stream.append(pytyr_state, actions=[pytyr_action]) == 0
    assert stream.append(pytyr_state) == 1
    assert stream.flush().num_graphs == 2

    pymimir_encoding = pymimir_encoder.encode(pymimir_state)
    pytyr_encoding = pytyr_encoder.encode(pytyr_state)
    assert pymimir_encoding.schema_fingerprint() == (
        pytyr_encoding.schema_fingerprint()
    )
    mixed = mifrost.batch_encodings([pymimir_encoding, pytyr_encoding], fast_path=True)
    assert mixed.num_graphs == 2
    assert mifrost.BatchEncoding.loads(mixed.dumps()).num_graphs == 2


def test_public_ilg_selection_and_coexistence_errors() -> None:
    _pymimir_reader, problem, pytyr_reader, pytyr_search = _backend_pair()
    pymimir_state = problem.get_initial_state()
    pytyr_state = pytyr_search.initial_node().get_state()
    pymimir_encoder = mifrost.ILGEncoder(problem.get_domain(), backend="pymimir")
    pytyr_encoder = mifrost.ILGEncoder(pytyr_reader._planning_task, backend="pytyr")

    for _ in range(2):
        assert pymimir_encoder.encode(pymimir_state).num_graphs == 1
        assert pytyr_encoder.encode(pytyr_state).num_graphs == 1

    with pytest.raises(TypeError, match="PlanningTask"):
        mifrost.ILGEncoder(problem.get_domain(), backend="pytyr")
    with pytest.raises(TypeError, match="Unsupported domain type"):
        mifrost.ILGEncoder(pytyr_reader._planning_task, backend="pymimir")
    with pytest.raises(TypeError, match="only PyTyr states"):
        pytyr_encoder.encode_batch([pytyr_state, pymimir_state])
    with pytest.raises(ValueError, match="backend must be"):
        mifrost.ILGEncoder(pytyr_reader._planning_task, backend="other")


def test_public_ilg_pytyr_only_source_blocker_encodes() -> None:
    package_parent = isolated_subprocess_package_parent(ROOT / "src")
    dependency_paths = [path for path in sys.path if "site-packages" in path]
    domain = ROOT / "data" / "pddl" / "blocks" / "domain.pddl"
    problem = ROOT / "data" / "pddl" / "blocks" / "small.pddl"
    script = f"""
import importlib.abc
import sys

sys.path[:0] = {([str(package_parent), *dependency_paths])!r}

class BlockPymimir(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if (fullname == "mifrost._pymimir_adapter" or fullname == "pymimir"
                or fullname.startswith("pymimir.")):
            raise ModuleNotFoundError("blocked Pymimir for ILG smoke")
        return None

sys.meta_path.insert(0, BlockPymimir())

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

options = ParserOptions()
planning_task = Parser({str(domain)!r}, options).parse_task({str(problem)!r}, options)
task = Task(planning_task)
context = ExecutionContext(1)
evaluator = AxiomEvaluatorFactory().create(task, context)
repository = StateRepositoryFactory().create(task)
generator = SuccessorGeneratorFactory().create(task, context)
state = generator.get_initial_node(repository, evaluator).get_state()
encoder = mifrost.ILGEncoder(planning_task)
assert encoder.backend == "pytyr"
assert encoder.encode(state).num_graphs == 1
assert encoder.encode_batch([state, state]).num_graphs == 2
stream = encoder.stream()
stream.append(state)
assert stream.flush().num_graphs == 1
assert "pymimir" not in sys.modules
"""
    subprocess.run(
        [sys.executable, "-S", "-c", script],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
