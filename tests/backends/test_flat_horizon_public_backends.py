from __future__ import annotations

from pathlib import Path
import subprocess
import sys
from typing import Any

import pytest

import mifrost
from mifrost.encoders.types import BatchParam

from . import isolated_subprocess_package_parent
from .test_flat_semantic_adapter import _normalized_relations
from .test_horizon_public_backends import (
    _aligned_horizon_transitions,
    _paired_dags,
)

ROOT = Path(__file__).resolve().parents[2]


def _normalized_target_rows(encoding: Any) -> tuple[tuple[Any, ...], ...]:
    data = encoding.as_pyg()
    return tuple(
        sorted(
            zip(
                (str(value) for value in data.target_names),
                data.target_indices.tolist(),
                data.target_candidate_ids.tolist(),
                data.target_depths.tolist(),
                data.target_group_ids.tolist(),
                (
                    str(data.node_names[int(position)])
                    for position in data.target_positions.tolist()
                ),
                strict=True,
            )
        )
    )


def _assert_flat_horizon_parity(actual: Any, expected: Any) -> None:
    actual_data = actual.as_pyg()
    expected_data = expected.as_pyg()
    assert set(actual_data.node_names) == set(expected_data.node_names)
    assert set(actual_data.object_names) == set(expected_data.object_names)
    assert _normalized_relations(actual) == _normalized_relations(expected)
    assert _normalized_target_rows(actual) == _normalized_target_rows(expected)


@pytest.mark.parametrize("mode", ["full", "delta", "action"])
@pytest.mark.parametrize("root_policy", ["include", "encode_only", "exclude"])
def test_public_flat_horizon_backends_match_modes_and_root_policies(
    mode: str, root_policy: str
) -> None:
    problem, pymimir_root, reader, pytyr_root, transitions = (
        _aligned_horizon_transitions(count=1)
    )
    native_dag, pytyr_dag = _paired_dags(transitions, pymimir_root, pytyr_root)
    config = {
        "transition_mode": mode,
        "root_policy": root_policy,
        "ignore_actions": mode != "action",
        "goal_derivations": {mifrost.GoalDerivation.plain},
    }
    pymimir_encoder = mifrost.FlatHorizonEncoder(problem.get_domain(), **config)
    pytyr_encoder = mifrost.FlatHorizonEncoder(reader._planning_task, **config)

    assert pymimir_encoder.backend == "pymimir"
    assert pytyr_encoder.backend == "pytyr"
    assert pymimir_encoder.relation_names == pytyr_encoder.relation_names
    assert pymimir_encoder.relation_arities == pytyr_encoder.relation_arities
    _assert_flat_horizon_parity(
        pytyr_encoder.encode(pytyr_root, dag=pytyr_dag),
        pymimir_encoder.encode(pymimir_root, dag=native_dag),
    )


def test_public_flat_horizon_topology_goals_batch_builder_and_stream() -> None:
    problem, pymimir_root, reader, pytyr_root, transitions = (
        _aligned_horizon_transitions(count=2)
    )
    native_dag, pytyr_dag = _paired_dags(transitions, pymimir_root, pytyr_root)
    pymimir_goals = list(problem.get_goal_condition().get_literals())
    pytyr_goals = list(reader.problem_snapshot().goals)
    config = {
        "enable_parent_relation": True,
        "enable_sibling_relation": True,
        "enable_cousin_relation": True,
        "include_lgan_edges": True,
        "max_goal_level": 1,
        "support_literals": True,
        "goal_derivations": {mifrost.GoalDerivation.plain},
    }
    pymimir_encoder = mifrost.FlatHorizonEncoder(problem.get_domain(), **config)
    encoder = mifrost.FlatHorizonEncoder(reader._planning_task, **config)
    expected = pymimir_encoder.encode(
        pymimir_root,
        dag=native_dag,
        goals=pymimir_goals,
        subgoal_layers=[pymimir_goals],
    )
    actual = encoder.encode(
        pytyr_root,
        dag=pytyr_dag,
        goals=pytyr_goals,
        subgoal_layers=[pytyr_goals],
    )
    _assert_flat_horizon_parity(actual, expected)

    assert (
        encoder.encode_batch(
            [pytyr_root, pytyr_root],
            dags=BatchParam.shared(pytyr_dag),
            goals=BatchParam.separate([pytyr_goals, []]),
            subgoal_layers=[pytyr_goals],
        ).num_graphs
        == 2
    )
    builder = mifrost.BatchBuilder()
    builder.set_graph_kind("flat")
    encoder._encode_one_into_builder(
        pytyr_root,
        builder,
        dag=pytyr_dag,
        goals=pytyr_goals,
        subgoal_layers=[pytyr_goals],
    )
    builder.next_graph()
    built = builder.build()
    encoder.engine.finalize_batch_encoding(built)
    _assert_flat_horizon_parity(built, actual)

    stream = encoder.mutable_stream()
    first = stream.append(pytyr_root, pytyr_dag, goals=pytyr_goals)
    second = stream.append(pytyr_root)
    stream.update(first, pytyr_root, pytyr_dag, subgoal_layers=[pytyr_goals])
    stream.remove(second)
    assert stream.flush().num_graphs == 1


def test_public_flat_horizon_native_identity_and_coexistence() -> None:
    problem, pymimir_root, reader, pytyr_root, transitions = (
        _aligned_horizon_transitions(count=1)
    )
    native_dag, pytyr_dag = _paired_dags(transitions, pymimir_root, pytyr_root)
    pymimir_encoder = mifrost.FlatHorizonEncoder(
        problem.get_domain(), backend="pymimir"
    )
    pytyr_encoder = mifrost.FlatHorizonEncoder(reader._planning_task, backend="pytyr")

    assert isinstance(pymimir_encoder.engine, mifrost.FlatHorizonEncoderEngine)
    assert isinstance(pymimir_encoder.config, mifrost.FlatHorizonEncoderConfig)
    assert pymimir_encoder.engine.config is pymimir_encoder.config
    for _ in range(2):
        assert pymimir_encoder.encode(pymimir_root, dag=native_dag).num_graphs == 1
        assert pytyr_encoder.encode(pytyr_root, dag=pytyr_dag).num_graphs == 1

    with pytest.raises(TypeError, match="PyTyr root state"):
        pytyr_encoder.encode(pymimir_root, dag=pytyr_dag)
    with pytest.raises(ValueError, match="backend must be"):
        mifrost.FlatHorizonEncoder(reader._planning_task, backend="other")
    with pytest.raises(ValueError, match="assemblies are not supported"):
        mifrost.FlatHorizonEncoder(
            reader._planning_task,
            backend="pytyr",
            assembly=mifrost.SemanticFlatHorizonAssemblyExtension(),
        )


@pytest.mark.parametrize(
    "encoder_type",
    [mifrost.FlatTransitionEncoder, mifrost.FlatTransitionEffectsEncoder],
)
def test_public_flat_transition_wrappers_accept_pytyr(
    encoder_type: type[Any],
) -> None:
    problem, pymimir_root, reader, pytyr_root, transitions = (
        _aligned_horizon_transitions(count=1)
    )
    _pymimir_action, pymimir_state, _pytyr_action, pytyr_state = transitions[0]
    pymimir_encoder = encoder_type(problem.get_domain())
    pytyr_encoder = encoder_type(reader._planning_task)

    expected = pymimir_encoder.encode(pymimir_root, successor=pymimir_state)
    actual = pytyr_encoder.encode(pytyr_root, successor=pytyr_state)
    assert _normalized_relations(actual) == _normalized_relations(expected)
    assert (
        actual.target_candidate_ids.tolist() == expected.target_candidate_ids.tolist()
    )
    assert actual.target_depths.tolist() == expected.target_depths.tolist()
    assert (
        pytyr_encoder.encode_batch(
            [pytyr_root, pytyr_root],
            successors=BatchParam.shared(pytyr_state),
        ).num_graphs
        == 2
    )


def test_public_flat_horizon_pytyr_only_source_blocker_encodes_real_dag() -> None:
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
            raise ModuleNotFoundError("blocked Pymimir for Flat Horizon smoke")
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
import rustworkx as rx
import mifrost

options = ParserOptions()
planning_task = Parser({str(domain)!r}, options).parse_task({str(problem)!r}, options)
task = Task(planning_task)
context = ExecutionContext(1)
evaluator = AxiomEvaluatorFactory().create(task, context)
repository = StateRepositoryFactory().create(task, evaluator)
generator = SuccessorGeneratorFactory().create(task, context, repository)
root = generator.get_initial_node()
successor = generator.get_labeled_successor_nodes(root)[0]
dag = rx.PyDiGraph()
root_index = dag.add_node(root.get_state())
successor_index = dag.add_node({{"state": successor.node.get_state(), "candidate_id": 9}})
dag.add_edge(root_index, successor_index, successor.label)
encoder = mifrost.FlatHorizonEncoder(planning_task)
assert encoder.backend == "pytyr"
assert encoder.encode(root.get_state(), dag=dag).target_candidate_ids.tolist() == [9]
assert encoder.encode_batch(
    [root.get_state(), root.get_state()], dags=[dag, dag]
).num_graphs == 2
transition_encoder = mifrost.FlatTransitionEncoder(planning_task)
assert transition_encoder.backend == "pytyr"
assert transition_encoder.encode(
    root.get_state(), successor=successor.node.get_state()
).num_graphs == 1
assert transition_encoder.encode_batch(
    [root.get_state(), root.get_state()],
    successors=mifrost.encoders.BatchParam.shared(successor.node.get_state()),
).num_graphs == 2
assert mifrost.FlatTransitionEffectsEncoder(planning_task).encode(
    root.get_state(), successor=successor.node.get_state()
).num_graphs == 1
assert mifrost.FlatHorizonEncoderConfig is (
    mifrost._neutral_core.SemanticFlatHorizonEncoderConfig
)
assert mifrost.FlatHorizonEncoderMode is (
    mifrost._neutral_core.SemanticHorizonEncoderMode
)
assert "pymimir" not in sys.modules
"""
    subprocess.run(
        [sys.executable, "-S", "-c", script],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
