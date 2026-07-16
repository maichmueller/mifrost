from __future__ import annotations

from pathlib import Path
import subprocess
import sys
from typing import Any

import pytest

import mifrost
from mifrost.encoders.types import BatchParam

from tests.encoding.test_semantic_hgraph_encoder import _assert_hgraph_parity
from tests.encoding.test_utils import adv_action, adv_state

from .test_semantic_parity import _backend_pair


ROOT = Path(__file__).resolve().parents[2]
rx = pytest.importorskip("rustworkx")


def _aligned_horizon_transitions(count: int = 2):
    pymimir_reader, problem, pytyr_reader, successor_generator = _backend_pair()
    pymimir_root = problem.get_initial_state()
    pytyr_root = successor_generator.get_initial_node()
    pymimir_successors = {
        pymimir_reader.action_key(action): (action, action.apply(pymimir_root))
        for action in pymimir_root.generate_applicable_actions()
    }
    pytyr_successors = {
        pytyr_reader.action_key(labeled.label): (
            labeled.label,
            labeled.node.get_state(),
        )
        for labeled in successor_generator.get_labeled_successor_nodes(pytyr_root)
    }
    shared = sorted(set(pymimir_successors).intersection(pytyr_successors), key=str)
    if len(shared) < count:
        pytest.skip(f"fixture does not expose {count} common Horizon transitions")
    return (
        problem,
        pymimir_root,
        pytyr_reader,
        pytyr_root.get_state(),
        [
            (
                pymimir_successors[key][0],
                pymimir_successors[key][1],
                pytyr_successors[key][0],
                pytyr_successors[key][1],
            )
            for key in shared[:count]
        ],
    )


def _paired_dags(
    transitions: list[tuple[Any, Any, Any, Any]], pymimir_root: Any, pytyr_root: Any
):
    native = mifrost.TransitionDAG(adv_state(pymimir_root))
    semantic_graph = rx.PyDiGraph()
    semantic_root = semantic_graph.add_node(
        {"state": pytyr_root, "display_name": str(adv_state(pymimir_root))}
    )
    for index, (
        pymimir_action,
        pymimir_state,
        pytyr_action,
        pytyr_state,
    ) in enumerate(transitions, start=1):
        native.register_transition(
            adv_state(pymimir_root),
            adv_state(pymimir_state),
            adv_action(pymimir_action),
        )
        semantic_state = semantic_graph.add_node(
            {
                "state": pytyr_state,
                "display_name": str(adv_state(pymimir_state)),
            }
        )
        semantic_graph.add_edge(semantic_root, semantic_state, pytyr_action)
        assert semantic_state == index
    return native, semantic_graph


def _assert_public_parity(
    pymimir_encoder: Any,
    pytyr_encoder: Any,
    pymimir_encoding: Any,
    pytyr_encoding: Any,
) -> None:
    _assert_hgraph_parity(
        pymimir_encoding.as_pyg(),
        pytyr_encoding.as_pyg(),
        native_relation_arities=dict(pymimir_encoder.relation_dict),
        semantic_relation_arities=dict(pytyr_encoder.relation_dict),
    )


@pytest.mark.parametrize("mode", ["full", "delta", "action"])
@pytest.mark.parametrize("root_policy", ["include", "encode_only", "exclude"])
def test_public_horizon_backends_match_modes_and_root_policies(
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
    pymimir_encoder = mifrost.HorizonEncoder(problem.get_domain(), **config)
    pytyr_encoder = mifrost.HorizonEncoder(reader._planning_task, **config)

    assert pymimir_encoder.backend == "pymimir"
    assert pytyr_encoder.backend == "pytyr"
    _assert_public_parity(
        pymimir_encoder,
        pytyr_encoder,
        pymimir_encoder.encode(pymimir_root, dag=native_dag),
        pytyr_encoder.encode(pytyr_root, dag=pytyr_dag),
    )


def test_public_horizon_backends_match_topology_goals_subgoals_and_metadata() -> None:
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
    pymimir_encoder = mifrost.HorizonEncoder(problem.get_domain(), **config)
    pytyr_encoder = mifrost.HorizonEncoder(reader._planning_task, **config)

    _assert_public_parity(
        pymimir_encoder,
        pytyr_encoder,
        pymimir_encoder.encode(
            pymimir_root,
            dag=native_dag,
            goals=pymimir_goals,
            subgoal_layers=[pymimir_goals],
        ),
        pytyr_encoder.encode(
            pytyr_root,
            dag=pytyr_dag,
            goals=pytyr_goals,
            subgoal_layers=[pytyr_goals],
        ),
    )


def test_public_horizon_pytyr_batch_builder_and_stream_lifecycle() -> None:
    _problem, _pymimir_root, reader, root, transitions = _aligned_horizon_transitions(
        count=1
    )
    _pymimir_action, _pymimir_state, pytyr_action, pytyr_state = transitions[0]
    graph = rx.PyDiGraph()
    root_index = graph.add_node(root)
    state_index = graph.add_node({"state": pytyr_state, "candidate_id": 17})
    graph.add_edge(root_index, state_index, pytyr_action)
    goals = list(reader.problem_snapshot().goals)
    encoder = mifrost.HorizonEncoder(reader._planning_task, max_goal_level=1)

    single = encoder.encode(root, dag=graph)
    assert single.target_candidate_ids.tolist() == [17]
    assert list(single.target_names) == [str(pytyr_state)]

    batch = encoder.encode_batch(
        [root, root],
        dags=BatchParam.shared(graph),
        goals=BatchParam.separate([goals, []]),
        subgoal_layers=[goals],
    )
    assert batch.num_graphs == 2

    builder = mifrost.BatchBuilder()
    builder.set_graph_kind("hetero")
    encoder._encode_one_into_builder(
        root,
        builder,
        dag=graph,
        goals=goals,
        subgoal_layers=[goals],
    )
    builder.next_graph()
    built = builder.build()
    _assert_hgraph_parity(
        encoder.encode(root, dag=graph, goals=goals, subgoal_layers=[goals]).as_pyg(),
        built.as_pyg(),
        native_relation_arities=dict(encoder.relation_dict),
        semantic_relation_arities=dict(encoder.relation_dict),
    )

    stream = encoder.stream()
    first = stream.append(root, graph, goals=goals)
    second = stream.append(root)
    stream.update(first, root, graph, subgoal_layers=[goals])
    stream.remove(second)
    assert stream.flush().num_graphs == 1
    assert stream.append(root, graph) == 0

    stream = encoder.stream()
    stream.set_reuse_removed(True)
    removed = stream.append(root)
    stream.remove(removed)
    assert stream.append(root, graph) == removed


def test_public_horizon_accepts_owned_semantic_dag_for_pytyr() -> None:
    _problem, _pymimir_root, reader, root, transitions = _aligned_horizon_transitions(
        count=1
    )
    _pymimir_action, _pymimir_state, pytyr_action, pytyr_state = transitions[0]
    graph = rx.PyDiGraph()
    root_index = graph.add_node(root)
    state_index = graph.add_node(pytyr_state)
    graph.add_edge(root_index, state_index, pytyr_action)
    encoder = mifrost.HorizonEncoder(reader._planning_task)
    owned = encoder._runtime._dag(root, graph)

    assert isinstance(owned, mifrost.SemanticTransitionDAG)
    _assert_hgraph_parity(
        encoder.encode(root, dag=graph).as_pyg(),
        encoder.encode(root, dag=owned).as_pyg(),
        native_relation_arities=dict(encoder.relation_dict),
        semantic_relation_arities=dict(encoder.relation_dict),
    )


def test_public_horizon_native_identity_coexistence_and_wrong_inputs() -> None:
    problem, pymimir_root, reader, pytyr_root, transitions = (
        _aligned_horizon_transitions(count=1)
    )
    native_dag, pytyr_dag = _paired_dags(transitions, pymimir_root, pytyr_root)
    pymimir_encoder = mifrost.HorizonEncoder(problem.get_domain(), backend="pymimir")
    pytyr_encoder = mifrost.HorizonEncoder(reader._planning_task, backend="pytyr")

    assert isinstance(pymimir_encoder.engine, mifrost.HorizonHGraphEncoderEngine)
    assert isinstance(pymimir_encoder.config, mifrost.HorizonEncoderConfig)
    assert pymimir_encoder.engine.config is pymimir_encoder.config
    retained = []
    for _ in range(2):
        retained.append(pymimir_encoder.encode(pymimir_root, dag=native_dag))
        retained.append(pytyr_encoder.encode(pytyr_root, dag=pytyr_dag))
    assert [encoding.num_graphs for encoding in retained] == [1, 1, 1, 1]

    with pytest.raises(TypeError, match="PyTyr root state"):
        pytyr_encoder.encode(pymimir_root, dag=pytyr_dag)
    with pytest.raises(TypeError, match="SemanticTransitionDAG"):
        pytyr_encoder.encode(pytyr_root, dag=native_dag)
    mixed_edges = rx.PyDiGraph()
    root_index = mixed_edges.add_node(pytyr_root)
    state_index = mixed_edges.add_node(transitions[0][3])
    mixed_edges.add_edge(root_index, state_index, transitions[0][0])
    with pytest.raises(TypeError, match="PyTyr ground action"):
        pytyr_encoder.encode(pytyr_root, dag=mixed_edges)
    with pytest.raises(TypeError, match="only PyTyr root states"):
        pytyr_encoder.encode_batch(
            [pytyr_root, pymimir_root], dags=[pytyr_dag, pytyr_dag]
        )
    with pytest.raises(TypeError, match="PlanningTask"):
        mifrost.HorizonEncoder(problem.get_domain(), backend="pytyr")
    with pytest.raises(TypeError, match="Unsupported domain type"):
        mifrost.HorizonEncoder(reader._planning_task, backend="pymimir")
    with pytest.raises(ValueError, match="backend must be"):
        mifrost.HorizonEncoder(reader._planning_task, backend="other")


def test_public_horizon_rejects_inconsistent_multi_parent_actions() -> None:
    _problem, _pymimir_root, reader, root, transitions = _aligned_horizon_transitions(
        count=2
    )
    _, _, action_a, state_a = transitions[0]
    _, _, action_b, state_b = transitions[1]
    graph = rx.PyDiGraph()
    root_index = graph.add_node(root)
    a_index = graph.add_node(state_a)
    b_index = graph.add_node(state_b)
    graph.add_edge(root_index, a_index, action_a)
    graph.add_edge(root_index, b_index, action_b)
    graph.add_edge(a_index, b_index, None)
    encoder = mifrost.HorizonEncoder(reader._planning_task)

    with pytest.raises(ValueError, match="same action"):
        encoder.encode(root, dag=graph)

    reverse = rx.PyDiGraph()
    root_index = reverse.add_node(root)
    a_index = reverse.add_node(state_a)
    b_index = reverse.add_node(state_b)
    reverse.add_edge(root_index, a_index, action_a)
    reverse.add_edge(a_index, b_index, None)
    reverse.add_edge(root_index, b_index, action_b)
    with pytest.raises(ValueError, match="same action"):
        encoder.encode(root, dag=reverse)


def test_public_horizon_pytyr_only_source_blocker_encodes_real_dag() -> None:
    package_parent = ROOT / "src"
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
            raise ModuleNotFoundError("blocked Pymimir for PyTyr Horizon smoke")
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
encoder = mifrost.HorizonEncoder(planning_task)
assert encoder.backend == "pytyr"
assert encoder.encode(root.get_state(), dag=dag).target_candidate_ids.tolist() == [9]
assert encoder.encode_batch(
    [root.get_state(), root.get_state()], dags=[dag, dag]
).num_graphs == 2
assert mifrost.HorizonEncoderConfig is mifrost._neutral_core.SemanticHorizonHGraphEncoderConfig
assert mifrost.HorizonEncoderMode is mifrost._neutral_core.SemanticHorizonEncoderMode
assert "pymimir" not in sys.modules
"""
    subprocess.run(
        [sys.executable, "-S", "-c", script],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
