from __future__ import annotations

from pathlib import Path
import subprocess
import sys
from typing import Any

import pytest

import mifrost
from mifrost.encoders.types import BatchParam

from tests.conftest import problem_setup
from tests.encoding.test_semantic_hgraph_encoder import _assert_hgraph_parity

from .test_semantic_parity import _backend_pair


ROOT = Path(__file__).resolve().parents[2]


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


@pytest.mark.parametrize(
    "config",
    [
        {},
        {"include_static": False, "include_empty_edge_types": False},
        {"add_nullary_predicates": True},
    ],
)
def test_public_hgraph_backends_match_base_configurations(
    config: dict[str, Any],
) -> None:
    _pymimir_reader, problem, reader, successor_generator = _backend_pair()
    pymimir_encoder = mifrost.HGraphEncoder(problem.get_domain(), **config)
    pytyr_encoder = mifrost.HGraphEncoder(reader._planning_task, **config)

    assert pymimir_encoder.backend == "pymimir"
    assert pytyr_encoder.backend == "pytyr"
    _assert_public_parity(
        pymimir_encoder,
        pytyr_encoder,
        pymimir_encoder.encode(problem.get_initial_state()),
        pytyr_encoder.encode(successor_generator.get_initial_node().get_state()),
    )


def test_public_hgraph_backends_match_all_optional_lanes_and_lgan() -> None:
    space, domain, problem = problem_setup("blocks", "small")
    pymimir_reader, _other_problem, reader, successor_generator = _backend_pair()
    pymimir_state = problem.get_initial_state()
    pytyr_root = successor_generator.get_initial_node()
    pytyr_state = pytyr_root.get_state()
    pymimir_actions = [
        action
        for action, _successor in space.get_forward_transitions(pymimir_state)
        if action is not None
    ]
    pytyr_actions = [
        successor.label
        for successor in successor_generator.get_labeled_successor_nodes(pytyr_root)
    ]
    pymimir_by_key = {
        pymimir_reader.action_key(action): action for action in pymimir_actions
    }
    pytyr_by_key = {reader.action_key(action): action for action in pytyr_actions}
    shared_keys = sorted(set(pymimir_by_key).intersection(pytyr_by_key), key=str)
    if not shared_keys:
        pytest.skip("fixture does not expose a common applicable action")
    action_key = shared_keys[0]
    pymimir_goals = list(problem.get_goal_condition().get_literals())
    pytyr_goals = list(reader.problem_snapshot().goals)
    config = {
        "ignore_actions": False,
        "include_lgan_edges": True,
        "max_goal_level": 1,
        "support_literals": True,
        "target_sources": {"action", "goal", "subgoal", "history"},
        "lgan_anchor_sources": {"action", "goal", "subgoal", "history"},
        "goal_derivations": {
            mifrost.GoalDerivation.plain,
            mifrost.GoalDerivation.satisfied,
            mifrost.GoalDerivation.unsatisfied,
        },
    }
    pymimir_encoder = mifrost.HGraphEncoder(domain, **config)
    pytyr_encoder = mifrost.HGraphEncoder(reader._planning_task, **config)

    _assert_public_parity(
        pymimir_encoder,
        pytyr_encoder,
        pymimir_encoder.encode(
            pymimir_state,
            goals=pymimir_goals,
            actions=[pymimir_by_key[action_key]],
            subgoal_layers=[pymimir_goals],
            history_subgoals=[(-2, pymimir_goals)],
            history_max_steps=4,
        ),
        pytyr_encoder.encode(
            pytyr_state,
            goals=pytyr_goals,
            actions=[pytyr_by_key[action_key]],
            subgoal_layers=[pytyr_goals],
            history_subgoals=[(-2, pytyr_goals)],
            history_max_steps=4,
        ),
    )


def test_public_hgraph_pytyr_batch_builder_and_streams() -> None:
    _pymimir_reader, _problem, reader, successor_generator = _backend_pair()
    root = successor_generator.get_initial_node()
    state = root.get_state()
    actions = [
        successor.label
        for successor in successor_generator.get_labeled_successor_nodes(root)
    ][:1]
    goals = list(reader.problem_snapshot().goals)
    encoder = mifrost.HGraphEncoder(
        reader._planning_task,
        ignore_actions=False,
        max_goal_level=1,
        target_sources={"action", "goal", "subgoal", "history"},
    )
    batch = encoder.encode_batch(
        [state, state],
        goals=BatchParam.separate([goals, []]),
        actions=actions,
        subgoal_layers=[goals],
        history_subgoals=[(-1, goals)],
        history_max_steps=3,
    )
    assert batch.num_graphs == 2

    builder = mifrost.BatchBuilder()
    builder.set_graph_kind("hetero")
    encoder._encode_one_into_builder(state, builder, actions=actions)
    builder.next_graph()
    built = builder.build()
    _assert_hgraph_parity(
        encoder.encode(state, actions=actions).as_pyg(),
        built.as_pyg(),
        native_relation_arities=dict(encoder.relation_dict),
        semantic_relation_arities=dict(encoder.relation_dict),
    )

    stream = encoder.stream()
    assert stream.append(state, actions=actions) == 0
    assert stream.append(state) == 1
    assert stream.flush().num_graphs == 2

    mutable = encoder.mutable_stream()
    first = mutable.append(state)
    second = mutable.append(state, actions=actions)
    mutable.update(first, state, actions=actions)
    mutable.remove(second)
    assert mutable.flush().num_graphs == 1

    mutable = encoder.mutable_stream()
    mutable.set_reuse_removed(True)
    removed = mutable.append(state)
    mutable.remove(removed)
    assert mutable.append(state) == removed


def test_public_hgraph_pymimir_stream_history_overloads() -> None:
    _reader, problem, _pytyr_reader, _successor_generator = _backend_pair()
    state = problem.get_initial_state()
    goals = list(problem.get_goal_condition().get_literals())
    history = [(-1, goals), (-3, goals)]
    encoder = mifrost.HGraphEncoder(problem.get_domain(), target_sources={"history"})

    stream = encoder.stream()
    assert (
        stream.append(
            state,
            goals=goals,
            history_subgoals=history,
            history_max_steps=1,
        )
        == 0
    )
    assert stream.flush().num_graphs == 1

    mutable = encoder.mutable_stream()
    stream_id = mutable.append(state)
    mutable.update(
        stream_id,
        state,
        goals=goals,
        history_subgoals=history,
        history_max_steps=1,
    )
    assert mutable.flush().num_graphs == 1


def test_public_hgraph_rejects_mixed_inputs_and_wrong_selection() -> None:
    _reader, problem, pytyr_reader, successor_generator = _backend_pair()
    pytyr_root = successor_generator.get_initial_node()
    pytyr_state = pytyr_root.get_state()
    pytyr_action = successor_generator.get_labeled_successor_nodes(pytyr_root)[0].label
    pymimir_state = problem.get_initial_state()
    pymimir_action = list(pymimir_state.generate_applicable_actions())[0]
    encoder = mifrost.HGraphEncoder(pytyr_reader._planning_task, ignore_actions=False)
    assert (
        mifrost.HGraphEncoder(pytyr_reader._planning_task, backend="pytyr")
        .encode(pytyr_state)
        .num_graphs
        == 1
    )
    assert (
        mifrost.HGraphEncoder(problem.get_domain(), backend="pymimir")
        .encode(pymimir_state)
        .num_graphs
        == 1
    )

    with pytest.raises(TypeError, match="only PyTyr states"):
        encoder.encode_batch([pytyr_state, pymimir_state])
    with pytest.raises(TypeError, match="expects a lifted or ground PyTyr state"):
        encoder.encode(pymimir_state)
    with pytest.raises(TypeError, match="only PyTyr ground actions"):
        encoder.encode(pytyr_state, actions=[pymimir_action])
    with pytest.raises(TypeError, match="only PyTyr ground actions"):
        encoder.encode_batch([pytyr_state], actions=[pytyr_action, pymimir_action])
    with pytest.raises(TypeError, match="PlanningTask"):
        mifrost.HGraphEncoder(problem.get_domain(), backend="pytyr")
    with pytest.raises(TypeError, match="Unsupported domain type"):
        mifrost.HGraphEncoder(pytyr_reader._planning_task, backend="pymimir")
    with pytest.raises(ValueError, match="backend must be"):
        mifrost.HGraphEncoder(pytyr_reader._planning_task, backend="other")


def test_public_hgraph_relation_updates_coexistence_and_derived_regression() -> None:
    _reader, problem, pytyr_reader, successor_generator = _backend_pair()
    pymimir_encoder = mifrost.HGraphEncoder(problem.get_domain())
    pytyr_encoder = mifrost.HGraphEncoder(pytyr_reader._planning_task)
    pymimir_relations = dict(pymimir_encoder.relation_dict)
    assert pymimir_relations == dict(pytyr_encoder.relation_dict)
    pymimir_encoder.update_relations(pymimir_relations)
    assert dict(pymimir_encoder.relation_dict) == pymimir_relations
    pytyr_encoder.update_relations(pymimir_relations)
    assert dict(pytyr_encoder.relation_dict) == pymimir_relations

    retained = []
    for _ in range(2):
        retained.append(pymimir_encoder.encode(problem.get_initial_state()))
        retained.append(
            pytyr_encoder.encode(successor_generator.get_initial_node().get_state())
        )
    assert len(retained) == 4

    horizon = mifrost.HorizonEncoder(problem.get_domain())
    assert horizon.backend == "pymimir"
    assert isinstance(horizon.engine, mifrost.HorizonHGraphEncoderEngine)
    assert isinstance(horizon.config, mifrost.HorizonEncoderConfig)
    with pytest.raises(TypeError, match="PlanningTask"):
        mifrost.HorizonEncoder(problem.get_domain(), backend="pytyr")


def test_public_hgraph_pytyr_only_source_blocker_encodes() -> None:
    package_parent = Path(mifrost._neutral_core.__file__).resolve().parent.parent
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
            raise ModuleNotFoundError("blocked Pymimir for PyTyr-only smoke")
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
repository = StateRepositoryFactory().create(task, evaluator)
successor_generator = SuccessorGeneratorFactory().create(task, context, repository)
state = successor_generator.get_initial_node().get_state()
encoder = mifrost.HGraphEncoder(planning_task)
assert encoder.backend == "pytyr"
assert encoder.encode_batch([state, state]).num_graphs == 2
assert mifrost.HGraphEncoderConfig is mifrost._neutral_core.SemanticHGraphEncoderConfig
assert "pymimir" not in sys.modules
"""
    subprocess.run(
        [sys.executable, "-S", "-c", script],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
