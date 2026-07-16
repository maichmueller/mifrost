from __future__ import annotations

from pathlib import Path
import subprocess
import sys
from typing import Any

import pytest

import mifrost
from mifrost.encoders.types import BatchParam

from tests.encoding.test_semantic_hgraph_encoder import _assert_hgraph_parity
from tests.encoding.test_semantic_successor_hgraph_encoder import (
    _assert_batch_hgraph_parity,
)

from . import isolated_subprocess_package_parent
from .test_semantic_parity import _backend_pair


ROOT = Path(__file__).resolve().parents[2]


def _aligned_transition():
    pymimir_reader, problem, pytyr_reader, successor_generator = _backend_pair()
    pymimir_current = problem.get_initial_state()
    pytyr_root = successor_generator.get_initial_node()
    pytyr_current = pytyr_root.get_state()
    pymimir_successors = {
        pymimir_reader.action_key(action): (action, action.apply(pymimir_current))
        for action in pymimir_current.generate_applicable_actions()
    }
    pytyr_successors = {
        pytyr_reader.action_key(labeled.label): (
            labeled.label,
            labeled.node.get_state(),
        )
        for labeled in successor_generator.get_labeled_successor_nodes(pytyr_root)
    }
    shared = sorted(set(pymimir_successors).intersection(pytyr_successors), key=str)
    if not shared:
        pytest.skip("fixture does not expose a common successor transition")
    key = shared[0]
    return (
        pymimir_reader,
        problem,
        pymimir_current,
        pymimir_successors[key][0],
        pymimir_successors[key][1],
        pytyr_reader,
        pytyr_current,
        pytyr_successors[key][0],
        pytyr_successors[key][1],
    )


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
    "encoder_type",
    [mifrost.TransitionHGraphEncoder, mifrost.TransitionEffectsHGraphEncoder],
)
@pytest.mark.parametrize(
    "config",
    [
        {},
        {"include_static": False, "include_empty_edge_types": False},
        {"add_nullary_predicates": True, "support_literals": True},
        {"export_node_names": True, "successor_suffix": "[after]"},
    ],
)
def test_public_transition_backends_match_full_delta_and_configurations(
    encoder_type: Any,
    config: dict[str, Any],
) -> None:
    (
        _pymimir_reader,
        problem,
        pymimir_current,
        _pymimir_action,
        pymimir_successor,
        pytyr_reader,
        pytyr_current,
        _pytyr_action,
        pytyr_successor,
    ) = _aligned_transition()
    pymimir_encoder = encoder_type(problem.get_domain(), **config)
    pytyr_encoder = encoder_type(pytyr_reader._planning_task, **config)

    assert pymimir_encoder.backend == "pymimir"
    assert pytyr_encoder.backend == "pytyr"
    _assert_public_parity(
        pymimir_encoder,
        pytyr_encoder,
        pymimir_encoder.encode(
            pymimir_current,
            successor=pymimir_successor,
        ),
        pytyr_encoder.encode(pytyr_current, successor=pytyr_successor),
    )


def test_public_transition_backends_match_goals_subgoals_and_satisfaction() -> None:
    (
        _pymimir_reader,
        problem,
        pymimir_current,
        _pymimir_action,
        pymimir_successor,
        pytyr_reader,
        pytyr_current,
        _pytyr_action,
        pytyr_successor,
    ) = _aligned_transition()
    pymimir_goals = list(problem.get_goal_condition().get_literals())
    pytyr_goals = list(pytyr_reader.problem_snapshot().goals)
    config = {
        "include_successor_goal_satisfaction": True,
        "max_goal_level": 1,
        "support_literals": True,
        "goal_derivations": {
            mifrost.GoalDerivation.plain,
            mifrost.GoalDerivation.satisfied,
            mifrost.GoalDerivation.unsatisfied,
        },
    }
    pymimir_encoder = mifrost.TransitionHGraphEncoder(problem.get_domain(), **config)
    pytyr_encoder = mifrost.TransitionHGraphEncoder(
        pytyr_reader._planning_task, **config
    )

    _assert_public_parity(
        pymimir_encoder,
        pytyr_encoder,
        pymimir_encoder.encode(
            pymimir_current,
            successor=pymimir_successor,
            goals=pymimir_goals,
            subgoal_layers=[pymimir_goals],
        ),
        pytyr_encoder.encode(
            pytyr_current,
            successor=pytyr_successor,
            goals=pytyr_goals,
            subgoal_layers=[pytyr_goals],
        ),
    )


@pytest.mark.parametrize(
    "encoder_type",
    [mifrost.TransitionHGraphEncoder, mifrost.TransitionEffectsHGraphEncoder],
)
def test_public_transition_pytyr_batch_builder_and_stream_lifecycle(
    encoder_type: Any,
) -> None:
    (
        _pymimir_reader,
        _problem,
        _pymimir_current,
        _pymimir_action,
        _pymimir_successor,
        pytyr_reader,
        current,
        _pytyr_action,
        successor,
    ) = _aligned_transition()
    goals = list(pytyr_reader.problem_snapshot().goals)
    encoder = encoder_type(pytyr_reader._planning_task, max_goal_level=1)

    batch = encoder.encode_batch(
        [current, current],
        successors=BatchParam.shared(successor),
        goals=BatchParam.separate([goals, []]),
        subgoal_layers=[goals],
    )
    assert batch.num_graphs == 2

    builder = mifrost.BatchBuilder()
    builder.set_graph_kind("hetero")
    encoder._encode_one_into_builder(
        current,
        builder,
        successor=successor,
        goals=goals,
        subgoal_layers=[goals],
    )
    builder.next_graph()
    built = builder.build()
    _assert_hgraph_parity(
        encoder.encode(
            current,
            successor=successor,
            goals=goals,
            subgoal_layers=[goals],
        ).as_pyg(),
        built.as_pyg(),
        native_relation_arities=dict(encoder.relation_dict),
        semantic_relation_arities=dict(encoder.relation_dict),
    )

    stream = encoder.stream()
    first = stream.append(current, successor, goals=goals)
    second = stream.append(current, successor)
    stream.update(first, current, successor, subgoal_layers=[goals])
    stream.remove(second)
    assert stream.flush().num_graphs == 1
    assert stream.append(current, successor) == 0

    stream = encoder.stream()
    stream.set_reuse_removed(True)
    removed = stream.append(current, successor)
    stream.remove(removed)
    assert stream.append(current, successor) == removed


def test_public_transition_native_batch_parity_and_pymimir_engine_contract() -> None:
    (
        _pymimir_reader,
        problem,
        pymimir_current,
        _pymimir_action,
        pymimir_successor,
        pytyr_reader,
        pytyr_current,
        _pytyr_action,
        pytyr_successor,
    ) = _aligned_transition()
    pymimir_encoder = mifrost.TransitionHGraphEncoder(problem.get_domain())
    pytyr_encoder = mifrost.TransitionHGraphEncoder(pytyr_reader._planning_task)

    assert isinstance(pymimir_encoder.engine, mifrost.SuccessorHGraphEncoderEngine)
    assert isinstance(pymimir_encoder.config, mifrost.SuccessorEncoderConfig)
    assert (
        pymimir_encoder.engine.config.successor_mode
        == mifrost.SuccessorEncoderMode.full
    )
    native = pymimir_encoder.encode_batch(
        [pymimir_current, pymimir_current],
        successors=[pymimir_successor, pymimir_successor],
    ).as_pyg(as_batch=True)
    semantic = pytyr_encoder.encode_batch(
        [pytyr_current, pytyr_current],
        successors=[pytyr_successor, pytyr_successor],
    ).as_pyg(as_batch=True)
    _assert_batch_hgraph_parity(
        native,
        semantic,
        native_relation_arities=dict(pymimir_encoder.relation_dict),
        semantic_relation_arities=dict(pytyr_encoder.relation_dict),
    )

    native_relations = dict(pymimir_encoder.relation_dict)
    pymimir_encoder.update_relations(native_relations)
    assert dict(pymimir_encoder.relation_dict) == native_relations
    pytyr_encoder.update_relations(native_relations)
    assert dict(pytyr_encoder.relation_dict) == native_relations

    pymimir_effects = mifrost.TransitionEffectsHGraphEncoder(problem.get_domain())
    assert isinstance(pymimir_effects.engine, mifrost.SuccessorHGraphEncoderEngine)
    assert isinstance(pymimir_effects.config, mifrost.SuccessorEncoderConfig)
    assert pymimir_effects.config.support_literals is True
    assert (
        pymimir_effects.engine.config.successor_mode
        == mifrost.SuccessorEncoderMode.delta
    )


def test_public_transition_coexists_and_rejects_wrong_or_mixed_inputs() -> None:
    (
        _pymimir_reader,
        problem,
        pymimir_current,
        pymimir_action,
        pymimir_successor,
        pytyr_reader,
        pytyr_current,
        pytyr_action,
        pytyr_successor,
    ) = _aligned_transition()
    pymimir_encoder = mifrost.TransitionEffectsHGraphEncoder(
        problem.get_domain(), backend="pymimir"
    )
    pytyr_encoder = mifrost.TransitionEffectsHGraphEncoder(
        pytyr_reader._planning_task, backend="pytyr"
    )
    retained = []
    for _ in range(2):
        retained.append(
            pymimir_encoder.encode(pymimir_current, successor=pymimir_successor)
        )
        retained.append(pytyr_encoder.encode(pytyr_current, successor=pytyr_successor))
    assert len(retained) == 4
    assert pytyr_encoder.config.support_literals is True

    with pytest.raises(TypeError, match="PyTyr current state"):
        pytyr_encoder.encode(pymimir_current, successor=pytyr_successor)
    with pytest.raises(TypeError, match="PyTyr successor state"):
        pytyr_encoder.encode(pytyr_current, successor=pymimir_successor)
    with pytest.raises(TypeError, match="only PyTyr current states"):
        pytyr_encoder.encode_batch(
            [pytyr_current, pymimir_current],
            successors=[pytyr_successor, pytyr_successor],
        )
    with pytest.raises(TypeError, match="successors entry at index 1"):
        pytyr_encoder.encode_batch(
            [pytyr_current, pytyr_current],
            successors=[pytyr_successor, pymimir_successor],
        )
    with pytest.raises(ValueError, match="do not support explicit action"):
        pytyr_encoder.encode(
            pytyr_current,
            successor=pytyr_successor,
            actions=[pytyr_action, pymimir_action],
        )
    with pytest.raises(TypeError, match="PlanningTask"):
        mifrost.TransitionHGraphEncoder(problem.get_domain(), backend="pytyr")
    with pytest.raises(TypeError, match="Unsupported domain type"):
        mifrost.TransitionHGraphEncoder(pytyr_reader._planning_task, backend="pymimir")
    with pytest.raises(ValueError, match="backend must be"):
        mifrost.TransitionHGraphEncoder(pytyr_reader._planning_task, backend="other")


def test_public_transition_pytyr_only_source_blocker_encodes_actual_successor() -> None:
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
            raise ModuleNotFoundError("blocked Pymimir for PyTyr transition smoke")
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
generator = SuccessorGeneratorFactory().create(task, context, repository)
root = generator.get_initial_node()
successor = generator.get_labeled_successor_nodes(root)[0].node
for encoder_type in (
    mifrost.TransitionHGraphEncoder,
    mifrost.TransitionEffectsHGraphEncoder,
):
    encoder = encoder_type(planning_task)
    assert encoder.backend == "pytyr"
    assert encoder.encode(
        root.get_state(), successor=successor.get_state()
    ).num_graphs == 1
    assert encoder.encode_batch(
        [root.get_state(), root.get_state()],
        successors=[successor.get_state(), successor.get_state()],
    ).num_graphs == 2
assert mifrost.SuccessorEncoderConfig is mifrost._neutral_core.SemanticSuccessorHGraphEncoderConfig
assert mifrost.SuccessorEncoderMode is mifrost._neutral_core.SemanticSuccessorEncoderMode
assert "pymimir" not in sys.modules
"""
    subprocess.run(
        [sys.executable, "-S", "-c", script],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
