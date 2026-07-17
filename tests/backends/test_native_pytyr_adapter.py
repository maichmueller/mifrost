from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch
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
from mifrost.backends.flat import FlatSemanticAdapter
from mifrost.backends.pytyr import (
    PyTyrSnapshotReader,
    SemanticFlatRelationEncoder,
    SemanticPlanningTaskAdapter,
)


pytyr_adapter = pytest.importorskip(
    "mifrost._pytyr_adapter",
    reason="the optional native PyTyr adapter is not built",
)

ROOT = Path(__file__).resolve().parents[2]
PARITY_CASES = (
    ("blocks", "small"),
    ("blocks_multiple", "blocks_b-2_v-1"),
    ("gripper", "gripper_b-1"),
    ("spanner", "small"),
    ("visitall", "visitall_x-2_y-1_r-50"),
    ("delivery", "instance_2x2_p-1_0"),
    ("reward", "instance_3x3_0"),
)


def _config() -> Any:
    return mifrost.FlatRelationEncoderConfig(
        ignore_zero_arity_relations=False,
        use_predicate_virtual_nodes=False,
        include_lgan_edges=False,
        target_sources={mifrost.TargetSource.actions, mifrost.TargetSource.goals},
    )


def _pytyr_pair(domain: str, problem: str) -> tuple[Any, Any]:
    directory = ROOT / "data" / "pddl" / domain
    options = ParserOptions()
    planning_task = Parser(str(directory / "domain.pddl"), options).parse_task(
        str(directory / f"{problem}.pddl"), options
    )
    task = Task(planning_task)
    context = ExecutionContext(1)
    evaluator = AxiomEvaluatorFactory().create(task, context)
    repository = StateRepositoryFactory().create(task, evaluator)
    successor_generator = SuccessorGeneratorFactory().create(task, context, repository)
    return PyTyrSnapshotReader(planning_task), successor_generator


def _assert_payload_equal(actual: Any, expected: Any) -> None:
    actual_payload = actual.as_pyg().to_dict()
    expected_payload = expected.as_pyg().to_dict()
    assert actual_payload.keys() == expected_payload.keys()
    for key in actual_payload:
        if torch.is_tensor(actual_payload[key]):
            assert torch.equal(actual_payload[key], expected_payload[key]), key
        else:
            assert actual_payload[key] == expected_payload[key], key


def _input_payload(value: Any) -> tuple[Any, ...]:
    return (
        tuple(value.objects),
        tuple((atom.predicate, tuple(atom.arguments)) for atom in value.state_facts),
        tuple(
            (literal.atom.predicate, tuple(literal.atom.arguments), literal.positive)
            for literal in value.goals
        ),
        tuple((action.action, tuple(action.arguments)) for action in value.actions),
        tuple(
            tuple(
                (
                    literal.atom.predicate,
                    tuple(literal.atom.arguments),
                    literal.positive,
                )
                for literal in layer
            )
            for layer in value.subgoal_layers
        ),
        tuple(
            (
                entry.dt,
                tuple(
                    (
                        literal.atom.predicate,
                        tuple(literal.atom.arguments),
                        literal.positive,
                    )
                    for literal in entry.literals
                ),
            )
            for entry in value.history
        ),
        value.history_max_steps,
    )


@pytest.mark.parametrize(("domain", "problem"), PARITY_CASES)
def test_native_pytyr_conversion_matches_python_semantic_contract(
    domain: str, problem: str
) -> None:
    reader, successor_generator = _pytyr_pair(domain, problem)
    root = successor_generator.get_initial_node()
    state = root.get_state()
    actions = [
        successor.label
        for successor in successor_generator.get_labeled_successor_nodes(root)
    ]

    native = SemanticFlatRelationEncoder(reader._planning_task, _config())
    semantic = FlatSemanticAdapter(reader, _config())

    native_input = native.make_input(state, actions)
    semantic_input = semantic.make_input(state, actions=actions)
    # The native path owns immutable task constants in its shared context, so
    # a per-state transport carries dynamic facts/actions only.
    assert native_input.objects == []
    assert native_input.goals == []
    assert len(native_input.state_facts) == len(semantic_input.state_facts) - len(
        reader.state_snapshot(state).static_atoms
    )
    assert tuple(
        (action.action, tuple(action.arguments)) for action in native_input.actions
    ) == tuple(
        (action.action, tuple(action.arguments)) for action in semantic_input.actions
    )
    _assert_payload_equal(
        native.encode(state, actions), semantic.encode(state, actions=actions)
    )


def test_native_pytyr_matches_native_pymimir_by_semantic_names() -> None:
    pytest.importorskip("pymimir")
    from .test_flat_semantic_adapter import _assert_semantically_equal
    from .test_semantic_parity import _backend_pair

    _, pymimir_problem, reader, successor_generator = _backend_pair()
    pymimir_root = pymimir_problem.get_initial_state()
    pymimir_actions = list(pymimir_root.generate_applicable_actions())
    pytyr_root = successor_generator.get_initial_node()
    pytyr_actions = [
        successor.label
        for successor in successor_generator.get_labeled_successor_nodes(pytyr_root)
    ]

    pymimir_engine = mifrost.FlatRelationEncoderEngine(
        pymimir_problem.get_domain()._advanced_domain, _config()
    )
    pymimir_goals = mifrost.GoalInputs(
        [
            literal._advanced_ground_literal
            for literal in pymimir_problem.get_goal_condition().get_literals()
        ],
        0,
    )
    pymimir_encoding = pymimir_engine.encode(
        pymimir_root._advanced_state,
        pymimir_goals,
        [action._advanced_ground_action for action in pymimir_actions],
    )

    pytyr_engine = SemanticFlatRelationEncoder(reader._planning_task, _config())
    pytyr_encoding = pytyr_engine.encode(pytyr_root.get_state(), pytyr_actions)

    _assert_semantically_equal(pytyr_encoding, pymimir_encoding)


def test_native_pytyr_optional_literal_lanes_match_semantic_contract() -> None:
    reader, successor_generator = _pytyr_pair("blocks", "small")
    state = successor_generator.get_initial_node().get_state()
    goals = reader.problem_snapshot().goals
    config = _config()
    config.max_goal_level = 1

    native = SemanticFlatRelationEncoder(reader._planning_task, config)
    semantic = FlatSemanticAdapter(reader, config)
    actual = native.make_input(
        state,
        goals=goals,
        subgoal_layers=[goals],
        history=[(-2, goals)],
        history_max_steps=4,
    )
    expected = semantic.make_input(
        state,
        goals=goals,
        subgoal_layers=[goals],
        history=[(-2, goals)],
        history_max_steps=4,
    )

    assert _input_payload(actual)[2:] == _input_payload(expected)[2:]
    _assert_payload_equal(
        native.engine.encode(actual), semantic.engine.encode(expected)
    )


def test_native_pytyr_raw_literal_lanes_match_semantic_contract() -> None:
    reader, successor_generator = _pytyr_pair("blocks", "small")
    state = successor_generator.get_initial_node().get_state()
    goal = reader._planning_task.get_task().get_goal()
    raw_goals = [
        *goal.get_static_facts(),
        *((fact, True) for fact in goal.get_positive_facts() if fact.has_value()),
        *((fact, False) for fact in goal.get_negative_facts() if fact.has_value()),
        *goal.get_derived_facts(),
    ]
    semantic_goals = reader.problem_snapshot().goals
    config = _config()
    config.max_goal_level = 1
    native = SemanticFlatRelationEncoder(reader._planning_task, config)
    semantic = FlatSemanticAdapter(reader, config)

    actual = native.make_input(
        state,
        goals=raw_goals,
        subgoal_layers=[raw_goals],
        history=[(-2, raw_goals)],
    )
    expected = semantic.make_input(
        state,
        goals=semantic_goals,
        subgoal_layers=[semantic_goals],
        history=[(-2, semantic_goals)],
    )

    assert _input_payload(actual)[2:] == _input_payload(expected)[2:]
    _assert_payload_equal(
        native.engine.encode(actual), semantic.engine.encode(expected)
    )


def test_native_pytyr_explicit_empty_goals_override_task_goals() -> None:
    reader, successor_generator = _pytyr_pair("blocks", "small")
    state = successor_generator.get_initial_node().get_state()
    native = SemanticFlatRelationEncoder(reader._planning_task, _config())

    assert not native.make_input(state).goals
    assert not native.make_input(state, goals=[]).goals
    assert native.encode(state).as_pyg().target_sizes.tolist()[0] > 0
    assert native.encode(state, goals=[]).as_pyg().target_sizes.tolist() == [0]


def test_native_pytyr_context_survives_adapter_and_omits_node_names() -> None:
    reader, successor_generator = _pytyr_pair("blocks", "small")
    state = successor_generator.get_initial_node().get_state()
    config = _config()
    config.export_node_names = False
    adapter = SemanticPlanningTaskAdapter(reader._planning_task)
    engine = adapter.make_flat_engine(config)
    input_value = adapter.make_input(state)
    del adapter

    payload = engine.encode(input_value).as_pyg().to_dict()
    assert input_value.objects == []
    assert input_value.goals == []
    assert "node_names" not in payload


def test_native_backends_remain_independent_when_interleaved() -> None:
    pytest.importorskip("pymimir")
    from .test_flat_semantic_adapter import _assert_semantically_equal
    from .test_semantic_parity import _backend_pair

    _, pymimir_problem, reader, successor_generator = _backend_pair()
    pymimir_root = pymimir_problem.get_initial_state()
    pytyr_root = successor_generator.get_initial_node().get_state()

    pymimir_engine = mifrost.FlatRelationEncoderEngine(
        pymimir_problem.get_domain()._advanced_domain, _config()
    )
    pymimir_goals = mifrost.GoalInputs(
        [
            literal._advanced_ground_literal
            for literal in pymimir_problem.get_goal_condition().get_literals()
        ],
        0,
    )
    pytyr_engine = SemanticFlatRelationEncoder(reader._planning_task, _config())
    expected = FlatSemanticAdapter(reader, _config()).encode(pytyr_root)

    retained = []
    for _ in range(3):
        pytyr_encoding = pytyr_engine.encode(pytyr_root)
        pymimir_encoding = pymimir_engine.encode(
            pymimir_root._advanced_state, pymimir_goals
        )
        _assert_payload_equal(pytyr_encoding, expected)
        _assert_semantically_equal(pymimir_encoding, expected)
        retained.extend((pytyr_encoding, pymimir_encoding))

    for index in range(0, len(retained), 2):
        _assert_payload_equal(retained[index], expected)
        _assert_semantically_equal(retained[index + 1], expected)
    assert isinstance(
        pytyr_engine.engine,
        mifrost._neutral_core.SemanticFlatRelationEncoderEngine,
    )
