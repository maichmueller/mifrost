from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
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

from .test_semantic_parity import PyTyrSearch


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
    repository = StateRepositoryFactory().create(task)
    generator = SuccessorGeneratorFactory().create(task, context)
    return PyTyrSnapshotReader(planning_task), PyTyrSearch(
        generator, repository, evaluator
    )


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
    reader, pytyr_search = _pytyr_pair(domain, problem)
    root = pytyr_search.initial_node()
    state = root.get_state()
    actions = [
        pytyr_search.action(successor) for successor in pytyr_search.successors(root)
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

    _, pymimir_problem, reader, pytyr_search = _backend_pair()
    pymimir_root = pymimir_problem.get_initial_state()
    pymimir_actions = list(pymimir_root.generate_applicable_actions())
    pytyr_root = pytyr_search.initial_node()
    pytyr_actions = [
        pytyr_search.action(successor)
        for successor in pytyr_search.successors(pytyr_root)
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
    reader, pytyr_search = _pytyr_pair("blocks", "small")
    state = pytyr_search.initial_node().get_state()
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
    reader, pytyr_search = _pytyr_pair("blocks", "small")
    state = pytyr_search.initial_node().get_state()
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
    reader, pytyr_search = _pytyr_pair("blocks", "small")
    state = pytyr_search.initial_node().get_state()
    native = SemanticFlatRelationEncoder(reader._planning_task, _config())

    assert not native.make_input(state).goals
    assert not native.make_input(state, goals=[]).goals
    assert native.encode(state).as_pyg().target_sizes.tolist()[0] > 0
    assert native.encode(state, goals=[]).as_pyg().target_sizes.tolist() == [0]


def test_native_pytyr_context_survives_adapter_and_omits_node_names() -> None:
    reader, pytyr_search = _pytyr_pair("blocks", "small")
    state = pytyr_search.initial_node().get_state()
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


def test_native_pytyr_adapter_converts_successor_states_lazily() -> None:
    """Regression test for the adapter's ground-atom cache.

    The adapter's static/fluent atom cache is seeded once, at construction,
    from the task-owned atom enumeration reachable from the initial state and
    goal. Lifted successor generation grounds further fluent atoms lazily as
    new states are discovered, so a state several transitions away from the
    root can contain atoms the adapter has never seen. Those atoms must
    convert via a cache-miss-computes-and-memoizes path, not raise "outside
    the adapter task context" (see semantic_flat_encoder.cpp cached_atom).
    """
    reader, pytyr_search = _pytyr_pair("blocks", "small")
    root = pytyr_search.initial_node()
    adapter = SemanticPlanningTaskAdapter(reader._planning_task)
    config = _config()
    engine = adapter.make_flat_engine(config)

    # Build the adapter and encode the root before any successor exists, so
    # its atom cache only contains what's reachable from the initial state
    # and goal, matching the construction-time enumeration exactly.
    assert engine.encode(adapter.make_input(root.get_state())).num_graphs == 1

    frontier = [root]
    visited_states = 0
    for _ in range(3):
        next_frontier = []
        for node in frontier:
            for labeled in pytyr_search.successors(node):
                state = labeled.node.get_state()
                # Previously raised ValueError("... outside the adapter task
                # context") for fluent atoms first grounded by this
                # successor, since the adapter's cache was immutable after
                # construction.
                input_value = adapter.make_input(state, [pytyr_search.action(labeled)])
                assert engine.encode(input_value).num_graphs == 1
                visited_states += 1
                next_frontier.append(labeled.node)
        frontier = next_frontier
        if not frontier:
            break
    assert visited_states > 0


def test_native_pytyr_adapter_atom_cache_survives_concurrent_python_calls() -> None:
    """One adapter's lazily-grown atom cache must not corrupt under threads.

    cached_atom() in semantic_flat_encoder.cpp mutates a per-adapter dense
    cache on a first sighting of a ground atom (see
    test_native_pytyr_adapter_converts_successor_states_lazily). That
    mutation is unsynchronized and only safe because pytyr_module.cpp never
    releases the GIL around adapter calls, so concurrent Python threads
    calling into one adapter are already serialized. This drives many
    threads through make_input for a set of successor states that all miss
    the cache on first use, so a future binding change that adds
    gil_scoped_release here without adding synchronization would show up as
    a crash, a wrong/torn SemanticAtom, or a mismatched encoding below.
    """
    reader, pytyr_search = _pytyr_pair("blocks", "small")
    root = pytyr_search.initial_node()
    adapter = SemanticPlanningTaskAdapter(reader._planning_task)
    engine = adapter.make_flat_engine(_config())

    successors = [
        (pytyr_search.action(labeled), labeled.node.get_state())
        for labeled in pytyr_search.successors(root)
    ]
    assert successors

    def convert(pair: tuple[Any, Any]) -> int:
        action, state = pair
        input_value = adapter.make_input(state, [action])
        return engine.encode(input_value).num_graphs

    states = successors * 8
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(convert, states))
    assert results == [1] * len(states)


def test_native_backends_remain_independent_when_interleaved() -> None:
    pytest.importorskip("pymimir")
    from .test_flat_semantic_adapter import _assert_semantically_equal
    from .test_semantic_parity import _backend_pair

    _, pymimir_problem, reader, pytyr_search = _backend_pair()
    pymimir_root = pymimir_problem.get_initial_state()
    pytyr_root = pytyr_search.initial_node().get_state()

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


# --- Direct-View PyTyr encoding -------------------------------------------
#
# The normal PyTyr encode runs the canonical engine inside the PyTyr module, so
# a Tyr state and its actions reach the algorithm as granular Views instead of
# as an owning SemanticFlatRelationInput. Only the finished neutral encoding
# crosses back through a capsule, because PyTyr and Pymimir ship incompatible
# nanobind ABI generations. These tests pin that the direct path agrees exactly
# with the compatibility path it replaced, for every family that has one.


def _assert_deep_equal(actual: Any, expected: Any, path: str = "") -> None:
    """Tensor-aware recursive equality.

    The hetero payload nests tensors inside tuples and dicts, where a bare `==`
    yields a tensor rather than a bool, so the flat comparison above cannot be
    reused for every family.
    """
    if torch.is_tensor(actual) or torch.is_tensor(expected):
        assert torch.is_tensor(actual) and torch.is_tensor(expected), path
        assert torch.equal(actual, expected), path
        return
    if isinstance(actual, dict):
        assert isinstance(expected, dict), path
        assert actual.keys() == expected.keys(), path
        for key in actual:
            _assert_deep_equal(actual[key], expected[key], f"{path}.{key}")
        return
    if isinstance(actual, (list, tuple)):
        assert isinstance(expected, (list, tuple)), path
        assert len(actual) == len(expected), path
        for index, (left, right) in enumerate(zip(actual, expected, strict=True)):
            _assert_deep_equal(left, right, f"{path}[{index}]")
        return
    assert actual == expected, path


def _assert_encoding_deep_equal(actual: Any, expected: Any) -> None:
    _assert_deep_equal(actual.as_pyg().to_dict(), expected.as_pyg().to_dict())


def _direct_cases() -> tuple[tuple[str, str, str], ...]:
    """(family, direct factory, engine factory) triples."""
    return (
        ("flat", "make_direct_flat_encoder", "make_flat_engine"),
        ("color", "make_direct_color_encoder", "make_color_engine"),
        ("hgraph", "make_direct_hgraph_encoder", "make_hgraph_engine"),
    )


def _family_config(family: str) -> Any:
    if family == "flat":
        return _config()
    if family == "color":
        return mifrost._neutral_core.SemanticColorEncoderConfig()
    return mifrost._neutral_core.SemanticHGraphEncoderConfig(
        target_sources={mifrost.TargetSource.goals}
    )


@pytest.mark.parametrize(
    ("family", "direct_factory", "engine_factory"), _direct_cases()
)
@pytest.mark.parametrize(("domain", "problem"), PARITY_CASES)
def test_direct_view_encode_matches_owned_input(
    family: str, direct_factory: str, engine_factory: str, domain: str, problem: str
) -> None:
    reader, pytyr_search = _pytyr_pair(domain, problem)
    state = pytyr_search.initial_node().get_state()
    adapter = SemanticPlanningTaskAdapter(reader._planning_task)
    config = _family_config(family)

    direct = getattr(adapter, direct_factory)(config)
    engine = getattr(adapter, engine_factory)(config)

    _assert_encoding_deep_equal(
        direct.encode(state),
        engine.encode(adapter.make_input(state)),
    )


@pytest.mark.parametrize(("domain", "problem"), PARITY_CASES)
def test_direct_view_flat_optional_lanes_match_owned_input(
    domain: str, problem: str
) -> None:
    reader, pytyr_search = _pytyr_pair(domain, problem)
    root = pytyr_search.initial_node()
    state = root.get_state()
    actions = [
        pytyr_search.action(successor) for successor in pytyr_search.successors(root)
    ]
    config = mifrost.FlatRelationEncoderConfig(
        max_goal_level=2,
        target_sources={
            mifrost.TargetSource.actions,
            mifrost.TargetSource.goals,
            mifrost.TargetSource.subgoals,
            mifrost.TargetSource.history,
        },
    )
    adapter = SemanticPlanningTaskAdapter(reader._planning_task)
    direct = adapter.make_direct_flat_encoder(config)
    engine = adapter.make_flat_engine(config)

    goals = list(reader.problem_snapshot().goals)
    lanes: dict[str, Any] = {
        "goals": goals,
        "subgoal_layers": [goals, goals],
        "history": [(-1, goals), (-2, goals)],
        "history_max_steps": 2,
    }

    _assert_encoding_deep_equal(
        direct.encode(state, actions, **lanes),
        engine.encode(adapter.make_input(state, actions, **lanes)),
    )


def test_direct_view_encoder_keeps_its_adapter_alive() -> None:
    """The encoder borrows the adapter's View context, so it must own a reference."""
    reader, pytyr_search = _pytyr_pair("blocks", "small")
    state = pytyr_search.initial_node().get_state()

    def make() -> Any:
        adapter = SemanticPlanningTaskAdapter(reader._planning_task)
        return adapter.make_direct_flat_encoder(_config())

    direct = make()
    import gc

    gc.collect()
    assert direct.encode(state).num_graphs == 1


def _two_states(domain: str, problem: str) -> tuple[Any, list[Any], list[Any]]:
    """A root and one successor, plus the root's applicable actions."""
    reader, pytyr_search = _pytyr_pair(domain, problem)
    root = pytyr_search.initial_node()
    labeled = list(pytyr_search.successors(root))
    states = [root.get_state()]
    if labeled:
        states.append(labeled[0].node.get_state())
    return reader, states, [pytyr_search.action(entry) for entry in labeled]


@pytest.mark.parametrize(
    ("family", "direct_factory", "engine_factory"), _direct_cases()
)
@pytest.mark.parametrize(("domain", "problem"), PARITY_CASES)
def test_direct_view_batch_matches_owned_input_batch(
    family: str, direct_factory: str, engine_factory: str, domain: str, problem: str
) -> None:
    reader, states, actions = _two_states(domain, problem)
    adapter = SemanticPlanningTaskAdapter(reader._planning_task)
    config = _family_config(family)
    direct = getattr(adapter, direct_factory)(config)
    engine = getattr(adapter, engine_factory)(config)

    # Mixed lanes: the first graph carries actions, the second does not. Color
    # has no action lane at all and rejects one outright.
    action_lanes: list[Any] = (
        [()] * len(states)
        if family == "color"
        else [actions] + [()] * (len(states) - 1)
    )

    _assert_encoding_deep_equal(
        direct.encode_batch(states, action_lanes),
        engine.encode_batch(adapter.make_inputs(states, action_lanes)),
    )


@pytest.mark.parametrize(
    ("family", "direct_factory", "engine_factory"), _direct_cases()
)
def test_direct_view_prepared_batch_matches_direct_batch(
    family: str, direct_factory: str, engine_factory: str
) -> None:
    """A stream prepares per append and flushes the handles; same result, replayable."""
    reader, states, actions = _two_states("blocks", "small")
    adapter = SemanticPlanningTaskAdapter(reader._planning_task)
    config = _family_config(family)
    direct = getattr(adapter, direct_factory)(config)

    action_lanes: list[Any] = (
        [()] * len(states)
        if family == "color"
        else [actions] + [()] * (len(states) - 1)
    )
    prepared = [
        direct.prepare(state, lane)
        for state, lane in zip(states, action_lanes, strict=True)
    ]
    expected = direct.encode_batch(states, action_lanes)

    _assert_encoding_deep_equal(direct.encode_prepared(prepared), expected)
    # Borrowed, not consumed: flushing the same handles again is well defined.
    _assert_encoding_deep_equal(direct.encode_prepared(prepared), expected)


def test_direct_view_prepared_handle_outlives_its_state() -> None:
    """A preparation owns compact pools, so the Tyr state may go away first."""
    import gc

    reader, states, actions = _two_states("blocks", "small")
    adapter = SemanticPlanningTaskAdapter(reader._planning_task)
    direct = adapter.make_direct_flat_encoder(_config())

    expected = direct.encode_batch(states[:1], [actions])
    prepared = [direct.prepare(states[0], actions)]
    del states
    gc.collect()

    _assert_encoding_deep_equal(direct.encode_prepared(prepared), expected)


def _successor_config() -> Any:
    return mifrost._neutral_core.SemanticSuccessorHGraphEncoderConfig(
        max_goal_level=1,
        support_literals=True,
        include_successor_goal_satisfaction=True,
        goal_derivations={
            mifrost.GoalDerivation.plain,
            mifrost.GoalDerivation.satisfied,
            mifrost.GoalDerivation.unsatisfied,
        },
    )


@pytest.mark.parametrize(("domain", "problem"), PARITY_CASES)
def test_direct_view_successor_matches_owned_inputs(domain: str, problem: str) -> None:
    reader, states, _actions = _two_states(domain, problem)
    if len(states) < 2:
        pytest.skip("fixture exposes no successor state")
    adapter = SemanticPlanningTaskAdapter(reader._planning_task)
    config = _successor_config()
    direct = adapter.make_direct_successor_encoder(config)
    engine = adapter.make_successor_hgraph_engine(config)

    # A goal repeated across levels resolves to its highest level; the owned and
    # direct routes must agree on that, not just on the flat lanes.
    goals = list(reader.problem_snapshot().goals)
    current, successor = states[0], states[1]

    _assert_encoding_deep_equal(
        direct.encode(current, successor, goals=goals, subgoal_layers=[goals]),
        engine.encode(
            adapter.make_input(current, goals=goals, subgoal_layers=[goals]),
            adapter.make_input(successor),
        ),
    )
    _assert_encoding_deep_equal(
        direct.encode_batch(
            [current, current],
            [successor, successor],
            goals=[goals, None],
            subgoal_layers=[[goals], []],
        ),
        engine.encode_batch(
            adapter.make_inputs(
                [current, current],
                [(), ()],
                goals=[goals, None],
                subgoal_layers=[[goals], []],
            ),
            adapter.make_inputs([successor, successor], [(), ()]),
        ),
    )


def test_direct_view_successor_prepared_batch_matches_batch() -> None:
    reader, states, _actions = _two_states("blocks", "small")
    if len(states) < 2:
        pytest.skip("fixture exposes no successor state")
    adapter = SemanticPlanningTaskAdapter(reader._planning_task)
    direct = adapter.make_direct_successor_encoder(_successor_config())
    goals = list(reader.problem_snapshot().goals)
    current, successor = states[0], states[1]

    expected = direct.encode_batch(
        [current, current],
        [successor, successor],
        goals=[goals, None],
        subgoal_layers=[[goals], []],
    )
    prepared = [
        direct.prepare(current, successor, goals=goals, subgoal_layers=[goals]),
        direct.prepare(current, successor),
    ]
    _assert_encoding_deep_equal(direct.encode_prepared(prepared), expected)


def _relation_probe(runtime: Any) -> dict[str, int]:
    """The runtime's arity table plus one relation it has never seen."""
    relations = dict(runtime.relation_dict)
    relations["mifrost_probe_relation"] = 2
    return relations


def test_hgraph_runtime_update_relations_reaches_the_direct_path() -> None:
    """A runtime keeps two engines; both must see a replaced arity table.

    The direct encoder owns its own engine instance, so updating only the
    compatibility engine leaves every encode on the table the direct encoder
    was constructed with -- and the output is still self-consistent, so only a
    comparison against the compatibility engine catches it.
    """
    from mifrost.backends.pytyr_hgraph import PyTyrHGraphRuntime

    reader, states, actions = _two_states("blocks", "small")
    runtime = PyTyrHGraphRuntime(
        reader._planning_task, mifrost._neutral_core.SemanticHGraphEncoderConfig()
    )
    state = states[0]

    before = runtime.encode(state, actions=actions)
    runtime.update_relations(_relation_probe(runtime))
    after = runtime.encode(state, actions=actions)

    assert set(after.as_pyg().node_types) != set(before.as_pyg().node_types)
    _assert_encoding_deep_equal(
        after, runtime.engine.encode(runtime._input(state, actions=actions))
    )


def test_transition_runtime_update_relations_reaches_the_direct_path() -> None:
    from mifrost.backends.pytyr_transition import PyTyrTransitionRuntime

    reader, states, _actions = _two_states("blocks", "small")
    if len(states) < 2:
        pytest.skip("fixture exposes no successor state")
    runtime = PyTyrTransitionRuntime(
        reader._planning_task,
        mifrost._neutral_core.SemanticSuccessorHGraphEncoderConfig(),
    )
    current, successor = states[0], states[1]

    before = runtime.encode(current, successor)
    runtime.update_relations(_relation_probe(runtime))
    after = runtime.encode(current, successor)

    assert set(after.as_pyg().node_types) != set(before.as_pyg().node_types)
    _assert_encoding_deep_equal(
        after, runtime.engine.encode(*runtime._inputs(current, successor))
    )


@pytest.mark.parametrize(("domain", "problem"), PARITY_CASES)
def test_color_and_hgraph_runtimes_match_their_owned_route(
    domain: str, problem: str
) -> None:
    """The public Color and HGraph runtimes now encode through Views."""
    from mifrost.backends.pytyr_color import PyTyrColorRuntime
    from mifrost.backends.pytyr_hgraph import PyTyrHGraphRuntime

    reader, states, actions = _two_states(domain, problem)
    goals = list(reader.problem_snapshot().goals)

    color = PyTyrColorRuntime(
        reader._planning_task, mifrost._neutral_core.SemanticColorEncoderConfig()
    )
    _assert_encoding_deep_equal(
        color.encode(states[0], goals=goals),
        color.engine.encode(color._input(states[0], goals=goals)),
    )
    _assert_encoding_deep_equal(
        color.encode_batch(states, goals=goals),
        color.engine.encode_batch(
            [color._input(state, goals=goals) for state in states]
        ),
    )

    hgraph = PyTyrHGraphRuntime(
        reader._planning_task, mifrost._neutral_core.SemanticHGraphEncoderConfig()
    )
    lanes: dict[str, Any] = {"goals": goals, "actions": actions}
    _assert_encoding_deep_equal(
        hgraph.encode(states[0], **lanes),
        hgraph.engine.encode(hgraph._input(states[0], **lanes)),
    )
    _assert_encoding_deep_equal(
        hgraph.encode_batch(states, goals=goals),
        hgraph.engine.encode_batch(
            [hgraph._input(state, goals=goals) for state in states]
        ),
    )


@pytest.mark.parametrize("family", ["color", "hgraph"])
def test_color_and_hgraph_streams_flush_prepared_handles(family: str) -> None:
    from mifrost.backends.pytyr_color import PyTyrColorRuntime
    from mifrost.backends.pytyr_hgraph import PyTyrHGraphRuntime

    reader, states, _actions = _two_states("blocks", "small")
    if family == "color":
        runtime: Any = PyTyrColorRuntime(
            reader._planning_task, mifrost._neutral_core.SemanticColorEncoderConfig()
        )
        stream = runtime.make_stream()
    else:
        runtime = PyTyrHGraphRuntime(
            reader._planning_task, mifrost._neutral_core.SemanticHGraphEncoderConfig()
        )
        stream = runtime.make_stream(mutable=True)

    for state in states:
        stream.append(state)
    expected = runtime.engine.encode_batch([runtime._input(s) for s in states])

    _assert_encoding_deep_equal(stream.flush(), expected)
    # Prepared handles are borrowed, not consumed: a second flush repeats.
    _assert_encoding_deep_equal(stream.flush(), expected)
