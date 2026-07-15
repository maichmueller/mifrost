from __future__ import annotations

from collections import Counter
from typing import Any

import pytest

import mifrost
from mifrost.backends.flat import FlatSemanticAdapter
from mifrost.backends.pymimir import PymimirSnapshotReader

from .test_semantic_hgraph_encoder import (
    _assert_hgraph_parity,
    _assert_pyg_values_equal,
    _canonical_symbol,
)
from .test_utils import adv_domain, adv_state, goal_inputs_from_problem


def _transition(space: Any, state: Any) -> tuple[Any, Any]:
    transitions = list(space.get_forward_transitions(state))
    if not transitions:
        pytest.skip("fixture does not expose an immediate successor")
    return transitions[0]


def _mode_name(mode: Any) -> str:
    core = mifrost._neutral_core
    if mode == core.SemanticSuccessorEncoderMode.full:
        return "full"
    return "delta"


def _facade(domain: Any, mode: Any, *, successor_suffix: str, **config: Any) -> Any:
    if _mode_name(mode) == "full":
        return mifrost.TransitionHGraphEncoder(
            domain, successor_suffix=successor_suffix, **config
        )
    return mifrost.TransitionEffectsHGraphEncoder(
        domain, successor_suffix=successor_suffix, **config
    )


def _semantic_engine(
    adapter: FlatSemanticAdapter,
    mode: Any,
    *,
    successor_suffix: str,
    **config: Any,
) -> Any:
    core = mifrost._neutral_core
    return core.SemanticSuccessorHGraphEncoderEngine(
        adapter.engine.predicates,
        adapter.engine.actions,
        core.SemanticSuccessorHGraphEncoderConfig(
            successor_mode=mode,
            successor_suffix=successor_suffix,
            **config,
        ),
    )


def _assert_facade_pair(
    domain: Any,
    problem: Any,
    current: Any,
    successor: Any,
    mode: Any,
    *,
    successor_suffix: str,
    config: dict[str, Any] | None = None,
    native_kwargs: dict[str, Any] | None = None,
    semantic_kwargs: dict[str, Any] | None = None,
) -> tuple[Any, Any, Any]:
    config = config or {}
    adapter = FlatSemanticAdapter(PymimirSnapshotReader(problem))
    native_encoder = _facade(domain, mode, successor_suffix=successor_suffix, **config)
    semantic_engine = _semantic_engine(
        adapter, mode, successor_suffix=successor_suffix, **config
    )
    native = native_encoder.encode(
        current, successor=successor, **(native_kwargs or {})
    ).as_pyg()
    semantic = semantic_engine.encode(
        adapter.make_input(current, **(semantic_kwargs or {})),
        adapter.make_input(successor),
    ).as_pyg()
    _assert_hgraph_parity(
        native,
        semantic,
        native_relation_arities=dict(native_encoder.relation_dict),
        semantic_relation_arities=dict(semantic_engine.relation_arities),
    )
    return native, semantic, semantic_engine


def _assert_batch_hgraph_parity(
    native: Any,
    semantic: Any,
    *,
    native_relation_arities: dict[str, int],
    semantic_relation_arities: dict[str, int],
) -> None:
    assert semantic_relation_arities == native_relation_arities
    assert semantic.num_graphs == native.num_graphs
    assert set(semantic.node_types) == set(native.node_types)
    assert set(semantic.edge_types) == set(native.edge_types)

    def node_names(data: Any, node_type: str) -> list[str]:
        flattened: list[Any] = []
        for item in data[node_type].node_names:
            if isinstance(item, (list, tuple)):
                flattened.extend(item)
            else:
                flattened.append(item)
        return [
            _canonical_symbol(str(name)) if node_type == "_symbol_" else str(name)
            for name in flattened
        ]

    for node_type in native.node_types:
        assert semantic[node_type].num_nodes == native[node_type].num_nodes
        assert semantic[node_type].ptr.equal(native[node_type].ptr)
        native_names = node_names(native, node_type)
        semantic_names = node_names(semantic, node_type)
        assert len(native_names) == len(native[node_type].x)
        assert len(semantic_names) == len(semantic[node_type].x)
        for graph_index in range(native.num_graphs):
            native_indices = [
                index
                for index, batch in enumerate(native[node_type].batch.tolist())
                if int(batch) == graph_index
            ]
            semantic_indices = [
                index
                for index, batch in enumerate(semantic[node_type].batch.tolist())
                if int(batch) == graph_index
            ]
            assert Counter(
                (
                    semantic_names[index],
                    tuple(semantic[node_type].x[index].tolist()),
                )
                for index in semantic_indices
            ) == Counter(
                (native_names[index], tuple(native[node_type].x[index].tolist()))
                for index in native_indices
            )

    for edge_type in native.edge_types:
        source_type, relation, target_type = edge_type
        native_sources = node_names(native, source_type)
        native_targets = node_names(native, target_type)
        semantic_sources = node_names(semantic, source_type)
        semantic_targets = node_names(semantic, target_type)

        def named_edges(data: Any, sources: list[str], targets: list[str]):
            result: Counter[Any] = Counter()
            for source, target in data[edge_type].edge_index.t().tolist():
                source_graph_index = int(data[source_type].batch[int(source)])
                target_graph_index = int(data[target_type].batch[int(target)])
                result[
                    (
                        source_graph_index,
                        target_graph_index,
                        sources[int(source)],
                        relation,
                        targets[int(target)],
                    )
                ] += 1
            return result

        assert named_edges(semantic, semantic_sources, semantic_targets) == named_edges(
            native, native_sources, native_targets
        )


@pytest.mark.parametrize("mode_name", ["full", "delta"])
def test_semantic_successor_matches_public_facades_on_representative_fixtures(
    small_blocks: tuple[Any, Any, Any], mode_name: str
) -> None:
    space, domain, problem = small_blocks
    current = problem.get_initial_state()
    _action, successor = _transition(space, current)
    core = mifrost._neutral_core
    mode = getattr(core.SemanticSuccessorEncoderMode, mode_name)
    suffix = "[suc]" if mode_name == "full" else ""

    _assert_facade_pair(
        domain,
        problem,
        current,
        successor,
        mode,
        successor_suffix=suffix,
    )


@pytest.mark.parametrize("mode_name", ["full", "delta"])
@pytest.mark.parametrize(
    "config",
    [
        {"include_static": False, "include_empty_edge_types": False},
        {"add_nullary_predicates": True},
        {"export_node_names": True, "support_literals": True},
    ],
)
def test_semantic_successor_matches_policy_modes(
    small_blocks: tuple[Any, Any, Any], mode_name: str, config: dict[str, Any]
) -> None:
    space, domain, problem = small_blocks
    current = problem.get_initial_state()
    _action, successor = _transition(space, current)
    mode = getattr(mifrost._neutral_core.SemanticSuccessorEncoderMode, mode_name)

    _assert_facade_pair(
        domain,
        problem,
        current,
        successor,
        mode,
        successor_suffix="[next]",
        config=config,
    )


def test_semantic_successor_matches_goals_subgoals_and_successor_satisfaction(
    small_blocks: tuple[Any, Any, Any],
) -> None:
    space, domain, problem = small_blocks
    current = problem.get_initial_state()
    _action, successor = _transition(space, current)
    reader = PymimirSnapshotReader(problem)
    native_goals = list(problem.get_goal_condition().get_literals())
    semantic_goals = list(reader.problem_snapshot().goals)
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

    native, semantic, _engine = _assert_facade_pair(
        domain,
        problem,
        current,
        successor,
        mifrost._neutral_core.SemanticSuccessorEncoderMode.full,
        successor_suffix="[after]",
        config=config,
        native_kwargs={
            "goals": native_goals,
            "subgoal_layers": [native_goals],
        },
        semantic_kwargs={
            "goals": semantic_goals,
            "subgoal_layers": [semantic_goals],
        },
    )

    assert any(
        native[node_type].num_nodes > 0
        and "[after]" in node_type
        and ("[sat]" in node_type or "[unsat]" in node_type)
        for node_type in native.node_types
    )
    assert set(native.node_types) == set(semantic.node_types)


def test_semantic_delta_forces_literal_support_and_omits_satisfaction(
    small_blocks: tuple[Any, Any, Any],
) -> None:
    space, domain, problem = small_blocks
    current = problem.get_initial_state()
    _action, successor = _transition(space, current)
    config = {
        "support_literals": False,
        "goal_derivations": {
            mifrost.GoalDerivation.plain,
            mifrost.GoalDerivation.satisfied,
            mifrost.GoalDerivation.unsatisfied,
        },
    }
    native, semantic, engine = _assert_facade_pair(
        domain,
        problem,
        current,
        successor,
        mifrost._neutral_core.SemanticSuccessorEncoderMode.delta,
        successor_suffix="[effects]",
        config=config,
    )

    assert engine.config.support_literals is True
    assert not any(
        native[node_type].num_nodes > 0
        and "[effects]" in node_type
        and ("[sat]" in node_type or "[unsat]" in node_type)
        for node_type in native.node_types
    )
    assert not any(
        semantic[node_type].num_nodes > 0
        and "[effects]" in node_type
        and ("[sat]" in node_type or "[unsat]" in node_type)
        for node_type in semantic.node_types
    )
    adapter = FlatSemanticAdapter(PymimirSnapshotReader(problem))
    dynamic_predicates = {
        index
        for index, predicate in enumerate(adapter.engine.predicates)
        if predicate.category != mifrost.SemanticPredicateCategory.static
    }

    def dynamic_facts(state: Any) -> set[tuple[int, tuple[int, ...]]]:
        return {
            (int(atom.predicate), tuple(int(argument) for argument in atom.arguments))
            for atom in adapter.make_input(state).state_facts
            if int(atom.predicate) in dynamic_predicates
        }

    current_facts = dynamic_facts(current)
    successor_facts = dynamic_facts(successor)
    emitted_effect_nodes = [
        node_type
        for node_type in semantic.node_types
        if "[effects]" in node_type and semantic[node_type].num_nodes > 0
    ]
    assert bool(emitted_effect_nodes) == (current_facts != successor_facts)


@pytest.mark.parametrize("mode_name", ["full", "delta"])
def test_semantic_successor_preserves_goal_targets_and_lgan(
    small_blocks: tuple[Any, Any, Any], mode_name: str
) -> None:
    space, domain, problem = small_blocks
    current = problem.get_initial_state()
    _action, successor = _transition(space, current)
    adapter = FlatSemanticAdapter(PymimirSnapshotReader(problem))
    core = mifrost._neutral_core
    semantic_mode = getattr(core.SemanticSuccessorEncoderMode, mode_name)
    native_mode = getattr(mifrost.SuccessorEncoderMode, mode_name)
    suffix = "[target]"

    native_config = mifrost.SuccessorEncoderConfig()
    native_config.successor_mode = native_mode
    native_config.successor_suffix = suffix
    native_config.include_lgan_edges = True
    native_config.target_sources = {mifrost.TargetSource.goals}
    native_engine = mifrost.SuccessorHGraphEncoderEngine(
        adv_domain(domain), native_config
    )
    semantic_engine = _semantic_engine(
        adapter,
        semantic_mode,
        successor_suffix=suffix,
        include_lgan_edges=True,
        target_sources={mifrost.TargetSource.goals},
    )

    native = native_engine.encode(
        adv_state(current),
        adv_state(successor),
        goal_inputs_from_problem(problem),
    ).as_pyg()
    semantic = semantic_engine.encode(
        adapter.make_input(current), adapter.make_input(successor)
    ).as_pyg()
    _assert_hgraph_parity(
        native,
        semantic,
        native_relation_arities=dict(native_engine.relation_dict),
        semantic_relation_arities=dict(semantic_engine.relation_arities),
    )
    assert list(native.target_groups) == list(semantic.target_groups) == ["goal"]
    assert any(
        edge_type[1] == native_config.lgan_tn_edge_pos
        for edge_type in native.edge_types
    )


@pytest.mark.parametrize("mode_name", ["full", "delta"])
def test_semantic_successor_batch_and_owned_builder_match_facades(
    small_blocks: tuple[Any, Any, Any], mode_name: str
) -> None:
    space, domain, problem = small_blocks
    current = problem.get_initial_state()
    transitions = list(space.get_forward_transitions(current))
    if not transitions:
        pytest.skip("fixture does not expose an immediate successor")
    successors = [transition[1] for transition in transitions[:2]]
    if len(successors) == 1:
        successors.append(successors[0])
    currents = [current, current]
    core = mifrost._neutral_core
    mode = getattr(core.SemanticSuccessorEncoderMode, mode_name)
    suffix = "[batch]"
    adapter = FlatSemanticAdapter(PymimirSnapshotReader(problem))
    facade = _facade(domain, mode, successor_suffix=suffix)
    engine = _semantic_engine(adapter, mode, successor_suffix=suffix)
    current_inputs = [adapter.make_input(value) for value in currents]
    successor_inputs = [adapter.make_input(value) for value in successors]

    native = facade.encode_batch(currents, successors=successors).as_pyg(as_batch=True)
    semantic = engine.encode_batch(current_inputs, successor_inputs).as_pyg(
        as_batch=True
    )
    _assert_batch_hgraph_parity(
        native,
        semantic,
        native_relation_arities=dict(facade.relation_dict),
        semantic_relation_arities=dict(engine.relation_arities),
    )

    builder = mifrost.BatchBuilder()
    for current_input, successor_input in zip(
        current_inputs, successor_inputs, strict=True
    ):
        engine.encode(current_input, successor_input, builder)
        builder.next_graph()
    composed = builder.build().as_pyg(as_batch=True)
    _assert_pyg_values_equal(composed, semantic)
    for node_type in semantic.node_types:
        assert composed[node_type].ptr.equal(semantic[node_type].ptr)


def test_semantic_successor_rejects_misaligned_objects_and_batch_shapes(
    small_blocks: tuple[Any, Any, Any],
) -> None:
    space, _domain, problem = small_blocks
    current = problem.get_initial_state()
    _action, successor = _transition(space, current)
    adapter = FlatSemanticAdapter(PymimirSnapshotReader(problem))
    engine = _semantic_engine(
        adapter,
        mifrost._neutral_core.SemanticSuccessorEncoderMode.full,
        successor_suffix="[suc]",
    )
    current_input = adapter.make_input(current)
    successor_input = adapter.make_input(successor)
    successor_input.objects = list(reversed(successor_input.objects))

    with pytest.raises(ValueError, match="identical ordered object tables"):
        engine.encode(current_input, successor_input)
    with pytest.raises(ValueError, match="equal length"):
        engine.encode_batch([current_input], [])


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda value: setattr(
                value,
                "state_facts",
                [*value.state_facts, mifrost.SemanticAtom(999, [])],
            ),
            "predicate index out of range",
        ),
        (
            lambda value: setattr(
                value,
                "state_facts",
                [*value.state_facts, mifrost.SemanticAtom(0, [])],
            ),
            "argument count",
        ),
    ],
)
def test_semantic_successor_validates_successor_state_shape(
    small_blocks: tuple[Any, Any, Any], mutate: Any, message: str
) -> None:
    space, _domain, problem = small_blocks
    current = problem.get_initial_state()
    _action, successor = _transition(space, current)
    adapter = FlatSemanticAdapter(PymimirSnapshotReader(problem))
    engine = _semantic_engine(
        adapter,
        mifrost._neutral_core.SemanticSuccessorEncoderMode.full,
        successor_suffix="[suc]",
    )
    successor_input = adapter.make_input(successor)
    mutate(successor_input)

    with pytest.raises(ValueError, match=message):
        engine.encode(adapter.make_input(current), successor_input)
