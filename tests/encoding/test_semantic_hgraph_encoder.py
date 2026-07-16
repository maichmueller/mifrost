from __future__ import annotations

from collections import Counter
from textwrap import dedent
from typing import Any

import pytest
import torch

import mifrost
from mifrost.backends.flat import FlatSemanticAdapter
from mifrost.backends.pymimir import PymimirSnapshotReader


def _semantic_engine(adapter: FlatSemanticAdapter, **config: Any) -> Any:
    core = mifrost._neutral_core
    return core.SemanticHGraphEncoderEngine(
        adapter.engine.predicates,
        adapter.engine.actions,
        core.SemanticHGraphEncoderConfig(**config),
    )


def _canonical_symbol(name: str) -> str:
    """Remove only backend-local repository identity from target symbol keys."""
    if not name.startswith("target:") or "|" not in name:
        return name
    parts = name.split("|")
    if parts[0] in {"target:goal", "target:subgoal"} and len(parts) == 6:
        parts[3] = "*"
    elif parts[0] == "target:history" and len(parts) == 7:
        parts[5] = "*"
    elif parts[0].removeprefix("target:").isdigit():
        parts[0] = "target:*"
    else:
        return name
    return "|".join(parts)


def _node_names(data: Any, node_type: str) -> list[str]:
    return [
        _canonical_symbol(str(name)) if node_type == "_symbol_" else str(name)
        for name in data[node_type].node_names
    ]


def _named_edges(data: Any, edge_type: tuple[str, str, str]) -> Counter[Any]:
    source_type, relation, target_type = edge_type
    source_names = _node_names(data, source_type)
    target_names = _node_names(data, target_type)
    return Counter(
        (
            source_names[int(source)],
            relation,
            target_names[int(target)],
        )
        for source, target in data[edge_type].edge_index.t().tolist()
    )


def _assert_hgraph_parity(
    native: Any,
    semantic: Any,
    *,
    native_relation_arities: dict[str, int],
    semantic_relation_arities: dict[str, int],
) -> None:
    assert semantic_relation_arities == native_relation_arities
    assert set(semantic.node_types) == set(native.node_types)
    assert set(semantic.edge_types) == set(native.edge_types)

    for node_type in native.node_types:
        assert semantic[node_type].num_nodes == native[node_type].num_nodes
        native_names = _node_names(native, node_type)
        semantic_names = _node_names(semantic, node_type)
        assert Counter(semantic_names) == Counter(native_names)
        assert Counter(
            (name, tuple(row))
            for name, row in zip(
                semantic_names, semantic[node_type].x.tolist(), strict=True
            )
        ) == Counter(
            (name, tuple(row))
            for name, row in zip(
                native_names, native[node_type].x.tolist(), strict=True
            )
        )

    for edge_type in native.edge_types:
        assert _named_edges(semantic, edge_type) == _named_edges(native, edge_type)

    assert Counter(
        _canonical_symbol(name) for name in semantic.object_names
    ) == Counter(_canonical_symbol(name) for name in native.object_names)
    for field in ("target_sizes",):
        assert hasattr(semantic, field) == hasattr(native, field)
        if hasattr(native, field):
            assert torch.equal(getattr(semantic, field), getattr(native, field)), field
    if hasattr(native, "target_positions"):
        native_symbols = _node_names(native, "_symbol_")
        semantic_symbols = _node_names(semantic, "_symbol_")
        assert sorted(semantic.target_indices.tolist()) == sorted(
            native.target_indices.tolist()
        )
        assert sorted(semantic.target_candidate_ids.tolist()) == sorted(
            native.target_candidate_ids.tolist()
        )
        semantic_rows = Counter(
            (semantic_symbols[int(position)], str(name), int(group))
            for position, name, group in zip(
                semantic.target_positions,
                semantic.target_names,
                semantic.target_group_ids,
                strict=True,
            )
        )
        native_rows = Counter(
            (native_symbols[int(position)], str(name), int(group))
            for position, name, group in zip(
                native.target_positions,
                native.target_names,
                native.target_group_ids,
                strict=True,
            )
        )
        assert semantic_rows == native_rows
        assert list(semantic.target_groups) == list(native.target_groups)


def _assert_pair(
    domain: Any,
    problem: Any,
    state: Any,
    adapter: FlatSemanticAdapter,
    semantic_input: Any,
    *,
    config: dict[str, Any],
    native_kwargs: dict[str, Any] | None = None,
) -> tuple[Any, Any]:
    native_encoder = mifrost.HGraphEncoder(domain, **config)
    semantic_engine = _semantic_engine(adapter, **config)
    native = native_encoder.encode(state, **(native_kwargs or {})).as_pyg()
    semantic = semantic_engine.encode(semantic_input).as_pyg()
    _assert_hgraph_parity(
        native,
        semantic,
        native_relation_arities=dict(native_encoder.relation_dict.items()),
        semantic_relation_arities=dict(semantic_engine.relation_arities),
    )
    return native, semantic


@pytest.mark.parametrize(
    "config",
    [
        {},
        {"include_static": False, "include_empty_edge_types": False},
        {"add_nullary_predicates": True},
    ],
)
def test_semantic_hgraph_matches_native_base_modes(
    small_blocks: tuple[Any, Any, Any], config: dict[str, Any]
) -> None:
    _space, domain, problem = small_blocks
    state = problem.get_initial_state()
    adapter = FlatSemanticAdapter(PymimirSnapshotReader(problem))

    native, semantic = _assert_pair(
        domain,
        problem,
        state,
        adapter,
        adapter.make_input(state),
        config=config,
    )
    if config.get("add_nullary_predicates"):
        assert "![nullary_symbol]!" in native.object_names
        assert "![nullary_symbol]!" in semantic.object_names
        if "handempty" in native.node_types:
            assert native["handempty"].node_names == semantic["handempty"].node_names


def test_semantic_hgraph_preserves_goal_derivations_and_duplicate_lane_targets(
    small_blocks: tuple[Any, Any, Any],
) -> None:
    _space, domain, problem = small_blocks
    state = problem.get_initial_state()
    reader = PymimirSnapshotReader(problem)
    adapter = FlatSemanticAdapter(reader)
    native_goals = list(problem.get_goal_condition().get_literals())
    semantic_goals = list(reader.problem_snapshot().goals)
    config = {
        "max_goal_level": 1,
        "support_literals": True,
        "target_sources": {
            mifrost.TargetSource.goals,
            mifrost.TargetSource.subgoals,
        },
        "goal_derivations": {
            mifrost.GoalDerivation.plain,
            mifrost.GoalDerivation.satisfied,
            mifrost.GoalDerivation.unsatisfied,
        },
    }

    native, semantic = _assert_pair(
        domain,
        problem,
        state,
        adapter,
        adapter.make_input(
            state,
            goals=semantic_goals,
            subgoal_layers=[semantic_goals],
        ),
        config=config,
        native_kwargs={"goals": native_goals, "subgoal_layers": [native_goals]},
    )

    assert Counter(native.target_names) == Counter(semantic.target_names)
    assert len(native.target_names) == 2 * len(native_goals)
    assert all(count == 2 for count in Counter(native.target_names).values())
    assert set(native.target_group_ids.tolist()) == {1}
    assert Counter(native.target_group_ids.tolist()) == Counter(
        semantic.target_group_ids.tolist()
    )
    assert any(node_type.endswith("[sg][sat]") for node_type in native.node_types)
    assert any(node_type.endswith("[sg][unsat]") for node_type in native.node_types)


def _applicable_actions(space: Any, state: Any, count: int = 1) -> list[Any]:
    result = [action for action, _ in space.get_forward_transitions(state) if action]
    if len(result) < count:
        pytest.skip("fixture does not have enough applicable actions")
    return result[:count]


def test_semantic_hgraph_preserves_action_target_offset_lgan_and_duplicates(
    small_blocks: tuple[Any, Any, Any],
) -> None:
    space, domain, problem = small_blocks
    state = problem.get_initial_state()
    adapter = FlatSemanticAdapter(PymimirSnapshotReader(problem))
    action = _applicable_actions(space, state)[0]
    config = {
        "ignore_actions": False,
        "include_lgan_edges": True,
        "target_sources": {mifrost.TargetSource.actions},
    }
    native, semantic = _assert_pair(
        domain,
        problem,
        state,
        adapter,
        adapter.make_input(state, actions=[action, action]),
        config=config,
        native_kwargs={"actions": [action, action]},
    )

    action_name = list(native.target_names)[0]
    action_type = action_name.removeprefix("(").split(" ", 1)[0]
    native_encoder = mifrost.HGraphEncoder(domain, **config)
    semantic_engine = _semantic_engine(adapter, **config)
    action_arity = next(
        spec.arity for spec in adapter.engine.actions if spec.name == action_type
    )
    assert native_encoder.relation_dict[action_type] == action_arity + 1
    assert semantic_engine.relation_arities[action_type] == action_arity + 1
    assert (
        Counter(native.target_names)
        == Counter(semantic.target_names)
        == Counter([action_name, action_name])
    )
    assert native.target_positions.tolist() == semantic.target_positions.tolist()
    assert len(set(native.target_positions.tolist())) == 1
    assert (
        len(native.object_names)
        == len(semantic.object_names)
        == len(adapter.make_input(state).objects) + 1
    )


def test_semantic_hgraph_lgan_goal_anchor_does_not_emit_target_metadata(
    small_blocks: tuple[Any, Any, Any],
) -> None:
    _space, domain, problem = small_blocks
    state = problem.get_initial_state()
    adapter = FlatSemanticAdapter(PymimirSnapshotReader(problem))
    config = {
        "include_lgan_edges": True,
        "lgan_anchor_sources": {mifrost.TargetSource.goals},
    }
    native, semantic = _assert_pair(
        domain,
        problem,
        state,
        adapter,
        adapter.make_input(state),
        config=config,
    )

    assert not hasattr(native, "target_positions")
    assert not hasattr(semantic, "target_positions")
    assert any(name.startswith("target:goal|") for name in native.object_names)
    assert any(name.startswith("target:goal|") for name in semantic.object_names)


def test_semantic_hgraph_keeps_distinct_action_schemas_as_distinct_targets(
    small_blocks: tuple[Any, Any, Any],
) -> None:
    space, domain, problem = small_blocks
    state = problem.get_initial_state()
    reader = PymimirSnapshotReader(problem)
    adapter = FlatSemanticAdapter(reader)
    by_schema: dict[str, Any] = {}
    for action, _ in space.get_forward_transitions(state):
        if action is not None:
            by_schema.setdefault(reader.action_key(action).action.name, action)
    if len(by_schema) < 2:
        pytest.skip("fixture does not expose two applicable action schemas")
    actions = list(by_schema.values())[:2]
    config = {
        "ignore_actions": False,
        "target_sources": {mifrost.TargetSource.actions},
    }

    native, semantic = _assert_pair(
        domain,
        problem,
        state,
        adapter,
        adapter.make_input(state, actions=actions),
        config=config,
        native_kwargs={"actions": actions},
    )
    assert len(set(native.target_positions.tolist())) == 2
    assert len(set(semantic.target_positions.tolist())) == 2
    assert (
        len(native.object_names)
        == len(semantic.object_names)
        == len(adapter.make_input(state).objects) + 2
    )


def test_semantic_hgraph_preserves_history_targets_and_filtering(
    small_blocks: tuple[Any, Any, Any],
) -> None:
    _space, domain, problem = small_blocks
    state = problem.get_initial_state()
    reader = PymimirSnapshotReader(problem)
    adapter = FlatSemanticAdapter(reader)
    native_goals = list(problem.get_goal_condition().get_literals())
    semantic_goals = list(reader.problem_snapshot().goals)
    config = {
        "support_literals": True,
        "include_lgan_edges": True,
        "target_sources": {mifrost.TargetSource.history},
    }
    native, semantic = _assert_pair(
        domain,
        problem,
        state,
        adapter,
        adapter.make_input(
            state,
            history=[(-1, semantic_goals), (-3, semantic_goals)],
            history_max_steps=2,
        ),
        config=config,
        native_kwargs={
            "history_subgoals": [(-1, native_goals), (-3, native_goals)],
            "history_max_steps": 2,
        },
    )
    assert torch.equal(native["history"].history_dt, semantic["history"].history_dt)
    assert native["history"].history_dt.flatten().tolist() == [-1.0]
    assert Counter(native.target_names) == Counter(semantic.target_names)
    assert len(native.target_names) == len(native_goals)
    assert all(name.startswith("history:-1#0:") for name in native.target_names)


def test_semantic_hgraph_preserves_legacy_zero_arity_action_format(
    tmp_path: Any,
) -> None:
    import pymimir

    domain_path = tmp_path / "domain.pddl"
    problem_path = tmp_path / "problem.pddl"
    domain_path.write_text(
        dedent(
            """
            (define (domain zero-action)
              (:requirements :strips)
              (:predicates (ready) (done))
              (:action finish
                :parameters ()
                :precondition (ready)
                :effect (done)))
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    problem_path.write_text(
        dedent(
            """
            (define (problem zero-action-problem)
              (:domain zero-action)
              (:init (ready))
              (:goal (done)))
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    domain = pymimir.Domain(domain_path)
    problem = pymimir.Problem(domain, problem_path, mode="lifted")
    state = problem.get_initial_state()
    action = list(state.generate_applicable_actions())[0]
    adapter = FlatSemanticAdapter(PymimirSnapshotReader(problem))
    config = {
        "ignore_actions": False,
        "target_sources": {mifrost.TargetSource.actions},
    }

    native, semantic = _assert_pair(
        domain,
        problem,
        state,
        adapter,
        adapter.make_input(state, actions=[action]),
        config=config,
        native_kwargs={"actions": [action]},
    )
    # RelationFormatter currently preserves this historical trailing space.
    assert list(native.target_names) == list(semantic.target_names) == ["(finish )"]


def _assert_pyg_values_equal(actual: Any, expected: Any) -> None:
    assert actual.to_dict().keys() == expected.to_dict().keys()
    for key, actual_store in actual.to_dict().items():
        expected_store = expected.to_dict()[key]
        if isinstance(actual_store, dict):
            assert actual_store.keys() == expected_store.keys()
            for field, value in actual_store.items():
                expected_value = expected_store[field]
                if torch.is_tensor(value):
                    assert torch.equal(value, expected_value), (key, field)
                else:
                    assert value == expected_value, (key, field)
        elif torch.is_tensor(actual_store):
            assert torch.equal(actual_store, expected_store), key
        else:
            assert actual_store == expected_store, key


def test_semantic_hgraph_batches_and_composes_with_owned_builder(
    small_blocks: tuple[Any, Any, Any],
) -> None:
    _space, _domain, problem = small_blocks
    state = problem.get_initial_state()
    adapter = FlatSemanticAdapter(PymimirSnapshotReader(problem))
    input_value = adapter.make_input(state)
    engine = _semantic_engine(adapter)

    batch = engine.encode_batch([input_value, input_value]).as_pyg(as_batch=True)
    builder = mifrost.BatchBuilder()
    engine.encode(input_value, builder)
    builder.next_graph()
    engine.encode(input_value, builder)
    builder.next_graph()
    composed = builder.build().as_pyg(as_batch=True)

    assert batch.num_graphs == composed.num_graphs == 2
    _assert_pyg_values_equal(composed, batch)
    for node_type in batch.node_types:
        assert torch.equal(composed[node_type].ptr, batch[node_type].ptr)


def _append_bad_object_atom(value: Any) -> None:
    source = next(atom for atom in value.state_facts if atom.arguments)
    value.state_facts = [
        *value.state_facts,
        mifrost.SemanticAtom(source.predicate, [999, *source.arguments[1:]]),
    ]


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda value: setattr(value, "objects", [*value.objects, value.objects[0]]),
            "unique object names",
        ),
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
        (
            _append_bad_object_atom,
            "object index out of range",
        ),
        (
            lambda value: setattr(
                value,
                "actions",
                [*value.actions, mifrost.SemanticGroundAction(999, [])],
            ),
            "action index out of range",
        ),
        (
            lambda value: setattr(
                value,
                "history",
                [*value.history, mifrost.SemanticHistoryEntry(0, [])],
            ),
            "negative dt",
        ),
    ],
)
def test_semantic_hgraph_rejects_invalid_semantic_inputs(
    small_blocks: tuple[Any, Any, Any], mutate: Any, message: str
) -> None:
    _space, _domain, problem = small_blocks
    state = problem.get_initial_state()
    adapter = FlatSemanticAdapter(PymimirSnapshotReader(problem))
    input_value = adapter.make_input(state)
    mutate(input_value)

    with pytest.raises(ValueError, match=message):
        _semantic_engine(adapter).encode(input_value)


def test_semantic_hgraph_rejects_unsupported_levels_and_missing_lgan_anchor(
    small_blocks: tuple[Any, Any, Any],
) -> None:
    _space, _domain, problem = small_blocks
    state = problem.get_initial_state()
    adapter = FlatSemanticAdapter(PymimirSnapshotReader(problem))
    input_value = adapter.make_input(state)
    input_value.subgoal_layers = [[]]
    with pytest.raises(ValueError, match="max_goal_level"):
        _semantic_engine(adapter).encode(input_value)

    with pytest.raises(ValueError, match="requires explicit target symbols"):
        _semantic_engine(adapter, include_lgan_edges=True).encode(
            adapter.make_input(state)
        )


def test_semantic_hgraph_relation_schema_can_be_replaced(
    small_blocks: tuple[Any, Any, Any],
) -> None:
    _space, _domain, problem = small_blocks
    adapter = FlatSemanticAdapter(PymimirSnapshotReader(problem))
    engine = _semantic_engine(adapter)

    engine.update_relations({"custom_rel": 2})
    assert dict(engine.relation_arities) == {"custom_rel": 2}

    with pytest.raises(ValueError, match="name must not be empty"):
        engine.update_relations({"": 1})
    with pytest.raises(ValueError, match="arity must be non-negative"):
        engine.update_relations({"bad": -1})
