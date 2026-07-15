from __future__ import annotations

import torch

import mifrost
from mifrost.backends.flat import FlatSemanticAdapter
from mifrost.backends.pymimir import PymimirSnapshotReader
from mifrost.backends.pytyr import PyTyrSnapshotReader

from .test_semantic_parity import _backend_pair


def _assert_payload_equal(actual, expected) -> None:
    actual_payload = actual.as_pyg().to_dict()
    expected_payload = expected.as_pyg().to_dict()
    assert actual_payload.keys() == expected_payload.keys()
    for key in actual_payload:
        if torch.is_tensor(actual_payload[key]):
            assert torch.equal(actual_payload[key], expected_payload[key]), key
        else:
            assert actual_payload[key] == expected_payload[key], key


def _normalized_relations(encoding) -> dict[str, tuple[tuple[str, ...], ...]]:
    """Compare flat relations by semantic names rather than backend indices."""

    data = encoding.as_pyg()
    node_names = list(data.node_names)
    relations: dict[str, tuple[tuple[str, ...], ...]] = {}
    for relation_name, rows in data.flattened_relations.items():
        named_rows = [
            tuple(node_names[int(entity_index)] for entity_index in row)
            for row in rows.tolist()
        ]
        relations[relation_name] = tuple(sorted(named_rows))
    return relations


def _normalized_targets(encoding) -> tuple[tuple[str, str, str, str], ...]:
    data = encoding.as_pyg()
    return tuple(
        sorted(
            (
                str(name),
                str(group),
                str(source),
                str(data.node_names[int(position)]),
            )
            for name, group, source, position in zip(
                data.target_names,
                (
                    data.target_groups[int(group_id)]
                    for group_id in data.target_group_ids.tolist()
                ),
                (
                    data.target_sources[int(group_id)]
                    for group_id in data.target_group_ids.tolist()
                ),
                data.target_positions.tolist(),
                strict=True,
            )
        )
    )


def _assert_semantically_equal(actual, expected) -> None:
    actual_data = actual.as_pyg()
    expected_data = expected.as_pyg()
    assert set(actual_data.node_names) == set(expected_data.node_names)
    assert set(actual_data.object_names) == set(expected_data.object_names)
    assert _normalized_relations(actual) == _normalized_relations(expected)
    assert _normalized_targets(actual) == _normalized_targets(expected)


def _config():
    return mifrost.FlatRelationEncoderConfig(
        ignore_zero_arity_relations=False,
        use_predicate_virtual_nodes=False,
        include_lgan_edges=False,
        target_sources={mifrost.TargetSource.actions, mifrost.TargetSource.goals},
    )


def test_semantic_adapters_match_native_pymimir_flat_encoding() -> None:
    pymimir_reader, pymimir_problem, pytyr_reader, successor_generator = _backend_pair()
    assert isinstance(pymimir_reader, PymimirSnapshotReader)
    assert isinstance(pytyr_reader, PyTyrSnapshotReader)
    pymimir_root = pymimir_problem.get_initial_state()
    pymimir_actions = list(pymimir_root.generate_applicable_actions())
    pytyr_root = successor_generator.get_initial_node()
    pytyr_actions = [
        labeled.label
        for labeled in successor_generator.get_labeled_successor_nodes(pytyr_root)
    ]

    native_engine = mifrost.FlatRelationEncoderEngine(
        pymimir_problem.get_domain()._advanced_domain, _config()
    )
    native_goals = mifrost.GoalInputs(
        [
            literal._advanced_ground_literal
            for literal in pymimir_problem.get_goal_condition().get_literals()
        ],
        0,
    )
    native = native_engine.encode(
        pymimir_root._advanced_state,
        native_goals,
        [action._advanced_ground_action for action in pymimir_actions],
    )

    pymimir_semantic = FlatSemanticAdapter(pymimir_reader, _config()).encode(
        pymimir_root, actions=pymimir_actions
    )
    pytyr_semantic = FlatSemanticAdapter(pytyr_reader, _config()).encode(
        pytyr_root.get_state(), actions=pytyr_actions
    )

    _assert_semantically_equal(pymimir_semantic, native)
    _assert_payload_equal(pytyr_semantic, pymimir_semantic)


def test_two_backend_semantic_engines_remain_independent_when_interleaved() -> None:
    pymimir_reader, pymimir_problem, pytyr_reader, successor_generator = _backend_pair()
    pymimir_adapter = FlatSemanticAdapter(pymimir_reader, _config())
    pytyr_adapter = FlatSemanticAdapter(pytyr_reader, _config())
    pymimir_root = pymimir_problem.get_initial_state()
    pytyr_root = successor_generator.get_initial_node().get_state()

    expected = pymimir_adapter.encode(pymimir_root)
    for _ in range(3):
        _assert_payload_equal(pytyr_adapter.encode(pytyr_root), expected)
        _assert_payload_equal(pymimir_adapter.encode(pymimir_root), expected)
