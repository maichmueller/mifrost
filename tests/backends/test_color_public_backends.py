from __future__ import annotations

from typing import Any

import pytest

import mifrost
from mifrost.encoders.types import BatchParam

from tests.encoding.test_semantic_color_encoder import (
    _edge_color_partition,
    _named_edges,
    _node_color_partition,
)

from .test_semantic_parity import _backend_pair


def _assert_color_semantically_equal(
    actual: Any, expected: Any, *, edge_features: bool
) -> None:
    actual_data = actual.as_pyg()
    expected_data = expected.as_pyg()
    assert set(actual_data.node_names) == set(expected_data.node_names)
    assert _named_edges(actual_data) == _named_edges(expected_data)
    if edge_features:
        assert _edge_color_partition(actual_data) == _edge_color_partition(
            expected_data
        )
    else:
        assert _node_color_partition(actual_data) == _node_color_partition(
            expected_data
        )


@pytest.mark.parametrize("edge_features", [False, True])
@pytest.mark.parametrize("predicate_nodes", [False, True])
def test_public_color_backends_match_all_configurations(
    edge_features: bool, predicate_nodes: bool
) -> None:
    _pymimir_reader, problem, reader, successor_generator = _backend_pair()
    pymimir_state = problem.get_initial_state()
    pytyr_state = successor_generator.get_initial_node().get_state()
    kwargs = {
        "edge_features": edge_features,
        "enable_global_predicate_nodes": predicate_nodes,
    }
    pymimir_encoder = mifrost.ColorEncoder(problem.get_domain(), **kwargs)
    pytyr_encoder = mifrost.ColorEncoder(reader._planning_task, **kwargs)

    assert pymimir_encoder.backend == "pymimir"
    assert pytyr_encoder.backend == "pytyr"
    _assert_color_semantically_equal(
        pytyr_encoder.encode(pytyr_state),
        pymimir_encoder.encode(pymimir_state),
        edge_features=edge_features,
    )


def test_public_color_pytyr_batch_and_stream_lanes() -> None:
    _pymimir_reader, _problem, reader, successor_generator = _backend_pair()
    state = successor_generator.get_initial_node().get_state()
    goals = list(reader.problem_snapshot().goals)
    encoder = mifrost.ColorEncoder(reader._planning_task)

    batch = encoder.encode_batch(
        [state, state],
        goals=BatchParam.separate([goals, []]),
        subgoal_layers=BatchParam.shared([goals]),
    )
    assert batch.num_graphs == 2

    stream = encoder.stream()
    first = stream.append(state, goals=goals)
    second = stream.append(state, goals=[])
    stream.update(first, state, goals=[])
    stream.remove(second)
    assert stream.flush().num_graphs == 1

    stream = encoder.stream()
    stream.set_reuse_removed(True)
    removed = stream.append(state)
    stream.remove(removed)
    assert stream.append(state) == removed


def test_public_color_backends_coexist_without_global_selection() -> None:
    _pymimir_reader, problem, reader, successor_generator = _backend_pair()
    pymimir_state = problem.get_initial_state()
    pytyr_state = successor_generator.get_initial_node().get_state()
    pymimir_encoder = mifrost.ColorEncoder(problem.get_domain())
    pytyr_encoder = mifrost.ColorEncoder(reader._planning_task)

    retained = []
    for _ in range(3):
        pymimir_encoding = pymimir_encoder.encode(pymimir_state)
        pytyr_encoding = pytyr_encoder.encode(pytyr_state)
        _assert_color_semantically_equal(
            pytyr_encoding, pymimir_encoding, edge_features=False
        )
        retained.extend((pymimir_encoding, pytyr_encoding))
    assert len(retained) == 6


def test_public_color_rejects_mixed_or_explicit_wrong_backend() -> None:
    _pymimir_reader, problem, reader, successor_generator = _backend_pair()
    pytyr_state = successor_generator.get_initial_node().get_state()
    pymimir_state = problem.get_initial_state()
    encoder = mifrost.ColorEncoder(reader._planning_task)

    with pytest.raises(TypeError, match="only PyTyr states"):
        encoder.encode_batch([pytyr_state, pymimir_state])
    with pytest.raises(TypeError, match="expects a lifted or ground PyTyr state"):
        encoder.encode(pymimir_state)
    with pytest.raises(TypeError, match="PlanningTask"):
        mifrost.ColorEncoder(problem.get_domain(), backend="pytyr")
    with pytest.raises(TypeError, match="Unsupported domain type"):
        mifrost.ColorEncoder(reader._planning_task, backend="pymimir")
    with pytest.raises(ValueError, match="backend must be"):
        mifrost.ColorEncoder(reader._planning_task, backend="other")


def test_public_color_pytyr_preserves_action_errors() -> None:
    _pymimir_reader, _problem, reader, successor_generator = _backend_pair()
    root = successor_generator.get_initial_node()
    state = root.get_state()
    action = successor_generator.get_labeled_successor_nodes(root)[0].label
    encoder = mifrost.ColorEncoder(reader._planning_task)

    with pytest.raises(ValueError, match="does not support action encoding"):
        encoder.encode(state, actions=[action])
    with pytest.raises(ValueError, match="does not support explicit action payloads"):
        encoder.encode_batch([state], actions=[action])
