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
    _pymimir_reader, problem, reader, pytyr_search = _backend_pair()
    pymimir_state = problem.get_initial_state()
    pytyr_state = pytyr_search.initial_node().get_state()
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
    _pymimir_reader, _problem, reader, pytyr_search = _backend_pair()
    state = pytyr_search.initial_node().get_state()
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
    _pymimir_reader, problem, reader, pytyr_search = _backend_pair()
    pymimir_state = problem.get_initial_state()
    pytyr_state = pytyr_search.initial_node().get_state()
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


def test_public_pymimir_color_default_batch_uses_narrow_wrapper_fast_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import mifrost.backends.pymimir_color as pymimir_color

    _pymimir_reader, problem, _reader, _pytyr_search = _backend_pair()
    state = problem.get_initial_state()
    advanced_state = state._advanced_state
    encoder = mifrost.ColorEncoder(problem.get_domain())
    original_prepare = pymimir_color.prepare_core_batch_inputs
    prepared_shapes: list[tuple[Any, Any, Any, Any]] = []

    def recording_prepare(
        states: Any,
        *,
        goals: Any = None,
        actions: Any = None,
        subgoal_layers: Any = None,
        **kwargs: Any,
    ) -> Any:
        prepared_shapes.append((states, goals, actions, subgoal_layers))
        return original_prepare(
            states,
            goals=goals,
            actions=actions,
            subgoal_layers=subgoal_layers,
            **kwargs,
        )

    monkeypatch.setattr(pymimir_color, "prepare_core_batch_inputs", recording_prepare)

    fast = encoder.encode_batch([state, state])
    assert fast.num_graphs == 2
    assert prepared_shapes == []

    mixed = encoder.encode_batch([state, advanced_state])
    assert mixed.num_graphs == 2
    assert len(prepared_shapes) == 1

    goals = list(problem.get_goal_condition().get_literals())
    explicit = encoder.encode_batch([state, state], goals=goals)
    assert explicit.num_graphs == 2
    assert prepared_shapes[-1][1] is goals


def test_public_pymimir_color_stream_default_bypasses_optional_payload_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _pymimir_reader, problem, _reader, _pytyr_search = _backend_pair()
    state = problem.get_initial_state()
    encoder = mifrost.ColorEncoder(problem.get_domain())
    runtime: Any = encoder._runtime
    original_single_payload = runtime._single_payload
    prepared_payloads: list[tuple[Any, Any, Any]] = []

    def recording_single_payload(
        state: Any,
        *,
        goals: Any = None,
        actions: Any = None,
        subgoal_layers: Any = None,
    ) -> Any:
        prepared_payloads.append((goals, actions, subgoal_layers))
        return original_single_payload(
            state,
            goals=goals,
            actions=actions,
            subgoal_layers=subgoal_layers,
        )

    monkeypatch.setattr(runtime, "_single_payload", recording_single_payload)
    stream = encoder.stream()

    default_id = stream.append(state)
    stream.update(default_id, state)
    assert prepared_payloads == []

    goals = list(problem.get_goal_condition().get_literals())
    explicit_id = stream.append(state, goals=goals)
    stream.update(explicit_id, state, subgoal_layers=[goals])
    assert prepared_payloads == [(goals, None, None), (None, None, [goals])]
    assert stream.flush().num_graphs == 2


def test_public_color_rejects_mixed_or_explicit_wrong_backend() -> None:
    _pymimir_reader, problem, reader, pytyr_search = _backend_pair()
    pytyr_state = pytyr_search.initial_node().get_state()
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
    _pymimir_reader, _problem, reader, pytyr_search = _backend_pair()
    root = pytyr_search.initial_node()
    state = root.get_state()
    action = pytyr_search.action(pytyr_search.successors(root)[0])
    encoder = mifrost.ColorEncoder(reader._planning_task)

    with pytest.raises(ValueError, match="does not support action encoding"):
        encoder.encode(state, actions=[action])
    with pytest.raises(ValueError, match="does not support explicit action payloads"):
        encoder.encode_batch([state], actions=[action])
