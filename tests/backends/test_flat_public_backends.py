from __future__ import annotations

from pathlib import Path
import subprocess
import sys
from typing import Any

import pytest
import torch

import mifrost
from mifrost.backends.pytyr import SemanticFlatRelationEncoder
from mifrost.encoders.types import BatchParam

from .test_native_pytyr_adapter import _config, _pytyr_pair


ROOT = Path(__file__).resolve().parents[2]


def _assert_encoding_equal(actual: Any, expected: Any) -> None:
    actual_payload = actual.as_pyg(as_batch=True).to_dict()
    expected_payload = expected.as_pyg(as_batch=True).to_dict()
    assert actual_payload.keys() == expected_payload.keys()
    for key in actual_payload:
        if torch.is_tensor(actual_payload[key]):
            assert torch.equal(actual_payload[key], expected_payload[key]), key
        else:
            assert actual_payload[key] == expected_payload[key], key


def test_public_flat_infers_pytyr_and_matches_native_adapter() -> None:
    reader, successor_generator = _pytyr_pair("blocks", "small")
    root = successor_generator.get_initial_node()
    state = root.get_state()
    actions = [
        successor.label
        for successor in successor_generator.get_labeled_successor_nodes(root)
    ]
    config = _config()
    encoder = mifrost.FlatRelationEncoder(
        reader._planning_task,
        ignore_zero_arity_relations=config.ignore_zero_arity_relations,
        target_sources=config.target_sources,
    )
    direct = SemanticFlatRelationEncoder(reader._planning_task, encoder.config)

    assert encoder.backend == "pytyr"
    assert dict(encoder.relation_dict) == dict(
        zip(encoder.relation_names, encoder.relation_arities, strict=True)
    )
    _assert_encoding_equal(
        encoder.encode(state, actions=actions),
        direct.encode(state, actions),
    )


def test_public_flat_pytyr_batch_supports_shared_and_separate_lanes() -> None:
    reader, successor_generator = _pytyr_pair("blocks", "small")
    root = successor_generator.get_initial_node()
    state = root.get_state()
    actions = [
        successor.label
        for successor in successor_generator.get_labeled_successor_nodes(root)
    ]
    goals = list(reader.problem_snapshot().goals)
    encoder = mifrost.FlatRelationEncoder(
        reader._planning_task,
        max_goal_level=1,
        target_sources={"goal", "subgoal", "action", "history"},
    )
    direct = SemanticFlatRelationEncoder(reader._planning_task, encoder.config)

    actual = encoder.encode_batch(
        [state, state],
        goals=BatchParam.separate([goals, []]),
        actions=actions,
        subgoal_layers=[goals],
        history_subgoals=[(-2, goals)],
        history_max_steps=4,
    )
    expected = direct.engine.encode_batch(
        [
            direct.make_input(
                state,
                actions,
                goals=goals,
                subgoal_layers=[goals],
                history=[(-2, goals)],
                history_max_steps=4,
            ),
            direct.make_input(
                state,
                actions,
                goals=[],
                subgoal_layers=[goals],
                history=[(-2, goals)],
                history_max_steps=4,
            ),
        ]
    )

    _assert_encoding_equal(actual, expected)


def test_public_flat_pytyr_streams_match_direct_batch() -> None:
    reader, successor_generator = _pytyr_pair("blocks", "small")
    root = successor_generator.get_initial_node()
    state = root.get_state()
    actions = [
        successor.label
        for successor in successor_generator.get_labeled_successor_nodes(root)
    ]
    encoder = mifrost.FlatRelationEncoder(
        reader._planning_task, target_sources={"action"}
    )

    stream = encoder.stream()
    assert stream.append(state, actions=actions) == 0
    assert stream.append(state) == 1
    _assert_encoding_equal(
        stream.flush(),
        encoder.encode_batch(
            [state, state], actions=BatchParam.separate([actions, None])
        ),
    )

    mutable = encoder.mutable_stream()
    first = mutable.append(state)
    second = mutable.append(state, actions=actions)
    mutable.update(first, state, actions=actions)
    mutable.remove(second)
    _assert_encoding_equal(
        mutable.flush(), encoder.encode_batch([state], actions=actions)
    )

    mutable = encoder.mutable_stream()
    mutable.set_reuse_removed(True)
    removed = mutable.append(state)
    mutable.remove(removed)
    assert mutable.append(state) == removed


def test_public_flat_rejects_mixed_backend_states() -> None:
    from .test_semantic_parity import _backend_pair

    _, pymimir_problem, reader, successor_generator = _backend_pair()
    pytyr_state = successor_generator.get_initial_node().get_state()
    pymimir_state = pymimir_problem.get_initial_state()
    encoder = mifrost.FlatRelationEncoder(reader._planning_task)

    with pytest.raises(TypeError, match="only PyTyr states"):
        encoder.encode_batch([pytyr_state, pymimir_state])
    with pytest.raises(TypeError, match="expects a lifted or ground PyTyr state"):
        encoder.encode(pymimir_state)


def test_public_flat_backends_coexist_without_global_selection() -> None:
    from .test_flat_semantic_adapter import _assert_semantically_equal
    from .test_semantic_parity import _backend_pair

    _, pymimir_problem, reader, successor_generator = _backend_pair()
    pymimir_state = pymimir_problem.get_initial_state()
    pymimir_actions = list(pymimir_state.generate_applicable_actions())
    pytyr_root = successor_generator.get_initial_node()
    pytyr_state = pytyr_root.get_state()
    pytyr_actions = [
        successor.label
        for successor in successor_generator.get_labeled_successor_nodes(pytyr_root)
    ]
    pymimir_encoder = mifrost.FlatRelationEncoder(
        pymimir_problem.get_domain(), target_sources={"goal", "action"}
    )
    pytyr_encoder = mifrost.FlatRelationEncoder(
        reader._planning_task, target_sources={"goal", "action"}
    )

    assert pymimir_encoder.backend == "pymimir"
    assert pytyr_encoder.backend == "pytyr"
    retained = []
    for _ in range(3):
        pymimir_encoding = pymimir_encoder.encode(
            pymimir_state, actions=pymimir_actions
        )
        pytyr_encoding = pytyr_encoder.encode(pytyr_state, actions=pytyr_actions)
        _assert_semantically_equal(pytyr_encoding, pymimir_encoding)
        retained.extend((pymimir_encoding, pytyr_encoding))
    assert len(retained) == 6


def test_public_flat_explicit_backend_validates_constructor_input() -> None:
    from .test_semantic_parity import _backend_pair

    _, pymimir_problem, reader, _ = _backend_pair()

    with pytest.raises(TypeError, match="PlanningTask"):
        mifrost.FlatRelationEncoder(pymimir_problem.get_domain(), backend="pytyr")
    with pytest.raises(TypeError, match="Unsupported domain type"):
        mifrost.FlatRelationEncoder(reader._planning_task, backend="pymimir")
    with pytest.raises(ValueError, match="backend must be"):
        mifrost.FlatRelationEncoder(reader._planning_task, backend="other")


def test_public_flat_pytyr_import_does_not_require_or_load_pymimir() -> None:
    script = """
import importlib.abc
import sys

class BlockPymimir(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "pymimir" or fullname.startswith("pymimir."):
            raise ModuleNotFoundError("blocked pymimir for single-backend smoke")
        return None

sys.meta_path.insert(0, BlockPymimir())
import mifrost
assert "pymimir" not in sys.modules
encoder_type = mifrost.FlatRelationEncoder
assert encoder_type.__name__ == "FlatRelationEncoder"
assert "pymimir" not in sys.modules
"""
    subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
