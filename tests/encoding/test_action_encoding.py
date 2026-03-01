from __future__ import annotations

import pytest

import mifrost
from mifrost.encoders import HGraphEncoder

from .test_utils import adv_action


def _first_actions(space, state, count: int = 2):
    transitions = list(space.get_forward_transitions(state))
    actions = [act for act, _ in transitions if act is not None]
    if len(actions) < count:
        import pytest

        pytest.skip("Fixture does not provide enough applicable actions.")
    return actions[:count]


def _has_action_node(data, action) -> bool:
    formatter = mifrost.RelationFormatter
    adv = adv_action(action)
    action_node = formatter.format_action(adv)
    action_type = formatter.format_action_schema(adv.get_action())
    node_names = list(getattr(data[action_type], "node_names", []))
    return action_node in node_names


def test_action_encoding_includes_all_applicable_actions(small_blocks):
    space, domain, problem = small_blocks
    state = problem.get_initial_state()
    transitions = list(space.get_forward_transitions(state))
    actions = [act for act, _ in transitions if act is not None]
    if not actions:
        import pytest

        pytest.skip("Fixture does not provide applicable actions.")

    encoder = HGraphEncoder(domain, ignore_actions=False)
    data = encoder.encode_pyg(state, actions=actions)

    symbol_type = mifrost.DEFAULT_SYMBOL_TYPE_ID
    symbol_names = list(getattr(data[symbol_type], "node_names", []))

    formatter = mifrost.RelationFormatter
    seen = set()
    for action in actions:
        adv = adv_action(action)
        action_node = formatter.format_action(adv)
        if action_node in seen:
            continue
        seen.add(action_node)
        action_type = formatter.format_action_schema(adv.get_action())
        node_names = list(getattr(data[action_type], "node_names", []))
        assert action_node in node_names, (
            f"Missing action node {action_node} of type {action_type}."
        )

        action_symbol = f"target:{adv.get_index()}|{action_node}"
        assert action_symbol in symbol_names, (
            f"Missing action symbol node {action_symbol} for action {action_node}."
        )

        forward_edges = data.get_edge_store(symbol_type, "0", action_type).edge_index
        src_indices, dst_indices = forward_edges
        src_idx = symbol_names.index(action_symbol)
        dst_idx = node_names.index(action_node)
        assert any(
            s == src_idx and d == dst_idx
            for s, d in zip(src_indices.tolist(), dst_indices.tolist())
        ), f"Missing edge {action_symbol} -> {action_node} at position 0."

        for pos, obj in enumerate(adv.get_objects(), start=1):
            obj_name = formatter.format_object(obj)
            assert obj_name in symbol_names, f"Missing object node {obj_name}."
            edges = data.get_edge_store(symbol_type, str(pos), action_type).edge_index
            src_indices, dst_indices = edges
            src_idx = symbol_names.index(obj_name)
            assert any(
                s == src_idx and d == dst_idx
                for s, d in zip(src_indices.tolist(), dst_indices.tolist())
            ), f"Missing edge {obj_name} -> {action_node} at position {pos}."


def test_hgraph_action_target_metadata_mode_is_opt_in(small_blocks):
    space, domain, problem = small_blocks
    state = problem.get_initial_state()
    action0, _action1 = _first_actions(space, state, count=2)

    encoder = HGraphEncoder(domain, ignore_actions=False)
    encoding = encoder.encode_batch([state], actions=[[action0]])
    data = encoding.as_pyg(as_batch=True)

    assert not encoding.has_field("target_positions")
    assert not encoding.has_field("target_indices")
    assert not hasattr(data, "target_positions")
    assert not hasattr(data, "target_indices")
    assert not hasattr(data, "target_names")


def test_hgraph_action_target_metadata_uses_action_input_positions(small_blocks):
    space, domain, problem = small_blocks
    state = problem.get_initial_state()
    action0, action1 = _first_actions(space, state, count=2)

    encoder = HGraphEncoder(domain, ignore_actions=False, export_action_targets=True)
    encoding = encoder.encode_batch([state], actions=[[action0, action1]])
    data = encoding.as_pyg(as_batch=True)

    assert encoding.has_field("target_positions")
    assert encoding.has_field("target_indices")
    assert encoding.has_field("target_candidate_ids")
    positions = data.target_positions.tolist()

    assert data.target_indices.tolist() == [0, 1]
    assert data.target_candidate_ids.tolist() == [0, 1]
    assert data.target_indices_ptr.tolist() == [0, 2]
    assert data.target_candidate_ids_ptr.tolist() == [0, 2]
    assert len(positions) == 2
    assert positions[0] != positions[1]
    assert positions[0] >= 0
    assert positions[1] >= 0
    assert data.target_positions_ptr.tolist() == [0, 2]
    assert hasattr(data, "target_names")
    assert hasattr(data, "target_symbol_prefix")
    assert len(list(data.target_names)) == 2
    assert data.target_symbol_prefix == "target:"


def test_hgraph_action_target_metadata_preserves_duplicates_and_empty_graphs(
    small_blocks,
):
    space, domain, problem = small_blocks
    state = problem.get_initial_state()
    action0, _action1 = _first_actions(space, state, count=2)

    encoder = HGraphEncoder(domain, ignore_actions=False, export_action_targets=True)
    encoding = encoder.encode_batch([state, state], actions=[[action0, action0], []])
    data = encoding.as_pyg(as_batch=True)

    assert encoding.has_field("target_positions")
    assert encoding.has_field("target_indices")
    assert encoding.has_field("target_candidate_ids")
    positions = data.target_positions.tolist()
    assert data.target_indices.tolist() == [0, 1]
    assert data.target_candidate_ids.tolist() == [0, 1]
    assert data.target_indices_ptr.tolist() == [0, 2, 2]
    assert data.target_candidate_ids_ptr.tolist() == [0, 2, 2]
    assert len(positions) == 2
    assert positions[0] == positions[1]
    assert data.target_positions_ptr.tolist() == [0, 2, 2]
    assert isinstance(list(data.target_names), list)
    assert data.target_symbol_prefix == "target:"


def test_hgraph_encode_rejects_tuple_nested_actions_with_horizon_guidance(small_blocks):
    space, domain, problem = small_blocks
    state = problem.get_initial_state()
    action0, action1 = _first_actions(space, state, count=2)

    encoder = HGraphEncoder(domain, ignore_actions=False)
    with pytest.raises(ValueError, match="use HorizonEncoder for IW lookahead"):
        encoder.encode(state, actions=[(action0, action1)])


def test_hgraph_encode_batch_rejects_nested_per_state_actions_with_guidance(
    small_blocks,
):
    space, domain, problem = small_blocks
    state = problem.get_initial_state()
    action0, action1 = _first_actions(space, state, count=2)

    encoder = HGraphEncoder(domain, ignore_actions=False)
    with pytest.raises(ValueError, match="use HorizonEncoder for IW lookahead"):
        encoder.encode_batch(
            [state, state],
            actions=[[(action0,)], [(action1,)]],
        )


def test_hgraph_encode_batch_accepts_flat_per_state_actions(small_blocks):
    space, domain, problem = small_blocks
    state = problem.get_initial_state()
    action0, action1 = _first_actions(space, state, count=2)

    encoder = HGraphEncoder(domain, ignore_actions=False)
    encoding = encoder.encode_batch([state, state], actions=[[action0], [action1]])

    assert encoding.num_graphs == 2


def test_hgraph_encode_batch_accepts_per_state_none_action_entries(small_blocks):
    space, domain, problem = small_blocks
    state = problem.get_initial_state()
    action0, _action1 = _first_actions(space, state, count=2)
    formatter = mifrost.RelationFormatter
    action_type = formatter.format_action_schema(adv_action(action0).get_action())

    encoder = HGraphEncoder(domain, ignore_actions=False)
    expected = encoder.encode_batch([state, state], actions=[[], [action0]])
    actual = encoder.encode_batch([state, state], actions=[None, [action0]])

    expected_pyg = expected.as_pyg(as_batch=True)
    actual_pyg = actual.as_pyg(as_batch=True)
    assert expected_pyg[action_type].num_nodes == actual_pyg[action_type].num_nodes
    assert list(expected_pyg[action_type].node_names) == list(
        actual_pyg[action_type].node_names
    )


def test_hgraph_encode_batch_preserves_per_state_generator_actions(small_blocks):
    space, domain, problem = small_blocks
    state = problem.get_initial_state()
    action0, _action1 = _first_actions(space, state, count=2)

    encoder = HGraphEncoder(domain, ignore_actions=False)
    generated = encoder.encode_batch(
        [state], actions=[(item for item in [action0])]
    ).as_pyg(as_batch=False)
    listed = encoder.encode_batch([state], actions=[[action0]]).as_pyg(as_batch=False)

    assert _has_action_node(generated, action0)
    assert _has_action_node(listed, action0)


def test_hgraph_encode_batch_accepts_shared_generator_actions(small_blocks):
    space, domain, problem = small_blocks
    state = problem.get_initial_state()
    action0, _action1 = _first_actions(space, state, count=2)

    encoder = HGraphEncoder(domain, ignore_actions=False)
    generated = encoder.encode_batch(
        [state], actions=(item for item in [action0])
    ).as_pyg(as_batch=False)
    listed = encoder.encode_batch([state], actions=[action0]).as_pyg(as_batch=False)

    assert _has_action_node(generated, action0)
    assert _has_action_node(listed, action0)
