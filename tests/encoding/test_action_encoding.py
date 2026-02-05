from __future__ import annotations

import mifrost
from mifrost.encoders import HGraphEncoder

from .test_utils import adv_action


def test_action_encoding_includes_all_applicable_actions(small_blocks):
    space, domain, problem = small_blocks
    state = problem.get_initial_state()
    transitions = list(space.get_forward_transitions(state))
    actions = [act for act, _ in transitions if act is not None]
    if not actions:
        import pytest

        pytest.skip("Fixture does not provide applicable actions.")

    encoder = HGraphEncoder(domain, ignore_actions=False)
    data = encoder.encode(state, actions=actions)

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
