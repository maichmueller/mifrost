from __future__ import annotations

import mifrost

from .test_utils import (
    adv_action,
    adv_domain,
    adv_state,
    goal_inputs_from_problem,
    parts_to_pyg,
)


def test_horizon_encoder_target_mapping_and_order(small_blocks):
    space, domain, problem = small_blocks
    root = problem.get_initial_state()

    transitions = list(space.get_forward_transitions(root))[:3]
    assert transitions, "Fixture should yield at least 1 transition"

    dag = mifrost.TransitionDAG(adv_state(root))
    for action, target in transitions:
        dag.register_transition(
            adv_state(root),
            adv_state(target),
            adv_action(action),
        )

    config = mifrost.HorizonEncoderConfig()
    config.transition_mode = mifrost.HorizonEncoderMode.Full
    encoder = mifrost.HorizonHGraphEncoderEngine(adv_domain(domain), config)
    goals = goal_inputs_from_problem(problem)
    parts = encoder.encode(adv_state(root), dag, goals)
    data = parts_to_pyg(parts)

    symbol_type = config.symbol_type_id
    symbol_node_names = list(getattr(data[symbol_type], "node_names", []))
    prefix = config.target_symbol_prefix
    target_positions = [
        idx
        for idx, name in enumerate(symbol_node_names)
        if str(name).startswith(prefix)
    ]
    assert target_positions == list(range(len(target_positions))), (
        f"Target positions should be contiguous from 0..n-1, got {target_positions}"
    )

    # Verify per-transition action mapping (edge from target symbol -> action node at pos 0)
    formatter = mifrost.RelationFormatter
    for k, (action, target) in enumerate(transitions, start=1):
        action = adv_action(action)
        action_type = formatter.format_action_schema(action.get_action())
        target_name = symbol_node_names[target_positions[k]]
        action_node_name = f"{target_name}|{formatter.format_action(action)}"

        node_names = list(getattr(data[action_type], "node_names", []))
        assert action_node_name in node_names, (
            f"Expected action node '{action_node_name}' in node names for type '{action_type}'."
        )
        action_idx = node_names.index(action_node_name)

        edge_type = (symbol_type, "0", action_type)
        edge_index = data[edge_type].edge_index
        src_indices = edge_index[0].tolist()
        dst_indices = edge_index[1].tolist()
        assert (target_positions[k], action_idx) in zip(src_indices, dst_indices), (
            f"No (target:{k} -> {action_node_name}) edge at position 0 for transition {k}."
        )
