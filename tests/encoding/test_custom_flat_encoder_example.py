from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

EXAMPLES_ENCODERS_DIR = Path(__file__).resolve().parents[2] / "examples" / "encoders"
if str(EXAMPLES_ENCODERS_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_ENCODERS_DIR))

from _custom_flat_target_mask_encoder import ExampleTargetMaskFlatEncoder


def _first_action(space, state):
    transitions = list(space.get_forward_transitions(state))
    actions = [action for action, _ in transitions if action is not None]
    if not actions:
        pytest.skip("Fixture does not provide applicable actions.")
    return actions[0]


def test_example_target_mask_flat_encoder_single_graph(small_blocks) -> None:
    space, domain, problem = small_blocks
    encoder = ExampleTargetMaskFlatEncoder(domain)
    state = problem.get_initial_state()
    action = _first_action(space, state)

    data = encoder.encode_pyg(state, actions=[action])

    assert encoder.relation_names == tuple(encoder.engine.relation_names)
    assert encoder.relation_arities == tuple(encoder.engine.relation_arities)
    assert encoder.relation_sources == tuple(encoder.engine.relation_sources)
    assert data.target_entity_count.tolist() == [1]
    mask = data.entity_is_target
    target_indices = data.target_entity_indices.long()
    expected = torch.zeros((data.num_nodes,), dtype=torch.float32)
    expected[target_indices] = 1.0
    assert torch.equal(mask, expected)
    assert torch.isclose(mask.sum(), torch.tensor(1.0, dtype=torch.float32))
    assert data.graph_target_entity_names(0) == [
        data.graph_node_names(0)[int(target_indices[0].item())]
    ]


def test_example_target_mask_flat_encoder_batch_graphs(small_blocks) -> None:
    space, domain, problem = small_blocks
    encoder = ExampleTargetMaskFlatEncoder(domain)
    state = problem.get_initial_state()
    action = _first_action(space, state)

    batch = encoder.encode_batch_pyg([state, state], actions=[[action], None])

    assert batch.target_entity_count.tolist() == [1, 0]
    first_start, first_end = batch.graph_node_range(0)
    second_start, second_end = batch.graph_node_range(1)
    first_mask = batch.entity_is_target[first_start:first_end]
    second_mask = batch.entity_is_target[second_start:second_end]
    assert torch.isclose(first_mask.sum(), torch.tensor(1.0, dtype=torch.float32))
    assert torch.isclose(second_mask.sum(), torch.tensor(0.0, dtype=torch.float32))
