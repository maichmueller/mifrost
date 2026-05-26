from __future__ import annotations

import pytest

from mifrost.encoders import FlatHorizonEncoder, FlatRootedHorizonEncoder

from .test_flat_horizon_encoder import (
    _assert_flat_batch_equal,
    _first_distinct_changed_transitions,
    _single_step_dag,
)


def test_flat_rooted_horizon_encoder_defaults_match_rooted_horizon_config(
    small_blocks,
):
    _space, domain, _problem = small_blocks
    encoder = FlatRootedHorizonEncoder(domain)

    assert encoder.config.root_policy.name == "encode_only"
    assert encoder.config.enable_parent_relation is True
    assert encoder.config.ignore_actions is False


def test_flat_rooted_horizon_encoder_matches_generic_rooted_horizon(
    small_blocks,
):
    space, domain, problem = small_blocks
    root = problem.get_initial_state()
    transitions = _first_distinct_changed_transitions(space, root, count=2)
    if len(transitions) < 2:
        pytest.skip("Fixture should yield at least 2 distinct changed transitions")
    dag = _single_step_dag(root, transitions[:2], candidate_ids=[101, 202])

    actual = FlatRootedHorizonEncoder(domain).encode_pyg(
        root,
        dag=dag,
        goals=list(problem.get_goal_condition().get_literals()),
    )
    expected = FlatHorizonEncoder(
        domain,
        root_policy="encode_only",
        enable_parent_relation=True,
        ignore_actions=False,
    ).encode_pyg(
        root,
        dag=dag,
        goals=list(problem.get_goal_condition().get_literals()),
    )

    _assert_flat_batch_equal(actual, expected)
