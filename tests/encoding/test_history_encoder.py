from __future__ import annotations

import pytest

import mifrost
from mifrost.encoders import HGraphEncoder
from mifrost.encoders.types import to_advanced_literal
from tests.conftest import problem_setup
from tests.encoding.test_utils import keywise_equal


DOMAIN_CASES = [
    ("blocks", "probBLOCKS-4-0"),
    ("gripper", "gripper_b-5"),
    ("spanner", "medium"),
    ("delivery", "instance_2x2_p-2_0"),
]


def _history_inputs(problem):
    goals = list(problem.get_goal_condition().get_literals())
    if not goals:
        pytest.skip("Problem has no goal literals for history encoding.")
    if len(goals) == 1:
        return goals, [(-1, goals)]
    if len(goals) == 2:
        return goals, [(-1, goals[:1]), (-2, goals[1:2])]
    return goals, [(-1, goals[:2]), (-2, goals[2:3])]


@pytest.mark.parametrize(
    "domain_name,problem_name",
    DOMAIN_CASES,
    ids=[f"{d}:{p}" for d, p in DOMAIN_CASES],
)
def test_history_nodes_and_edges(domain_name, problem_name):
    _, domain, problem = problem_setup(domain_name, problem_name)
    encoder = HGraphEncoder(domain)
    state = problem.get_initial_state()
    goals, history_subgoals = _history_inputs(problem)

    data = encoder.encode(
        state,
        goals=goals,
        history_subgoals=history_subgoals,
    )

    assert "history" in data.node_types
    history_node_names = list(getattr(data["history"], "node_names", []))
    assert history_node_names, "Expected history node names to be populated."

    history_dt = data["history"]["history_dt"]
    if history_dt.ndim == 2:
        history_dt = history_dt.squeeze(-1)
    history_dt_values = history_dt.tolist()
    expected_dt_values = sorted([dt for dt, _ in history_subgoals])
    assert history_dt_values == expected_dt_values

    history_rel = mifrost.DEFAULT_HISTORY_LINK_RELATION
    formatter = mifrost.RelationFormatter

    for history_idx, (dt, literals) in enumerate(
        sorted(history_subgoals, key=lambda x: x[0])
    ):
        assert history_dt_values[history_idx] == dt
        for literal in literals:
            adv_literal = to_advanced_literal(literal)
            predicate = adv_literal.get_atom().get_predicate()
            node_type = formatter.format_predicate(
                predicate,
                None,
                None,
                adv_literal.get_polarity(),
            )
            node_key = formatter.format_literal(adv_literal, None)

            node_names = list(getattr(data[node_type], "node_names", []))
            assert node_key in node_names, f"Missing literal node {node_key}"

            literal_idx = node_names.index(node_key)

            forward_type = (node_type, history_rel, "history")
            reverse_type = ("history", history_rel, node_type)
            assert forward_type in data.edge_types
            assert reverse_type in data.edge_types

            forward_edges = data[forward_type].edge_index
            reverse_edges = data[reverse_type].edge_index
            assert any(
                src == literal_idx and dst == history_idx
                for src, dst in zip(
                    forward_edges[0].tolist(), forward_edges[1].tolist()
                )
            ), f"Missing history edge {node_key} -> history:{dt}"
            assert any(
                src == history_idx and dst == literal_idx
                for src, dst in zip(
                    reverse_edges[0].tolist(), reverse_edges[1].tolist()
                )
            ), f"Missing history edge history:{dt} -> {node_key}"


@pytest.mark.parametrize(
    "domain_name,problem_name",
    DOMAIN_CASES,
    ids=[f"{d}:{p}" for d, p in DOMAIN_CASES],
)
def test_history_max_steps_filters(domain_name, problem_name):
    _, domain, problem = problem_setup(domain_name, problem_name)
    encoder = HGraphEncoder(domain)
    state = problem.get_initial_state()
    goals, history_subgoals = _history_inputs(problem)

    data = encoder.encode(
        state,
        goals=goals,
        history_subgoals=history_subgoals,
        history_max_steps=1,
    )

    history_dt = data["history"]["history_dt"]
    if history_dt.ndim == 2:
        history_dt = history_dt.squeeze(-1)
    history_dt_values = history_dt.tolist()
    assert history_dt_values == [-1], (
        "history_max_steps=1 should retain only dt=-1 entries."
    )


@pytest.mark.parametrize(
    "domain_name,problem_name",
    DOMAIN_CASES,
    ids=[f"{d}:{p}" for d, p in DOMAIN_CASES],
)
def test_history_batch_and_stream_parity(domain_name, problem_name):
    space, domain, problem = problem_setup(domain_name, problem_name)
    encoder = HGraphEncoder(domain)
    initial = problem.get_initial_state()
    states = [initial]

    transitions = list(space.get_forward_transitions(initial))
    if transitions:
        _action, next_state = transitions[0]
        states.append(next_state)
    else:
        pytest.skip("No forward transitions available to build batch history test.")

    goals, history_subgoals = _history_inputs(problem)
    history_per_state = [history_subgoals for _ in states]

    batch_parts = encoder.encode_batch_parts(
        states,
        goals=goals,
        history_subgoals=history_per_state,
    )

    stream = encoder.stream()
    for state in states:
        stream.append(state, goals=goals, history_subgoals=history_subgoals)
    stream_parts = stream.flush_batch_encoding_py()

    keywise_equal(batch_parts, stream_parts)


def test_history_custom_relation_override():
    _, domain, problem = problem_setup("blocks", "probBLOCKS-4-0")
    encoder = HGraphEncoder(domain, history_link_relation="_custom_history_")
    state = problem.get_initial_state()
    goals, history_subgoals = _history_inputs(problem)

    data = encoder.encode(
        state,
        goals=goals,
        history_subgoals=history_subgoals,
    )

    history_edges = [
        edge_type for edge_type in data.edge_types if "history" in edge_type
    ]
    assert history_edges, "Expected history edges in encoded graph."
    assert all(edge_type[1] == "_custom_history_" for edge_type in history_edges)
