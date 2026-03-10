from __future__ import annotations

import pytest

import mifrost
from mifrost.encoders import HGraphEncoder
from mifrost.encoders.types import to_advanced_literal
from tests.conftest import problem_setup
from tests.encoding.test_utils import keywise_equal


DOMAIN_CASES = [
    ("blocks", "smedium"),
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


def _filtered_history_inputs(history_subgoals, history_max_steps: int | None = None):
    out = []
    for dt, literals in history_subgoals:
        if history_max_steps is not None and abs(int(dt)) > history_max_steps:
            continue
        out.append((int(dt), list(literals)))
    return sorted(out, key=lambda item: item[0])


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

    data = encoder.encode_pyg(
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

    data = encoder.encode_pyg(
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

    batch_parts = encoder._encode_batch(
        states,
        goals=goals,
        history_subgoals=history_per_state,
    )

    stream = encoder.stream()
    for state in states:
        stream.append(state, goals=goals, history_subgoals=history_subgoals)
    stream_parts = stream.flush()

    keywise_equal(batch_parts, stream_parts)


@pytest.mark.parametrize(
    "domain_name,problem_name",
    DOMAIN_CASES,
    ids=[f"{d}:{p}" for d, p in DOMAIN_CASES],
)
def test_history_target_metadata(domain_name, problem_name):
    _, domain, problem = problem_setup(domain_name, problem_name)
    encoder = HGraphEncoder(
        domain,
        target_sources=[mifrost.TargetSource.History],
    )
    state = problem.get_initial_state()
    goals, history_subgoals = _history_inputs(problem)
    filtered_history = _filtered_history_inputs(history_subgoals)

    encoding = encoder.encode(state, goals=goals, history_subgoals=history_subgoals)
    data = encoding.as_pyg(as_batch=True)

    expected_target_count = sum(len(literals) for _dt, literals in filtered_history)
    assert expected_target_count > 0
    assert encoding.has_field("target_positions")
    assert encoding.has_field("target_indices")
    assert encoding.has_field("target_candidate_ids")
    assert encoding.has_field("target_group_ids")
    assert list(data.target_groups) == ["history"]
    assert len(data.target_positions.tolist()) == expected_target_count
    assert data.target_indices.tolist() == list(range(expected_target_count))
    assert data.target_candidate_ids.tolist() == list(range(expected_target_count))
    assert data.target_group_ids.tolist() == [0] * expected_target_count
    assert len(list(data.target_names)) == expected_target_count
    assert all(str(name).startswith("history:") for name in data.target_names)

    expected_literals = []
    formatter = mifrost.RelationFormatter
    for dt, literals in filtered_history:
        for literal in literals:
            formatted = formatter.format_literal(to_advanced_literal(literal), None)
            expected_literals.append((dt, formatted))
    for target_name, (dt, literal_name) in zip(
        list(data.target_names), expected_literals, strict=True
    ):
        assert f"history:{dt}" in str(target_name)
        assert literal_name in str(target_name)


@pytest.mark.parametrize(
    "domain_name,problem_name",
    DOMAIN_CASES,
    ids=[f"{d}:{p}" for d, p in DOMAIN_CASES],
)
def test_history_lgan_anchor_sources_work_without_target_metadata(
    domain_name, problem_name
):
    _, domain, problem = problem_setup(domain_name, problem_name)
    encoder = HGraphEncoder(
        domain,
        include_lgan_edges=True,
        lgan_anchor_sources=[mifrost.TargetSource.History],
    )
    state = problem.get_initial_state()
    goals, history_subgoals = _history_inputs(problem)

    data = encoder.encode_pyg(
        state,
        goals=goals,
        history_subgoals=history_subgoals,
    )

    assert not hasattr(data, "target_positions")
    symbol_names = list(getattr(data[mifrost.DEFAULT_SYMBOL_TYPE_ID], "node_names", []))
    assert any(name.startswith("target:history|") for name in symbol_names)
    tn_edge_types = [
        edge_type
        for edge_type in data.edge_types
        if edge_type[1] == encoder.lgan_tn_edge_pos
    ]
    assert tn_edge_types
    assert (
        sum(int(data[edge_type].edge_index.size(1)) for edge_type in tn_edge_types) > 0
    )


def test_history_target_metadata_disambiguates_same_literal_across_timesteps():
    _, domain, problem = problem_setup("blocks", "smedium")
    goals = list(problem.get_goal_condition().get_literals())
    if not goals:
        pytest.skip("Problem has no goal literals for history encoding.")

    goal = goals[0]
    encoder = HGraphEncoder(domain, target_sources=[mifrost.TargetSource.History])
    data = encoder.encode_pyg(
        problem.get_initial_state(),
        goals=goals,
        history_subgoals=[(-1, [goal]), (-2, [goal])],
    )

    assert list(data.target_groups) == ["history"]
    assert data.target_indices.tolist() == [0, 1]
    assert data.target_candidate_ids.tolist() == [0, 1]
    assert data.target_group_ids.tolist() == [0, 0]
    assert len(list(data.target_names)) == 2
    assert data.target_positions.tolist()[0] != data.target_positions.tolist()[1]
    assert list(data.target_names)[0] != list(data.target_names)[1]


def test_history_custom_relation_override():
    _, domain, problem = problem_setup("blocks", "smedium")
    encoder = HGraphEncoder(domain, history_link_relation="_custom_history_")
    state = problem.get_initial_state()
    goals, history_subgoals = _history_inputs(problem)

    data = encoder.encode_pyg(
        state,
        goals=goals,
        history_subgoals=history_subgoals,
    )

    history_edges = [
        edge_type for edge_type in data.edge_types if "history" in edge_type
    ]
    assert history_edges, "Expected history edges in encoded graph."
    assert all(edge_type[1] == "_custom_history_" for edge_type in history_edges)
