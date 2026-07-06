from __future__ import annotations

import networkx as nx
import pytest
import torch
from torch_geometric.data import Batch

import mifrost
from mifrost.encoders import (
    FlatRelationData,
    FlatRelationEncoder,
    _encoding_dict_to_pyg,
)
from mifrost.encoders.flat import _relation_name_color
from mifrost.encoders.flat_data import flat_relation_data_from_pyg

from .test_utils import adv_action, relation_major_from_graph_major


def _assert_flat_batch_equal(
    actual: FlatRelationData, expected: FlatRelationData
) -> None:
    assert type(actual).__name__ == type(expected).__name__
    assert actual.schema == expected.schema
    assert torch.equal(actual.x, expected.x)
    assert torch.equal(actual.node_sizes, expected.node_sizes)
    assert torch.equal(actual.object_sizes, expected.object_sizes)
    assert torch.equal(actual.object_indices, expected.object_indices)
    actual_entity_role_ids = getattr(actual, "entity_role_ids", None)
    expected_entity_role_ids = getattr(expected, "entity_role_ids", None)
    if actual_entity_role_ids is None or expected_entity_role_ids is None:
        assert actual_entity_role_ids is expected_entity_role_ids
    else:
        assert torch.equal(actual_entity_role_ids, expected_entity_role_ids)
    actual_history_entity_sizes = getattr(actual, "history_entity_sizes", None)
    expected_history_entity_sizes = getattr(expected, "history_entity_sizes", None)
    if actual_history_entity_sizes is None or expected_history_entity_sizes is None:
        assert actual_history_entity_sizes is expected_history_entity_sizes
    else:
        assert torch.equal(
            actual_history_entity_sizes,
            expected_history_entity_sizes,
        )
    actual_history_entity_indices = getattr(actual, "history_entity_indices", None)
    expected_history_entity_indices = getattr(expected, "history_entity_indices", None)
    if actual_history_entity_indices is None or expected_history_entity_indices is None:
        assert actual_history_entity_indices is expected_history_entity_indices
    else:
        assert torch.equal(
            actual_history_entity_indices,
            expected_history_entity_indices,
        )
    actual_history_entity_dt = getattr(actual, "history_entity_dt", None)
    expected_history_entity_dt = getattr(expected, "history_entity_dt", None)
    if actual_history_entity_dt is None or expected_history_entity_dt is None:
        assert actual_history_entity_dt is expected_history_entity_dt
    else:
        assert torch.equal(actual_history_entity_dt, expected_history_entity_dt)
    actual_target_entity_sizes = getattr(actual, "target_entity_sizes", None)
    expected_target_entity_sizes = getattr(expected, "target_entity_sizes", None)
    if actual_target_entity_sizes is None or expected_target_entity_sizes is None:
        assert actual_target_entity_sizes is expected_target_entity_sizes
    else:
        assert torch.equal(actual_target_entity_sizes, expected_target_entity_sizes)
    actual_target_entity_indices = getattr(actual, "target_entity_indices", None)
    expected_target_entity_indices = getattr(expected, "target_entity_indices", None)
    if actual_target_entity_indices is None or expected_target_entity_indices is None:
        assert actual_target_entity_indices is expected_target_entity_indices
    else:
        assert torch.equal(
            actual_target_entity_indices,
            expected_target_entity_indices,
        )
    actual_target_entity_group_ids = getattr(actual, "target_entity_group_ids", None)
    expected_target_entity_group_ids = getattr(
        expected, "target_entity_group_ids", None
    )
    if (
        actual_target_entity_group_ids is None
        or expected_target_entity_group_ids is None
    ):
        assert actual_target_entity_group_ids is expected_target_entity_group_ids
    else:
        assert torch.equal(
            actual_target_entity_group_ids,
            expected_target_entity_group_ids,
        )
    actual_target_sizes = getattr(actual, "target_sizes", None)
    expected_target_sizes = getattr(expected, "target_sizes", None)
    if actual_target_sizes is None or expected_target_sizes is None:
        assert actual_target_sizes is expected_target_sizes
    else:
        assert torch.equal(actual_target_sizes, expected_target_sizes)
    actual_relation_instance_sizes = getattr(actual, "relation_instance_sizes", None)
    expected_relation_instance_sizes = getattr(
        expected, "relation_instance_sizes", None
    )
    if (
        actual_relation_instance_sizes is None
        or expected_relation_instance_sizes is None
    ):
        assert actual_relation_instance_sizes is expected_relation_instance_sizes
    else:
        assert torch.equal(
            actual_relation_instance_sizes,
            expected_relation_instance_sizes,
        )
    for field_name in (
        "target_positions",
        "target_indices",
        "target_candidate_ids",
        "target_group_ids",
        "lgan_tn_sizes",
        "lgan_tn_relation_indices",
        "lgan_tn_entity_indices",
        "lgan_nn_sizes",
        "lgan_nn_relation_indices",
        "lgan_nn_entity_indices",
        "lgan_rr_sizes",
        "lgan_rr_src_relation_indices",
        "lgan_rr_dst_relation_indices",
    ):
        actual_field = getattr(actual, field_name, None)
        expected_field = getattr(expected, field_name, None)
        if actual_field is None or expected_field is None:
            assert actual_field is expected_field
        else:
            assert torch.equal(actual_field, expected_field)
    assert torch.equal(actual.relation_counts, expected.relation_counts)
    assert torch.equal(actual.relation_args, expected.relation_args)
    expected_batch = getattr(expected, "batch", None)
    actual_batch = getattr(actual, "batch", None)
    if torch.is_tensor(expected_batch):
        assert torch.equal(actual_batch, expected_batch)
    else:
        assert actual_batch is expected_batch
    expected_ptr = getattr(expected, "ptr", None)
    actual_ptr = getattr(actual, "ptr", None)
    if torch.is_tensor(expected_ptr):
        assert torch.equal(actual_ptr, expected_ptr)
    else:
        assert actual_ptr is expected_ptr
    assert getattr(actual, "node_names", None) == getattr(expected, "node_names", None)
    assert getattr(actual, "object_names", None) == getattr(
        expected, "object_names", None
    )
    assert getattr(actual, "target_names", None) == getattr(
        expected, "target_names", None
    )
    assert getattr(actual, "target_groups", None) == getattr(
        expected, "target_groups", None
    )
    assert getattr(actual, "target_entity_groups", None) == getattr(
        expected, "target_entity_groups", None
    )
    assert getattr(actual, "entity_role_names", None) == getattr(
        expected, "entity_role_names", None
    )
    assert getattr(actual, "target_sources", None) == getattr(
        expected, "target_sources", None
    )
    assert getattr(actual, "lgan_anchor_sources", None) == getattr(
        expected, "lgan_anchor_sources", None
    )
    assert getattr(actual, "include_lgan_edges", None) == getattr(
        expected, "include_lgan_edges", None
    )
    assert getattr(actual, "use_predicate_virtual_nodes", None) == getattr(
        expected, "use_predicate_virtual_nodes", None
    )
    assert getattr(actual, "target_symbol_prefix", None) == getattr(
        expected, "target_symbol_prefix", None
    )
    assert getattr(actual, "entity_node_type", None) == getattr(
        expected, "entity_node_type", None
    )
    assert getattr(actual, "lgan_tn_edge_pos", None) == getattr(
        expected, "lgan_tn_edge_pos", None
    )
    assert getattr(actual, "lgan_nn_edge_pos", None) == getattr(
        expected, "lgan_nn_edge_pos", None
    )
    assert getattr(actual, "lgan_rr_edge_pos", None) == getattr(
        expected, "lgan_rr_edge_pos", None
    )
    assert getattr(actual, "relation_args_layout", None) == getattr(
        expected, "relation_args_layout", None
    )


def _first_action(space, state):
    transitions = list(space.get_forward_transitions(state))
    actions = [action for action, _ in transitions if action is not None]
    if not actions:
        pytest.skip("Fixture does not provide applicable actions.")
    return actions[0]


def _problem_goals(problem):
    goals = list(problem.get_goal_condition().get_literals())
    if not goals:
        pytest.skip("Fixture does not provide goal literals.")
    return goals


def _two_problem_goals(problem):
    goals = _problem_goals(problem)
    if len(goals) < 2:
        pytest.skip("Fixture does not provide enough distinct goal literals.")
    return goals[:2]


def _adv_goal_literal(goal):
    return getattr(goal, "_advanced_ground_literal", goal)


def _history_inputs(problem):
    goals = _problem_goals(problem)
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


def _relation_slot_roles(data: FlatRelationData, relation_name: str) -> tuple[str, ...]:
    return data.schema.slot_roles[data.schema.name_to_id[relation_name]]


def _nonempty_relation_names(
    data: FlatRelationData,
    *,
    source: str | None = None,
    name_contains: str | None = None,
) -> list[str]:
    out: list[str] = []
    for relation_idx, relation_name in enumerate(data.schema.names):
        if source is not None and data.schema.sources[relation_idx] != source:
            continue
        if name_contains is not None and name_contains not in relation_name:
            continue
        if data.flattened_relations[relation_name].shape[0] > 0:
            out.append(relation_name)
    return out


def test_flat_relation_encoder_returns_flat_relation_data(small_blocks):
    _space, domain, problem = small_blocks
    encoder = FlatRelationEncoder(domain)
    data = encoder.encode_pyg(problem.get_initial_state())

    assert isinstance(data, FlatRelationData)
    assert data.schema.names == tuple(encoder.engine.relation_names)
    assert data.schema.arities == tuple(encoder.engine.relation_arities)
    assert data.relation_counts.shape == (1, len(data.schema.names))
    assert data.node_sizes.shape == (1,)
    assert data.object_sizes.shape == (1,)
    assert torch.equal(
        data.object_indices,
        torch.arange(data.object_sizes[0].item(), dtype=torch.long),
    )


def test_flat_relation_predicate_virtual_nodes_default_to_disabled(small_blocks):
    _space, domain, problem = small_blocks
    state = problem.get_initial_state()

    actual = FlatRelationEncoder(domain).encode_pyg(state)
    expected = FlatRelationEncoder(
        domain,
        use_predicate_virtual_nodes=False,
    ).encode_pyg(state)

    assert actual.use_predicate_virtual_nodes is False
    assert expected.use_predicate_virtual_nodes is False
    _assert_flat_batch_equal(actual, expected)


def test_flat_relation_predicate_virtual_nodes_augment_predicate_slots_and_preserve_arguments(
    small_blocks,
):
    _space, domain, problem = small_blocks
    state = problem.get_initial_state()

    base = FlatRelationEncoder(
        domain,
        ignore_zero_arity_relations=False,
    ).encode_pyg(state)
    data = FlatRelationEncoder(
        domain,
        ignore_zero_arity_relations=False,
        use_predicate_virtual_nodes=True,
    ).encode_pyg(state)

    predicate_role_id = list(data.entity_role_names).index("predicate_virtual")
    object_count = int(data.object_sizes[0].item())
    assert data.graph_entity_roles(0)[:object_count] == ["object"] * object_count
    assert data.schema.encoded_arities == tuple(data.schema.arities)
    assert data.schema.logical_arities == tuple(
        sum(role == "argument_slot" for role in roles)
        for roles in data.schema.slot_roles
    )

    checked_relations = 0
    for relation_name in _nonempty_relation_names(data):
        if "predicate_slot" not in _relation_slot_roles(data, relation_name):
            continue
        actual = data.flattened_relations[relation_name]
        expected = base.flattened_relations[relation_name]
        predicate_slot_index = _relation_slot_roles(data, relation_name).index(
            "predicate_slot"
        )
        assert actual.shape[1] == expected.shape[1] + 1
        assert torch.equal(
            torch.cat(
                (
                    actual[:, :predicate_slot_index],
                    actual[:, predicate_slot_index + 1 :],
                ),
                dim=1,
            ),
            expected,
        )
        predicate_rows = actual[:, predicate_slot_index].long()
        if predicate_rows.numel() > 0:
            assert torch.equal(
                data.graph_entity_role_ids(0)[predicate_rows],
                torch.full_like(predicate_rows, predicate_role_id),
            )
        checked_relations += 1

    assert checked_relations > 0


def test_flat_relation_predicate_virtual_nodes_keep_action_slots_distinct(
    small_blocks,
):
    space, domain, problem = small_blocks
    state = problem.get_initial_state()
    action = _first_action(space, state)

    base = FlatRelationEncoder(domain).encode_pyg(state, actions=[action])
    data = FlatRelationEncoder(
        domain,
        use_predicate_virtual_nodes=True,
    ).encode_pyg(state, actions=[action])

    relation_name = _nonempty_relation_names(data, source="action")[0]
    assert _relation_slot_roles(data, relation_name)[0] == "action_slot"
    assert "predicate_slot" not in _relation_slot_roles(data, relation_name)
    assert torch.equal(
        data.flattened_relations[relation_name],
        base.flattened_relations[relation_name],
    )


def test_flat_relation_predicate_virtual_nodes_compose_with_goal_and_history_auxiliary_slots(
    small_blocks,
):
    _space, domain, problem = small_blocks
    state = problem.get_initial_state()
    goal = _problem_goals(problem)[0]
    goals, history_subgoals = _history_inputs(problem)

    goal_base = FlatRelationEncoder(
        domain,
        max_goal_level=1,
        target_sources=[mifrost.TargetSource.goals],
    ).encode_pyg(state, goals=[goal])
    goal_data = FlatRelationEncoder(
        domain,
        max_goal_level=1,
        target_sources=[mifrost.TargetSource.goals],
        use_predicate_virtual_nodes=True,
    ).encode_pyg(state, goals=[goal])

    goal_relation_name = _nonempty_relation_names(goal_data, source="goal")[0]
    assert _relation_slot_roles(goal_data, goal_relation_name)[:2] == (
        "goal_target_slot",
        "predicate_slot",
    )
    assert torch.equal(
        goal_data.flattened_relations[goal_relation_name][:, 0],
        goal_base.flattened_relations[goal_relation_name][:, 0],
    )
    assert torch.equal(
        goal_data.flattened_relations[goal_relation_name][:, 2:],
        goal_base.flattened_relations[goal_relation_name][:, 1:],
    )

    history_base = FlatRelationEncoder(
        domain,
        target_sources=[mifrost.TargetSource.history],
    ).encode_pyg(state, goals=goals, history_subgoals=history_subgoals)
    history_data = FlatRelationEncoder(
        domain,
        target_sources=[mifrost.TargetSource.history],
        use_predicate_virtual_nodes=True,
    ).encode_pyg(state, goals=goals, history_subgoals=history_subgoals)

    history_relation_name = _nonempty_relation_names(history_data, source="history")[0]
    assert _relation_slot_roles(history_data, history_relation_name)[:3] == (
        "history_target_slot",
        "history_slot",
        "predicate_slot",
    )
    assert torch.equal(
        history_data.flattened_relations[history_relation_name][:, :2],
        history_base.flattened_relations[history_relation_name][:, :2],
    )
    assert torch.equal(
        history_data.flattened_relations[history_relation_name][:, 3:],
        history_base.flattened_relations[history_relation_name][:, 2:],
    )


def test_flat_relation_predicate_virtual_nodes_batching_matches_single_graph_slices(
    small_blocks,
):
    space, domain, problem = small_blocks
    state = problem.get_initial_state()
    goal = _problem_goals(problem)[0]
    action = _first_action(space, state)

    encoder = FlatRelationEncoder(
        domain,
        use_predicate_virtual_nodes=True,
        target_sources=[mifrost.TargetSource.goals],
    )
    actual = encoder.encode_batch(
        [state, state],
        goals=[[goal], [goal]],
        actions=[[action], [action]],
    ).as_pyg(as_batch=True)
    expected = flat_relation_data_from_pyg(
        Batch.from_data_list(
            [
                encoder.encode_pyg(state, goals=[goal], actions=[action]),
                encoder.encode_pyg(state, goals=[goal], actions=[action]),
            ]
        )
    )

    _assert_flat_batch_equal(actual, expected)


def test_flat_relation_pyg_output_exposes_encoder_config_attrs(small_blocks):
    space, domain, problem = small_blocks
    state = problem.get_initial_state()
    action = _first_action(space, state)
    goals, history_subgoals = _history_inputs(problem)
    encoder = FlatRelationEncoder(
        domain,
        include_lgan_edges=True,
        lgan_anchor_sources=[mifrost.TargetSource.goals, mifrost.TargetSource.history],
        target_sources=[mifrost.TargetSource.actions, mifrost.TargetSource.goals],
        target_symbol_prefix="tg:",
    )

    data = encoder.encode(
        state, goals=goals, actions=[action], history_subgoals=history_subgoals
    ).as_pyg(as_batch=False)

    assert data.entity_node_type == encoder.entity_node_type
    assert data.include_lgan_edges is encoder.include_lgan_edges
    assert list(data.target_sources) == ["goal", "action"]
    assert list(data.lgan_anchor_sources) == ["goal", "history"]
    assert data.target_symbol_prefix == encoder.target_symbol_prefix
    flattened = data.flattened_relations
    assert set(flattened.keys()) == set(data.schema.names)
    for relation_name, tensor in flattened.items():
        relation_id = data.schema.name_to_id[relation_name]
        assert tensor.shape[1] == data.schema.arities[relation_id]


def test_flat_relation_batch_matches_from_data_list(small_blocks):
    space, domain, problem = small_blocks
    encoder = FlatRelationEncoder(domain)
    states = [
        problem.get_initial_state(),
        space._advanced_state_space_sampler.sample_state_n_steps_from_goal(0),
    ]

    actual = encoder.encode_batch(states).as_pyg(as_batch=True)
    expected = flat_relation_data_from_pyg(
        Batch.from_data_list([encoder.encode_pyg(state) for state in states])
    )

    _assert_flat_batch_equal(actual, expected)


def test_flat_relation_relation_major_packing_is_opt_in(small_blocks):
    space, domain, problem = small_blocks
    states = [
        problem.get_initial_state(),
        space._advanced_state_space_sampler.sample_state_n_steps_from_goal(0),
    ]

    graph_major = FlatRelationEncoder(domain).encode_batch(states)
    relation_major = FlatRelationEncoder(
        domain,
        pack_relation_args_relation_major=True,
    ).encode_batch(states)

    relation_arities = torch.as_tensor(
        graph_major.graph_attrs["relation_arities"],
        dtype=torch.long,
    )
    graph_major_counts = graph_major.get_field("relation_counts")
    graph_major_args = graph_major.get_field("relation_args")
    expected_relation_major = relation_major_from_graph_major(
        graph_major_args,
        graph_major_counts,
        relation_arities,
    )

    assert graph_major.graph_attrs["relation_args_layout"] == "graph_major"
    assert relation_major.graph_attrs["relation_args_layout"] == "relation_major"
    assert torch.equal(relation_major.get_field("relation_counts"), graph_major_counts)
    assert torch.equal(
        relation_major.get_field("relation_args"), expected_relation_major
    )


def test_flat_relation_relation_major_flattened_view_matches_graph_major(
    small_blocks,
):
    space, domain, problem = small_blocks
    states = [
        problem.get_initial_state(),
        space._advanced_state_space_sampler.sample_state_n_steps_from_goal(0),
    ]

    graph_major = FlatRelationEncoder(domain).encode_batch(states).as_pyg(as_batch=True)
    relation_major = (
        FlatRelationEncoder(
            domain,
            pack_relation_args_relation_major=True,
        )
        .encode_batch(states)
        .as_pyg(as_batch=True)
    )

    assert relation_major.relation_args_layout == "relation_major"
    assert not torch.equal(graph_major.relation_args, relation_major.relation_args)
    for graph_index in range(2):
        graph_major_view = graph_major.flattened_relations_view(graph_index=graph_index)
        relation_major_view = relation_major.flattened_relations_view(
            graph_index=graph_index
        )
        for relation_name in graph_major.schema.names:
            assert torch.equal(
                relation_major_view[relation_name],
                graph_major_view[relation_name],
            )

    graph_major_all = graph_major.flattened_relations_view()
    relation_major_all = relation_major.flattened_relations_view()
    for relation_name in graph_major.schema.names:
        assert torch.equal(
            relation_major_all[relation_name],
            graph_major_all[relation_name],
        )

    with pytest.raises(ValueError, match="not representable"):
        relation_major.relation_slot_offsets(graph_index=1)


def test_flat_relation_relation_major_single_graph_keeps_args(small_blocks):
    _, domain, problem = small_blocks
    state = problem.get_initial_state()

    graph_major = FlatRelationEncoder(domain).encode(state)
    relation_major = FlatRelationEncoder(
        domain,
        pack_relation_args_relation_major=True,
    ).encode(state)

    assert graph_major.graph_attrs["relation_args_layout"] == "graph_major"
    assert relation_major.graph_attrs["relation_args_layout"] == "relation_major"
    assert torch.equal(
        relation_major.get_field("relation_counts"),
        graph_major.get_field("relation_counts"),
    )
    assert torch.equal(
        relation_major.get_field("relation_args"),
        graph_major.get_field("relation_args"),
    )


def test_flat_relation_python_conversion_matches_native(small_blocks):
    space, domain, problem = small_blocks
    encoder = FlatRelationEncoder(domain)
    states = [
        problem.get_initial_state(),
        space._advanced_state_space_sampler.sample_state_n_steps_from_goal(0),
    ]

    encoding = encoder.encode_batch(states)
    actual = _encoding_dict_to_pyg(encoding.as_dict(), as_batch=True)
    expected = encoding.as_pyg(as_batch=True)

    _assert_flat_batch_equal(actual, expected)


def test_flat_relation_batch_flattened_relations_group_by_relation(small_blocks):
    space, domain, problem = small_blocks
    encoder = FlatRelationEncoder(domain)
    states = [
        problem.get_initial_state(),
        space._advanced_state_space_sampler.sample_state_n_steps_from_goal(0),
    ]

    batch = encoder.encode_batch(states).as_pyg(as_batch=True)
    combined = batch.flattened_relations_view()
    per_graph = [
        batch.flattened_relations_view(graph_index=graph_index)
        for graph_index in range(batch.num_graphs)
    ]

    for relation_name in batch.schema.names:
        expected = torch.cat(
            [graph_view[relation_name] for graph_view in per_graph],
            dim=0,
        )
        assert torch.equal(combined[relation_name], expected)


def test_flat_relation_encoder_supports_explicit_actions_with_target_entities(
    small_blocks,
):
    space, domain, problem = small_blocks
    encoder = FlatRelationEncoder(domain)
    state = problem.get_initial_state()
    action = _first_action(space, state)
    adv = adv_action(action)
    formatter = mifrost.RelationFormatter

    data = encoder.encode_pyg(state, actions=[action])

    assert data.target_entity_sizes.tolist() == [1]
    assert data.target_entity_indices.tolist() == [int(data.object_sizes[0].item())]
    target_entity_index = int(data.target_entity_indices[0].item())
    assert data.graph_target_entity_names(0) == [formatter.format_action(adv)]

    flattened = data.flattened_relations
    action_schema = formatter.format_action_schema(adv.get_action())
    assert action_schema in flattened
    action_relation = flattened[action_schema]
    assert action_relation.shape == (1, len(adv.get_objects()) + 1)
    assert int(action_relation[0, 0].item()) == target_entity_index

    action_arg_names = [
        data.graph_node_names(0)[int(idx)] for idx in action_relation[0, 1:].tolist()
    ]
    assert action_arg_names == [obj.get_name() for obj in adv.get_objects()]


def test_flat_relation_batch_matches_from_data_list_with_actions(small_blocks):
    space, domain, problem = small_blocks
    encoder = FlatRelationEncoder(domain)
    state = problem.get_initial_state()
    action = _first_action(space, state)

    actual = encoder.encode_batch([state, state], actions=[[action], None]).as_pyg(
        as_batch=True
    )
    expected = flat_relation_data_from_pyg(
        Batch.from_data_list(
            [
                encoder.encode_pyg(state, actions=[action]),
                encoder.encode_pyg(state),
            ]
        )
    )

    _assert_flat_batch_equal(actual, expected)


def test_flat_relation_batch_accepts_generator_actions(small_blocks):
    space, domain, problem = small_blocks
    encoder = FlatRelationEncoder(domain)
    state = problem.get_initial_state()
    action = _first_action(space, state)

    generated = encoder.encode_batch(
        [state],
        actions=(item for item in [action]),
    ).as_pyg(as_batch=False)
    listed = encoder.encode_batch([state], actions=[action]).as_pyg(as_batch=False)

    _assert_flat_batch_equal(generated, listed)


def test_flat_relation_batch_accepts_per_state_none_action_entries(small_blocks):
    space, domain, problem = small_blocks
    encoder = FlatRelationEncoder(domain)
    state = problem.get_initial_state()
    action = _first_action(space, state)

    actual = encoder.encode_batch([state, state], actions=[None, [action]]).as_pyg(
        as_batch=True
    )
    expected = encoder.encode_batch([state, state], actions=[[], [action]]).as_pyg(
        as_batch=True
    )

    _assert_flat_batch_equal(actual, expected)


def test_flat_relation_batch_matches_from_data_list_with_history(small_blocks):
    _space, domain, problem = small_blocks
    encoder = FlatRelationEncoder(domain)
    state = problem.get_initial_state()
    goals, history_subgoals = _history_inputs(problem)

    actual = encoder.encode_batch(
        [state, state],
        goals=goals,
        history_subgoals=[history_subgoals, [(-1, [goals[0]])]],
    ).as_pyg(as_batch=True)
    expected = flat_relation_data_from_pyg(
        Batch.from_data_list(
            [
                encoder.encode_pyg(
                    state,
                    goals=goals,
                    history_subgoals=history_subgoals,
                ),
                encoder.encode_pyg(
                    state,
                    goals=goals,
                    history_subgoals=[(-1, [goals[0]])],
                ),
            ]
        )
    )

    _assert_flat_batch_equal(actual, expected)


def test_flat_relation_action_target_metadata_enabled_for_action_source(
    small_blocks,
):
    space, domain, problem = small_blocks
    state = problem.get_initial_state()
    action = _first_action(space, state)
    adv = adv_action(action)
    formatter = mifrost.RelationFormatter

    encoder = FlatRelationEncoder(
        domain,
        target_sources=[mifrost.TargetSource.actions],
    )
    encoding = encoder.encode_batch([state], actions=[[action]])
    data = encoding.as_pyg(as_batch=True)

    assert encoding.has_field("target_sizes")
    assert encoding.has_field("target_positions")
    assert encoding.has_field("target_indices")
    assert encoding.has_field("target_candidate_ids")
    assert encoding.has_field("target_group_ids")
    assert data.target_sizes.tolist() == [1]
    assert data.target_positions.tolist() == data.target_entity_indices.tolist()
    assert data.target_indices.tolist() == [0]
    assert data.target_candidate_ids.tolist() == [0]
    assert data.target_group_ids.tolist() == [0]
    assert data.target_entity_group_ids.tolist() == [0]
    assert (
        data.graph_target_positions(0).tolist() == data.target_entity_indices.tolist()
    )
    assert data.graph_target_names(0) == [formatter.format_action(adv)]
    assert list(data.target_groups) == ["action"]
    assert list(data.target_entity_groups) == ["action"]
    assert data.target_symbol_prefix == "target:"


def test_flat_relation_native_target_names_materialize_on_access(small_blocks):
    space, domain, problem = small_blocks
    state = problem.get_initial_state()
    action = _first_action(space, state)

    encoder = FlatRelationEncoder(
        domain,
        target_sources=[mifrost.TargetSource.actions],
    )
    encoding = encoder.encode_batch([state], actions=[[action]])

    target_names = list(encoding.target_names)
    assert len(target_names) == 1
    assert encoding.graph_attrs["target_names"] == target_names
    assert encoding.as_dict()["graph_attrs"]["target_names"] == target_names


def test_flat_relation_action_target_metadata_can_be_disabled(small_blocks):
    space, domain, problem = small_blocks
    state = problem.get_initial_state()
    action = _first_action(space, state)

    encoder = FlatRelationEncoder(domain)
    encoding = encoder.encode_batch([state], actions=[[action]])
    data = encoding.as_pyg(as_batch=True)

    assert not encoding.has_field("target_sizes")
    assert not encoding.has_field("target_positions")
    assert not encoding.has_field("target_indices")
    assert not hasattr(data, "target_positions")
    assert not hasattr(data, "target_indices")
    assert not hasattr(data, "target_names")


def test_flat_relation_action_target_metadata_preserves_duplicates_and_empty_graphs(
    small_blocks,
):
    space, domain, problem = small_blocks
    state = problem.get_initial_state()
    action = _first_action(space, state)
    action_name = mifrost.RelationFormatter.format_action(adv_action(action))

    encoder = FlatRelationEncoder(
        domain,
        target_sources=[mifrost.TargetSource.actions],
    )
    actual = encoder.encode_batch(
        [state, state],
        actions=[[action, action], []],
    ).as_pyg(as_batch=True)
    expected = flat_relation_data_from_pyg(
        Batch.from_data_list(
            [
                encoder.encode_pyg(state, actions=[action, action]),
                encoder.encode_pyg(state, actions=[]),
            ]
        )
    )

    assert actual.target_sizes.tolist() == [2, 0]
    assert actual.target_indices.tolist() == [0, 1]
    assert actual.target_candidate_ids.tolist() == [0, 1]
    assert actual.target_positions.tolist()[0] == actual.target_positions.tolist()[1]
    assert actual.graph_target_names(0) == [action_name, action_name]
    _assert_flat_batch_equal(actual, expected)


def test_flat_relation_goal_target_source_adjusts_root_goal_arities_and_metadata(
    small_blocks,
):
    _space, domain, problem = small_blocks
    state = problem.get_initial_state()
    goal = _problem_goals(problem)[0]
    formatter = mifrost.RelationFormatter

    base_encoder = FlatRelationEncoder(domain, max_goal_level=1)
    goal_encoder = FlatRelationEncoder(
        domain,
        max_goal_level=1,
        target_sources=[mifrost.TargetSource.goals],
    )

    assert goal_encoder.engine.relation_names == base_encoder.engine.relation_names

    base_data = base_encoder.encode_pyg(state, goals=[goal])
    goal_data = goal_encoder.encode_pyg(state, goals=[goal])
    root_goal_name = formatter.format_literal(_adv_goal_literal(goal), 0)

    nonempty_goal_relations = [
        relation_name
        for relation_idx, relation_name in enumerate(goal_data.schema.names)
        if goal_data.schema.sources[relation_idx] == "goal"
        and base_data.flattened_relations[relation_name].shape[0] > 0
    ]
    assert nonempty_goal_relations
    for relation_name in nonempty_goal_relations:
        assert (
            goal_data.flattened_relations[relation_name].shape[1]
            == base_data.flattened_relations[relation_name].shape[1] + 1
        )
        assert torch.equal(
            goal_data.flattened_relations[relation_name][:, 0],
            goal_data.target_positions,
        )

    assert goal_data.target_entity_sizes.tolist() == [1]
    assert goal_data.target_sizes.tolist() == [1]
    assert goal_data.graph_target_names(0) == [root_goal_name]
    assert goal_data.graph_target_entity_names(0, group="goal") == [root_goal_name]
    assert (
        goal_data.graph_target_positions(0).tolist()
        == goal_data.target_positions.tolist()
    )
    assert goal_data.graph_target_entity_indices(0, group="goal").tolist() == (
        goal_data.target_entity_indices.tolist()
    )
    assert goal_data.target_group_ids.tolist() == [0]
    assert goal_data.target_entity_group_ids.tolist() == [0]
    assert list(goal_data.target_groups) == ["goal"]
    assert list(goal_data.target_entity_groups) == ["goal", "action"]


def test_flat_relation_subgoal_target_source_adjusts_only_layered_goal_arities(
    small_blocks,
):
    _space, domain, problem = small_blocks
    state = problem.get_initial_state()
    goal = _problem_goals(problem)[0]
    formatter = mifrost.RelationFormatter

    base_encoder = FlatRelationEncoder(domain, max_goal_level=1)
    subgoal_encoder = FlatRelationEncoder(
        domain,
        max_goal_level=1,
        target_sources=[mifrost.TargetSource.subgoals],
    )

    assert subgoal_encoder.engine.relation_names == base_encoder.engine.relation_names

    base_root = base_encoder.encode_pyg(state, goals=[goal])
    subgoal_root = subgoal_encoder.encode_pyg(state, goals=[goal])
    base_subgoal = base_encoder.encode_pyg(state, goals=[], subgoal_layers=[[goal]])
    subgoal_data = subgoal_encoder.encode_pyg(state, goals=[], subgoal_layers=[[goal]])
    subgoal_name = formatter.format_literal(_adv_goal_literal(goal), 1)

    root_goal_relations = [
        relation_name
        for relation_idx, relation_name in enumerate(subgoal_root.schema.names)
        if subgoal_root.schema.sources[relation_idx] == "goal"
        and base_root.flattened_relations[relation_name].shape[0] > 0
    ]
    assert root_goal_relations
    for relation_name in root_goal_relations:
        assert (
            subgoal_root.flattened_relations[relation_name].shape[1]
            == base_root.flattened_relations[relation_name].shape[1]
        )

    subgoal_relations = [
        relation_name
        for relation_idx, relation_name in enumerate(subgoal_data.schema.names)
        if subgoal_data.schema.sources[relation_idx] == "goal"
        and base_subgoal.flattened_relations[relation_name].shape[0] > 0
    ]
    assert subgoal_relations
    for relation_name in subgoal_relations:
        assert (
            subgoal_data.flattened_relations[relation_name].shape[1]
            == base_subgoal.flattened_relations[relation_name].shape[1] + 1
        )
        assert torch.equal(
            subgoal_data.flattened_relations[relation_name][:, 0],
            subgoal_data.target_positions,
        )

    assert subgoal_data.target_entity_sizes.tolist() == [1]
    assert subgoal_data.target_sizes.tolist() == [1]
    assert subgoal_data.graph_target_names(0) == [subgoal_name]
    assert subgoal_data.graph_target_entity_names(0, group="subgoal") == [subgoal_name]
    assert subgoal_data.target_group_ids.tolist() == [0]
    assert subgoal_data.target_entity_group_ids.tolist() == [0]
    assert list(subgoal_data.target_groups) == ["subgoal"]
    assert list(subgoal_data.target_entity_groups) == ["subgoal", "action"]


def test_flat_relation_subgoal_target_visualization_preserves_argument_edges(
    small_blocks,
):
    _space, domain, problem = small_blocks
    state = problem.get_initial_state()
    goal = next(
        (
            candidate
            for candidate in _problem_goals(problem)
            if candidate.get_atom().get_predicate().get_arity() > 0
        ),
        None,
    )
    if goal is None:
        pytest.skip("Fixture does not provide any positive-arity goal literals.")
    encoder = FlatRelationEncoder(
        domain,
        max_goal_level=1,
        target_sources=[mifrost.TargetSource.subgoals],
    )

    data = encoder.encode_pyg(state, goals=[], subgoal_layers=[[goal]])
    graph = encoder.to_networkx(data)
    target_name = mifrost.RelationFormatter.format_literal(_adv_goal_literal(goal), 1)
    incoming_target_edges = [
        (src, attrs)
        for src, _dst, attrs in graph.in_edges(target_name, data=True)
        if attrs.get("position") == "0"
    ]
    assert len(incoming_target_edges) == 1
    relation_node = incoming_target_edges[0][0]

    atom = goal.get_atom()
    objects = (
        list(atom.get_terms())
        if hasattr(atom, "get_terms")
        else list(atom.get_objects())
    )

    assert {
        (dst, attrs["position"])
        for _src, dst, attrs in graph.out_edges(relation_node, data=True)
    } == {
        (target_name, "0"),
        *{
            (obj.get_name(), str(arg_idx))
            for arg_idx, obj in enumerate(objects, start=1)
        },
    }


def test_flat_relation_debug_colors_depend_only_on_relation_name():
    assert _relation_name_color("[+]on[sg]") == _relation_name_color("[+]on[sg]")
    assert _relation_name_color("[+]on[sg]") != _relation_name_color("clear")


def test_flat_relation_mixed_target_sources_preserve_order_and_grouping(small_blocks):
    space, domain, problem = small_blocks
    state = problem.get_initial_state()
    goal0, goal1 = _two_problem_goals(problem)
    action = _first_action(space, state)
    formatter = mifrost.RelationFormatter

    base_encoder = FlatRelationEncoder(domain, max_goal_level=1)
    encoder = FlatRelationEncoder(
        domain,
        max_goal_level=1,
        target_sources=[
            mifrost.TargetSource.goals,
            mifrost.TargetSource.subgoals,
            mifrost.TargetSource.actions,
        ],
    )
    data = encoder.encode_pyg(
        state,
        goals=[goal0],
        subgoal_layers=[[goal1]],
        actions=[action],
    )

    assert encoder.engine.relation_names == base_encoder.engine.relation_names
    assert list(data.target_groups) == ["goal", "subgoal", "action"]
    assert list(data.target_entity_groups) == ["goal", "subgoal", "action"]
    assert data.target_indices.tolist() == [0, 1, 2]
    assert data.target_candidate_ids.tolist() == [0, 1, 2]
    assert data.target_group_ids.tolist() == [0, 1, 2]
    assert data.target_entity_group_ids.tolist() == [0, 1, 2]
    assert data.graph_target_names(0) == [
        formatter.format_literal(_adv_goal_literal(goal0), 0),
        formatter.format_literal(_adv_goal_literal(goal1), 1),
        formatter.format_action(adv_action(action)),
    ]
    assert data.graph_target_entity_names(0, group="goal") == [
        formatter.format_literal(_adv_goal_literal(goal0), 0)
    ]
    assert data.graph_target_entity_names(0, group="subgoal") == [
        formatter.format_literal(_adv_goal_literal(goal1), 1)
    ]
    assert data.graph_target_entity_names(0, group="action") == [
        formatter.format_action(adv_action(action))
    ]
    assert data.graph_target_entity_group_ids(0).tolist() == [0, 1, 2]
    assert data.graph_target_positions(0).tolist() == data.target_positions.tolist()


def test_flat_relation_goal_targets_preserve_duplicate_candidates_and_empty_graphs(
    small_blocks,
):
    _space, domain, problem = small_blocks
    state = problem.get_initial_state()
    goal = _problem_goals(problem)[0]
    goal_name = mifrost.RelationFormatter.format_literal(_adv_goal_literal(goal), 0)

    encoder = FlatRelationEncoder(
        domain,
        target_sources=[mifrost.TargetSource.goals],
    )
    actual = encoder.encode_batch(
        [state, state],
        goals=[[goal, goal], []],
    ).as_pyg(as_batch=True)
    expected = flat_relation_data_from_pyg(
        Batch.from_data_list(
            [
                encoder.encode_pyg(state, goals=[goal, goal]),
                encoder.encode_pyg(state, goals=[]),
            ]
        )
    )

    assert actual.target_entity_sizes.tolist() == [1, 0]
    assert actual.target_sizes.tolist() == [2, 0]
    assert actual.target_indices.tolist() == [0, 1]
    assert actual.target_candidate_ids.tolist() == [0, 1]
    assert actual.target_group_ids.tolist() == [0, 0]
    assert actual.target_positions.tolist()[0] == actual.target_positions.tolist()[1]
    assert actual.graph_target_names(0) == [goal_name, goal_name]
    assert actual.graph_target_entity_names(0, group="goal") == [goal_name]
    _assert_flat_batch_equal(actual, expected)


def test_flat_relation_history_entities_and_relations(small_blocks):
    _space, domain, problem = small_blocks
    encoder = FlatRelationEncoder(domain)
    goals, history_subgoals = _history_inputs(problem)

    data = encoder.encode_pyg(
        problem.get_initial_state(),
        goals=goals,
        history_subgoals=history_subgoals,
    )

    filtered_history = _filtered_history_inputs(history_subgoals)
    expected_history_literals = sum(len(literals) for _dt, literals in filtered_history)
    assert data.history_entity_sizes.tolist() == [len(filtered_history)]
    assert data.graph_history_entity_dt(0).tolist() == [
        dt for dt, _literals in filtered_history
    ]
    assert all(
        name.startswith("history:") for name in data.graph_history_entity_names(0)
    )
    emitted_history_literals = sum(
        int(data.flattened_relations[relation_name].shape[0])
        for relation_idx, relation_name in enumerate(data.schema.names)
        if data.schema.sources[relation_idx] == "history"
    )
    assert emitted_history_literals == expected_history_literals


def test_flat_relation_history_max_steps_filters(small_blocks):
    _space, domain, problem = small_blocks
    encoder = FlatRelationEncoder(domain)
    goals, history_subgoals = _history_inputs(problem)

    data = encoder.encode_pyg(
        problem.get_initial_state(),
        goals=goals,
        history_subgoals=history_subgoals,
        history_max_steps=1,
    )

    assert data.history_entity_sizes.tolist() == [1]
    assert data.graph_history_entity_dt(0).tolist() == [-1]


def test_flat_relation_history_target_metadata(small_blocks):
    _space, domain, problem = small_blocks
    encoder = FlatRelationEncoder(
        domain,
        target_sources=[mifrost.TargetSource.history],
    )
    goals, history_subgoals = _history_inputs(problem)
    filtered_history = _filtered_history_inputs(history_subgoals)

    data = encoder.encode_pyg(
        problem.get_initial_state(),
        goals=goals,
        history_subgoals=history_subgoals,
    )

    expected_target_count = sum(len(literals) for _dt, literals in filtered_history)
    assert data.target_sizes.tolist() == [expected_target_count]
    assert list(data.target_groups) == ["history"]
    assert list(data.target_entity_groups) == ["action", "history"]
    assert data.target_indices.tolist() == list(range(expected_target_count))
    assert data.target_candidate_ids.tolist() == list(range(expected_target_count))
    assert data.target_group_ids.tolist() == [0] * expected_target_count
    assert data.graph_target_entity_names(
        0, group="history"
    ) == data.graph_target_names(0)
    assert data.graph_target_positions(0).tolist() == (
        data.graph_target_entity_indices(0, group="history").tolist()
    )
    assert data.graph_target_entity_group_ids(0).tolist() == [1] * expected_target_count
    assert all(name.startswith("history:") for name in data.graph_target_names(0))


def test_flat_relation_history_target_metadata_disambiguates_same_literal_across_timesteps(
    small_blocks,
):
    _space, domain, problem = small_blocks
    goals = _problem_goals(problem)
    goal = goals[0]
    encoder = FlatRelationEncoder(
        domain,
        target_sources=[mifrost.TargetSource.history],
    )
    data = encoder.encode_pyg(
        problem.get_initial_state(),
        goals=goals,
        history_subgoals=[(-1, [goal]), (-2, [goal])],
    )

    assert list(data.target_groups) == ["history"]
    assert data.target_indices.tolist() == [0, 1]
    assert data.target_candidate_ids.tolist() == [0, 1]
    assert data.target_group_ids.tolist() == [0, 0]
    assert data.target_positions.tolist()[0] != data.target_positions.tolist()[1]
    assert data.graph_target_names(0)[0] != data.graph_target_names(0)[1]


def test_flat_relation_history_visualization_marks_history_entities(small_blocks):
    _space, domain, problem = small_blocks
    encoder = FlatRelationEncoder(
        domain,
        target_sources=[mifrost.TargetSource.history],
    )
    goals, history_subgoals = _history_inputs(problem)
    data = encoder.encode_pyg(
        problem.get_initial_state(),
        goals=goals,
        history_subgoals=history_subgoals,
    )

    graph = encoder.to_networkx(data)
    history_entities = [
        node
        for node, attrs in graph.nodes(data=True)
        if attrs.get("entity_kind") == "history_entity"
    ]
    history_targets = [
        node
        for node, attrs in graph.nodes(data=True)
        if attrs.get("target_group") == "history"
    ]

    assert history_entities
    assert history_targets
    assert all(
        graph.nodes[node].get("history_dt") is not None for node in history_entities
    )


def test_flat_relation_encoder_rejects_reserved_state_target_source(
    small_blocks,
):
    _space, domain, _problem = small_blocks

    with pytest.raises(
        ValueError,
        match="reserved for the upcoming flat successor/horizon encoders",
    ):
        FlatRelationEncoder(domain, target_sources=[mifrost.TargetSource.states])


def test_flat_relation_lgan_actions_use_action_target_entities_without_target_metadata(
    small_blocks,
):
    space, domain, problem = small_blocks
    state = problem.get_initial_state()
    action = _first_action(space, state)
    encoder = FlatRelationEncoder(domain, include_lgan_edges=True)

    data = encoder.encode_pyg(state, actions=[action])

    assert data.relation_instance_sizes.tolist() == [
        int(data.relation_counts.sum().item())
    ]
    assert getattr(data, "target_sizes", None) is None
    assert data.graph_target_entity_names(0, group="action")
    tn_edges = data.graph_lgan_tn_edges(0)
    nn_edges = data.graph_lgan_nn_edges(0)
    rr_edges = data.graph_lgan_rr_edges(0)
    assert tn_edges.shape[1] > 0
    assert nn_edges.shape[1] > 0
    assert rr_edges.shape[1] > 0
    action_entity_indices = set(
        data.graph_target_entity_indices(0, group="action").tolist()
    )
    assert set(tn_edges[1].tolist()).issubset(action_entity_indices)


def test_flat_relation_lgan_rejects_missing_anchor_rows(small_blocks):
    _space, domain, problem = small_blocks
    encoder = FlatRelationEncoder(domain, include_lgan_edges=True)

    with pytest.raises(ValueError, match="requires LGAN anchor entity rows"):
        encoder.encode(problem.get_initial_state())


def test_flat_relation_lgan_goal_anchor_sources_work_without_target_metadata(
    small_blocks,
):
    _space, domain, problem = small_blocks
    goals = _problem_goals(problem)
    encoder = FlatRelationEncoder(
        domain,
        include_lgan_edges=True,
        lgan_anchor_sources=[mifrost.TargetSource.goals],
    )

    data = encoder.encode_pyg(problem.get_initial_state(), goals=goals)

    assert getattr(data, "target_sizes", None) is None
    goal_target_entities = set(
        data.graph_target_entity_indices(0, group="goal").tolist()
    )
    assert goal_target_entities
    tn_edges = data.graph_lgan_tn_edges(0)
    assert tn_edges.shape[1] > 0
    assert set(tn_edges[1].tolist()).issubset(goal_target_entities)


def test_flat_relation_lgan_goal_target_entities_are_used_as_anchors(small_blocks):
    _space, domain, problem = small_blocks
    goals = _problem_goals(problem)
    encoder = FlatRelationEncoder(
        domain,
        include_lgan_edges=True,
        lgan_anchor_sources=[mifrost.TargetSource.goals],
        target_sources=[mifrost.TargetSource.goals],
    )

    data = encoder.encode_pyg(problem.get_initial_state(), goals=goals)

    goal_target_entities = set(
        data.graph_target_entity_indices(0, group="goal").tolist()
    )
    assert goal_target_entities
    tn_edges = data.graph_lgan_tn_edges(0)
    assert tn_edges.shape[1] > 0
    assert set(tn_edges[1].tolist()).issubset(goal_target_entities)


def test_flat_relation_lgan_history_anchor_sources_work_without_target_metadata(
    small_blocks,
):
    _space, domain, problem = small_blocks
    _goals, history_subgoals = _history_inputs(problem)
    encoder = FlatRelationEncoder(
        domain,
        include_lgan_edges=True,
        lgan_anchor_sources=[mifrost.TargetSource.history],
    )

    data = encoder.encode_pyg(
        problem.get_initial_state(),
        history_subgoals=history_subgoals,
    )

    assert getattr(data, "target_sizes", None) is None
    history_target_entities = set(
        data.graph_target_entity_indices(0, group="history").tolist()
    )
    assert history_target_entities
    tn_edges = data.graph_lgan_tn_edges(0)
    assert tn_edges.shape[1] > 0
    assert set(tn_edges[1].tolist()).issubset(history_target_entities)


def test_flat_relation_lgan_anchor_sources_are_ignored_when_lgan_is_disabled(
    small_blocks,
):
    _space, domain, problem = small_blocks
    goals = _problem_goals(problem)
    encoder = FlatRelationEncoder(
        domain,
        include_lgan_edges=False,
        lgan_anchor_sources=[mifrost.TargetSource.goals],
    )

    data = encoder.encode_pyg(problem.get_initial_state(), goals=goals)

    assert data.target_entity_sizes.tolist() == [0]
    assert getattr(data, "lgan_tn_sizes", None) is None


def test_flat_relation_lgan_does_not_change_relation_schema_or_payload(
    small_blocks,
):
    space, domain, problem = small_blocks
    state = problem.get_initial_state()
    action = _first_action(space, state)
    base_encoder = FlatRelationEncoder(domain)
    lgan_encoder = FlatRelationEncoder(domain, include_lgan_edges=True)

    base = base_encoder.encode_pyg(state, actions=[action])
    lgan = lgan_encoder.encode_pyg(state, actions=[action])

    assert base.schema.names == lgan.schema.names
    assert base.schema.arities == lgan.schema.arities
    assert base.schema.sources == lgan.schema.sources
    assert torch.equal(base.relation_counts, lgan.relation_counts)
    assert torch.equal(base.relation_args, lgan.relation_args)


def test_flat_relation_lgan_batch_matches_from_data_list(small_blocks):
    space, domain, problem = small_blocks
    state = problem.get_initial_state()
    action = _first_action(space, state)
    encoder = FlatRelationEncoder(domain, include_lgan_edges=True)

    actual = encoder.encode_batch([state, state], actions=[[action], [action]]).as_pyg(
        as_batch=True
    )
    expected = flat_relation_data_from_pyg(
        Batch.from_data_list(
            [
                encoder.encode_pyg(state, actions=[action]),
                encoder.encode_pyg(state, actions=[action]),
            ]
        )
    )

    assert actual.graph_relation_instance_range(0) == (
        0,
        int(actual.relation_instance_sizes[0].item()),
    )
    assert actual.graph_relation_instance_range(1) == (
        int(actual.relation_instance_sizes[0].item()),
        int(actual.relation_instance_sizes.sum().item()),
    )
    _assert_flat_batch_equal(actual, expected)


def test_flat_relation_lgan_visualization_overlays_packed_edges(small_blocks):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")

    space, domain, problem = small_blocks
    state = problem.get_initial_state()
    action = _first_action(space, state)
    encoder = FlatRelationEncoder(domain, include_lgan_edges=True)
    data = encoder.encode_pyg(state, actions=[action])
    graph = encoder.to_networkx(data)

    lgan_edge_count = (
        data.graph_lgan_tn_edges(0).shape[1]
        + data.graph_lgan_nn_edges(0).shape[1]
        + data.graph_lgan_rr_edges(0).shape[1]
    )
    lgan_edges = [
        (src, dst, attrs.get("lgan_kind"))
        for src, dst, attrs in graph.edges(data=True)
        if attrs.get("lgan_kind") is not None
    ]

    assert lgan_edges
    assert len(lgan_edges) == lgan_edge_count
    assert (
        graph.number_of_edges()
        == sum(
            tensor.shape[0] * tensor.shape[1]
            for tensor in data.flattened_relations_view(graph_index=0).values()
        )
        + lgan_edge_count
    )

    ax = encoder.draw(data)
    assert ax is not None


def test_flat_relation_visualization_is_reconstructable(small_blocks):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")

    _space, domain, problem = small_blocks
    encoder = FlatRelationEncoder(domain)
    data = encoder.encode_pyg(problem.get_initial_state())
    graph = encoder.to_networkx(data)

    assert isinstance(graph, nx.MultiDiGraph)
    relation_nodes = [
        node
        for node, attrs in graph.nodes(data=True)
        if attrs.get("kind") == "relation"
    ]
    assert len(relation_nodes) == int(data.relation_counts.sum().item())

    expected_edges = sum(
        tensor.shape[0] * tensor.shape[1]
        for tensor in data.flattened_relations_view(graph_index=0).values()
    )
    assert graph.number_of_edges() == expected_edges

    ax = encoder.draw(data)
    assert ax is not None
