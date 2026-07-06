from __future__ import annotations

import networkx as nx
import pymimir as mimir
import pytest
import torch
from torch_geometric.data import Batch

import mifrost
from mifrost.encoders import FlatHorizonEncoder
from mifrost.encoders.flat_data import flat_relation_data_from_pyg

from .test_flat_relation_encoder import _assert_flat_batch_equal
from .test_utils import adv_action, adv_state, relation_major_from_graph_major


def _first_distinct_changed_transitions(space, root, count: int = 2):
    out = []
    seen_targets: set[str] = set()
    root_repr = str(adv_state(root))
    for action, target in space.get_forward_transitions(root):
        if action is None or target is None:
            continue
        target_repr = str(adv_state(target))
        if target_repr == root_repr or target_repr in seen_targets:
            continue
        seen_targets.add(target_repr)
        out.append((action, target))
        if len(out) >= count:
            return out
    if not out:
        pytest.skip("Fixture should yield at least 1 distinct changed transition")
    return out


def _single_step_dag(root, transitions, *, candidate_ids: list[int] | None = None):
    dag = mifrost.TransitionDAG(adv_state(root))
    for idx, (action, target) in enumerate(transitions):
        kwargs = {}
        if candidate_ids is not None:
            kwargs["candidate_id"] = candidate_ids[idx]
        dag.register_transition(
            adv_state(root),
            adv_state(target),
            adv_action(action),
            **kwargs,
        )
    return dag


def _delta_literals(root, target, problem):
    root_atoms = set(root.get_atoms(ignore_static=True))
    target_atoms = set(target.get_atoms(ignore_static=True))
    out = [
        mimir.GroundLiteral.new(atom, True, problem)
        for atom in (target_atoms - root_atoms)
    ]
    out.extend(
        mimir.GroundLiteral.new(atom, False, problem)
        for atom in (root_atoms - target_atoms)
    )
    return out


def _positive_goals_for_removed_facts(root, target, problem):
    goals = [
        mimir.GroundLiteral.new(literal.get_atom(), True, problem)
        for literal in _delta_literals(root, target, problem)
        if not literal.get_polarity()
    ]
    if not goals:
        pytest.skip("Fixture should yield at least one removed fluent literal")
    return goals


def test_flat_horizon_encoder_emits_state_target_entities_and_metadata(small_blocks):
    space, domain, problem = small_blocks
    root = problem.get_initial_state()
    transitions = _first_distinct_changed_transitions(space, root, count=1)
    dag = _single_step_dag(root, transitions, candidate_ids=[101])

    data = FlatHorizonEncoder(domain, ignore_actions=False).encode_pyg(
        root,
        dag=dag,
        goals=list(problem.get_goal_condition().get_literals()),
    )

    assert data.target_entity_groups == ["state"]
    assert data.target_groups == ["state"]
    assert data.target_entity_sizes.tolist() == [1]
    assert data.target_entity_group_ids.tolist() == [0]
    assert data.graph_target_entity_names(0) == ["target:1"]
    assert data.target_sizes.tolist() == [1]
    assert data.graph_target_indices(0).tolist() == [1]
    assert data.graph_target_candidate_ids(0).tolist() == [101]
    assert data.graph_target_depths(0).tolist() == [1]
    assert data.graph_target_positions(0).tolist() == [
        data.graph_target_entity_indices(0)[0].item()
    ]
    assert len(data.graph_target_names(0)) == 1


def test_flat_horizon_predicate_virtual_nodes_default_to_disabled(small_blocks):
    space, domain, problem = small_blocks
    root = problem.get_initial_state()
    transitions = _first_distinct_changed_transitions(space, root, count=1)
    dag = _single_step_dag(root, transitions, candidate_ids=[101])

    actual = FlatHorizonEncoder(domain, ignore_actions=False).encode_pyg(
        root,
        dag=dag,
        goals=list(problem.get_goal_condition().get_literals()),
    )
    expected = FlatHorizonEncoder(
        domain,
        ignore_actions=False,
        use_predicate_virtual_nodes=False,
    ).encode_pyg(
        root,
        dag=dag,
        goals=list(problem.get_goal_condition().get_literals()),
    )

    assert actual.use_predicate_virtual_nodes is False
    assert expected.use_predicate_virtual_nodes is False
    _assert_flat_batch_equal(actual, expected)


def test_flat_horizon_predicate_virtual_nodes_follow_state_slot_and_preserve_arguments(
    small_blocks,
):
    space, domain, problem = small_blocks
    root = problem.get_initial_state()
    transitions = _first_distinct_changed_transitions(space, root, count=1)
    dag = _single_step_dag(root, transitions, candidate_ids=[101])

    base = FlatHorizonEncoder(
        domain,
        ignore_actions=False,
    ).encode_pyg(
        root,
        dag=dag,
        goals=list(problem.get_goal_condition().get_literals()),
    )
    data = FlatHorizonEncoder(
        domain,
        ignore_actions=False,
        use_predicate_virtual_nodes=True,
    ).encode_pyg(
        root,
        dag=dag,
        goals=list(problem.get_goal_condition().get_literals()),
    )

    relation_name = next(
        name
        for idx, name in enumerate(data.schema.names)
        if data.schema.sources[idx] == "state"
        and "[state]" in name
        and data.flattened_relations[name].shape[0] > 0
    )
    assert data.schema.slot_roles[data.schema.name_to_id[relation_name]][:2] == (
        "state_slot",
        "predicate_slot",
    )
    assert torch.equal(
        data.flattened_relations[relation_name][:, 0],
        base.flattened_relations[relation_name][:, 0],
    )
    assert torch.equal(
        data.flattened_relations[relation_name][:, 2:],
        base.flattened_relations[relation_name][:, 1:],
    )
    predicate_role_id = list(data.entity_role_names).index("predicate_virtual")
    predicate_rows = data.flattened_relations[relation_name][:, 1].long()
    assert torch.equal(
        data.graph_entity_role_ids(0)[predicate_rows],
        torch.full_like(predicate_rows, predicate_role_id),
    )


def test_flat_horizon_explicit_candidate_ids_are_preserved(small_blocks):
    space, domain, problem = small_blocks
    root = problem.get_initial_state()
    transitions = _first_distinct_changed_transitions(space, root, count=2)
    if len(transitions) < 2:
        pytest.skip("Fixture should yield at least 2 distinct changed transitions")
    dag = _single_step_dag(root, transitions[:2], candidate_ids=[101, 202])

    data = FlatHorizonEncoder(domain, ignore_actions=False).encode_pyg(
        root,
        dag=dag,
        goals=list(problem.get_goal_condition().get_literals()),
    )
    index_to_candidate = dict(
        zip(
            data.graph_target_indices(0).tolist(),
            data.graph_target_candidate_ids(0).tolist(),
            strict=True,
        )
    )

    assert index_to_candidate == {1: 101, 2: 202}


def test_flat_horizon_export_node_names_false_skips_target_names(small_blocks):
    space, domain, problem = small_blocks
    root = problem.get_initial_state()
    transitions = _first_distinct_changed_transitions(space, root, count=1)
    dag = _single_step_dag(root, transitions, candidate_ids=[101])

    data = FlatHorizonEncoder(
        domain,
        ignore_actions=False,
        export_node_names=False,
    ).encode_pyg(
        root,
        dag=dag,
        goals=list(problem.get_goal_condition().get_literals()),
    )

    assert not hasattr(data, "target_names")
    assert data.graph_target_names(0) == []


def test_flat_horizon_native_target_names_materialize_on_access(small_blocks):
    space, domain, problem = small_blocks
    root = problem.get_initial_state()
    transitions = _first_distinct_changed_transitions(space, root, count=1)
    dag = _single_step_dag(root, transitions, candidate_ids=[101])

    encoding = FlatHorizonEncoder(domain, ignore_actions=False).encode(
        root,
        dag=dag,
        goals=list(problem.get_goal_condition().get_literals()),
    )

    target_names = list(encoding.target_names)
    assert len(target_names) == 1
    assert encoding.graph_attrs["target_names"] == target_names
    assert encoding.as_dict()["graph_attrs"]["target_names"] == target_names


def test_flat_horizon_goal_derivations_can_exclude_plain_goal_literals(small_blocks):
    space, domain, problem = small_blocks
    root = problem.get_initial_state()
    transitions = _first_distinct_changed_transitions(space, root, count=1)
    dag = _single_step_dag(root, transitions, candidate_ids=[101])

    data = FlatHorizonEncoder(
        domain,
        ignore_actions=False,
        goal_derivations={
            mifrost.GoalDerivation.satisfied,
            mifrost.GoalDerivation.unsatisfied,
        },
    ).encode_pyg(
        root,
        dag=dag,
        goals=list(problem.get_goal_condition().get_literals()),
    )

    schema_names = set(data.schema.names)
    assert any("[g][sat]" in name for name in schema_names)
    assert any("[g][unsat]" in name for name in schema_names)
    assert not any(
        "[g]" in name and "[sat" not in name and "[unsat]" not in name
        for name in schema_names
    )


def test_flat_horizon_delta_registers_state_literal_relations_without_plain_goals(
    small_blocks,
):
    space, domain, problem = small_blocks
    root = problem.get_initial_state()
    transitions = _first_distinct_changed_transitions(space, root, count=1)
    action, target = transitions[0]

    dag = mifrost.TransitionDAG(adv_state(root))
    dag.register_transition(
        adv_state(root),
        adv_state(target),
        adv_action(action),
        delta_literals=_delta_literals(root, target, problem),
    )

    data = FlatHorizonEncoder(
        domain,
        transition_mode="delta",
        root_policy="exclude",
        ignore_actions=True,
        goal_derivations={
            mifrost.GoalDerivation.satisfied,
            mifrost.GoalDerivation.unsatisfied,
            mifrost.GoalDerivation.added_satisfied,
            mifrost.GoalDerivation.added_unsatisfied,
        },
    ).encode_pyg(
        root,
        dag=dag,
        goals=list(problem.get_goal_condition().get_literals()),
    )

    assert any(name.startswith("[+]") for name in data.schema.names)
    assert not any("[state]" in name for name in data.schema.names)


def test_flat_horizon_delta_emits_removed_goal_satisfaction_literals(small_blocks):
    _space, domain, problem = small_blocks
    root = problem.get_initial_state()
    (action, target) = _first_distinct_changed_transitions(_space, root, count=1)[0]

    dag = mifrost.TransitionDAG(adv_state(root))
    dag.register_transition(
        adv_state(root),
        adv_state(target),
        adv_action(action),
        delta_literals=_delta_literals(root, target, problem),
    )

    removed_goals = _positive_goals_for_removed_facts(root, target, problem)
    data = FlatHorizonEncoder(
        domain,
        transition_mode="delta",
        root_policy="exclude",
        ignore_actions=True,
        goal_derivations={mifrost.GoalDerivation.added_unsatisfied},
    ).encode_pyg(
        root,
        dag=dag,
        goals=removed_goals,
    )

    emitted = {
        name: tensor
        for name, tensor in data.flattened_relations.items()
        if tensor.shape[0] > 0
    }
    assert any("[sat-]" in name for name in emitted)
    assert not any("[sat+]" in name for name in emitted)


def test_flat_horizon_rejects_partial_explicit_candidate_ids(small_blocks):
    space, domain, problem = small_blocks
    root = problem.get_initial_state()
    transitions = _first_distinct_changed_transitions(space, root, count=2)
    if len(transitions) < 2:
        pytest.skip("Fixture should yield at least 2 distinct changed transitions")

    dag = mifrost.TransitionDAG(adv_state(root))
    dag.register_transition(
        adv_state(root),
        adv_state(transitions[0][1]),
        adv_action(transitions[0][0]),
        candidate_id=7,
    )
    dag.register_transition(
        adv_state(root),
        adv_state(transitions[1][1]),
        adv_action(transitions[1][0]),
    )

    encoder = FlatHorizonEncoder(domain, ignore_actions=False)
    with pytest.raises(ValueError, match="missing candidate_id for target node index"):
        encoder.encode(
            root,
            dag=dag,
            goals=list(problem.get_goal_condition().get_literals()),
        )


def test_flat_horizon_rejects_duplicate_explicit_candidate_ids(small_blocks):
    space, domain, problem = small_blocks
    root = problem.get_initial_state()
    transitions = _first_distinct_changed_transitions(space, root, count=2)
    if len(transitions) < 2:
        pytest.skip("Fixture should yield at least 2 distinct changed transitions")

    dag = _single_step_dag(root, transitions[:2], candidate_ids=[9, 9])
    encoder = FlatHorizonEncoder(domain, ignore_actions=False)
    with pytest.raises(ValueError, match="duplicate candidate_id"):
        encoder.encode(
            root,
            dag=dag,
            goals=list(problem.get_goal_condition().get_literals()),
        )


def test_flat_horizon_excluded_root_uses_private_carrier_row(small_blocks):
    space, domain, problem = small_blocks
    root = problem.get_initial_state()
    transitions = _first_distinct_changed_transitions(space, root, count=1)
    action, _target = transitions[0]
    adv_ground_action = adv_action(action)
    formatter = mifrost.RelationFormatter

    data = FlatHorizonEncoder(domain, ignore_actions=False).encode_pyg(
        root,
        dag=_single_step_dag(root, transitions),
        goals=list(problem.get_goal_condition().get_literals()),
    )

    target_entity_indices = set(data.graph_target_entity_indices(0).tolist())
    object_indices = set(data.object_indices.tolist())
    assert target_entity_indices

    saw_base_relation = False
    saw_state_relation = False
    for relation_name, instances in data.flattened_relations.items():
        if not instances.numel():
            continue
        if relation_name == formatter.format_action_schema(
            adv_ground_action.get_action()
        ):
            assert set(instances[:, 0].tolist()).issubset(target_entity_indices)
            continue
        if "[state]" in relation_name:
            saw_state_relation = True
            assert set(instances[:, 0].tolist()).issubset(target_entity_indices)
        else:
            saw_base_relation = True
            assert set(instances[:, 0].tolist()).issubset(object_indices)

    assert saw_base_relation
    assert saw_state_relation

    action_schema = formatter.format_action_schema(adv_ground_action.get_action())
    action_relation = data.flattened_relations[action_schema]
    assert action_relation.shape[0] == 1
    assert action_relation[0, 0].item() == data.graph_target_positions(0)[0].item()


def test_flat_horizon_excluded_root_skips_root_parent_edges(small_blocks):
    space, domain, problem = small_blocks
    root = problem.get_initial_state()
    transitions = _first_distinct_changed_transitions(space, root, count=1)
    dag = _single_step_dag(root, transitions, candidate_ids=[101])

    data = FlatHorizonEncoder(
        domain,
        transition_mode="delta",
        root_policy="exclude",
        enable_parent_relation=True,
        enable_sibling_relation=False,
        enable_cousin_relation=False,
        ignore_actions=True,
    ).encode_pyg(
        root,
        dag=dag,
        goals=list(problem.get_goal_condition().get_literals()),
    )

    parent_instances = data.flattened_relations.get("_parent_")
    assert parent_instances is not None
    assert parent_instances.numel() == 0


def test_flat_horizon_topology_relation_schema_arities_match_packed_payloads(
    small_blocks,
):
    space, domain, problem = small_blocks
    root = problem.get_initial_state()
    transitions = _first_distinct_changed_transitions(space, root, count=2)
    if len(transitions) < 2:
        pytest.skip("Fixture does not provide enough distinct transitions.")
    dag = _single_step_dag(root, transitions[:2], candidate_ids=[101, 102])

    data = FlatHorizonEncoder(
        domain,
        transition_mode="delta",
        root_policy="exclude",
        enable_parent_relation=True,
        enable_sibling_relation=True,
        enable_cousin_relation=True,
        ignore_actions=True,
    ).encode_pyg(
        root,
        dag=dag,
        goals=list(problem.get_goal_condition().get_literals()),
    )

    for relation_name in ("_parent_", "_sibling_", "_cousin_"):
        relation_id = data.schema.name_to_id[relation_name]
        instances = data.flattened_relations[relation_name]
        assert data.schema.arities[relation_id] == 2
        assert data.schema.logical_arities[relation_id] == 0
        assert data.schema.encoded_arities[relation_id] == 2
        assert data.schema.slot_roles[relation_id] == ("state_slot", "state_slot")
        if instances.numel() > 0:
            assert instances.shape[1] == 2


def test_flat_horizon_batch_matches_from_data_list(small_blocks):
    space, domain, problem = small_blocks
    root = problem.get_initial_state()
    transitions = _first_distinct_changed_transitions(space, root, count=1)
    succ_state = transitions[0][1]

    root_dag = _single_step_dag(root, transitions, candidate_ids=[11])
    succ_dag = mifrost.TransitionDAG(adv_state(succ_state))
    encoder = FlatHorizonEncoder(domain, ignore_actions=False)

    actual = encoder.encode_batch([root, succ_state], dags=[root_dag, succ_dag]).as_pyg(
        as_batch=True
    )
    expected = flat_relation_data_from_pyg(
        Batch.from_data_list(
            [
                encoder.encode_pyg(root, dag=root_dag),
                encoder.encode_pyg(succ_state, dag=succ_dag),
            ]
        )
    )

    _assert_flat_batch_equal(actual, expected)


def test_flat_horizon_relation_major_packing_is_opt_in(small_blocks):
    space, domain, problem = small_blocks
    root = problem.get_initial_state()
    transitions = _first_distinct_changed_transitions(space, root, count=1)
    succ_state = transitions[0][1]

    root_dag = _single_step_dag(root, transitions, candidate_ids=[11])
    succ_dag = mifrost.TransitionDAG(adv_state(succ_state))
    graph_major_encoder = FlatHorizonEncoder(domain, ignore_actions=False)
    relation_major_encoder = FlatHorizonEncoder(
        domain,
        ignore_actions=False,
        pack_relation_args_relation_major=True,
    )

    graph_major = graph_major_encoder.encode_batch(
        [root, succ_state], dags=[root_dag, succ_dag]
    )
    relation_major = relation_major_encoder.encode_batch(
        [root, succ_state], dags=[root_dag, succ_dag]
    )

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


def test_flat_horizon_to_networkx_exposes_state_target_metadata(small_blocks):
    space, domain, problem = small_blocks
    root = problem.get_initial_state()
    transitions = _first_distinct_changed_transitions(space, root, count=1)
    dag = _single_step_dag(root, transitions, candidate_ids=[101])
    encoder = FlatHorizonEncoder(domain, ignore_actions=False)
    data = encoder.encode_pyg(
        root,
        dag=dag,
        goals=list(problem.get_goal_condition().get_literals()),
    )

    graph = encoder.to_networkx(data)

    assert isinstance(graph, nx.MultiDiGraph)
    target_node_name = data.graph_target_entity_names(0)[0]
    attrs = graph.nodes[target_node_name]
    assert attrs["target_group"] == "state"
    assert attrs["target_index"] == 1
    assert attrs["target_candidate_id"] == 101
    assert attrs["target_depth"] == 1
    assert len(attrs["target_rows"]) == 1


def test_flat_horizon_lgan_uses_candidate_state_rows(small_blocks):
    space, domain, problem = small_blocks
    root = problem.get_initial_state()
    transitions = _first_distinct_changed_transitions(space, root, count=1)
    dag = _single_step_dag(root, transitions, candidate_ids=[101])
    encoder = FlatHorizonEncoder(
        domain,
        ignore_actions=False,
        include_lgan_edges=True,
    )

    data = encoder.encode_pyg(
        root,
        dag=dag,
        goals=list(problem.get_goal_condition().get_literals()),
    )

    assert data.relation_instance_sizes.tolist() == [
        int(data.relation_counts.sum().item())
    ]
    tn_edges = data.graph_lgan_tn_edges(0)
    nn_edges = data.graph_lgan_nn_edges(0)
    rr_edges = data.graph_lgan_rr_edges(0)
    assert tn_edges.shape[1] > 0
    assert nn_edges.shape[0] == 2
    assert rr_edges.shape[0] == 2
    assert set(tn_edges[1].tolist()).issubset(
        set(data.graph_target_positions(0).tolist())
    )
    assert data.include_lgan_edges is True
    assert data.entity_node_type == encoder.entity_node_type


def test_flat_horizon_delta_literals_match_state_diff_fallback(small_blocks):
    space, domain, problem = small_blocks
    root = problem.get_initial_state()
    transitions = _first_distinct_changed_transitions(space, root, count=1)
    action, target = transitions[0]

    baseline_dag = _single_step_dag(root, transitions, candidate_ids=[101])
    annotated_dag = mifrost.TransitionDAG(adv_state(root))
    annotated_dag.register_transition(
        adv_state(root),
        adv_state(target),
        adv_action(action),
        candidate_id=101,
        delta_literals=_delta_literals(root, target, problem),
    )

    encoder = FlatHorizonEncoder(
        domain,
        transition_mode="delta",
        ignore_actions=False,
        ignore_zero_arity_relations=False,
    )
    actual = encoder.encode(root, dag=annotated_dag).as_pyg(as_batch=True)
    expected = encoder.encode(root, dag=baseline_dag).as_pyg(as_batch=True)

    _assert_flat_batch_equal(actual, expected)


def test_flat_horizon_lgan_respects_root_candidate_exclusion(small_blocks):
    space, domain, problem = small_blocks
    root = problem.get_initial_state()
    transitions = _first_distinct_changed_transitions(space, root, count=1)
    dag = _single_step_dag(root, transitions, candidate_ids=[101])
    encoder = FlatHorizonEncoder(
        domain,
        ignore_actions=False,
        include_lgan_edges=True,
        root_policy="exclude",
    )

    data = encoder.encode_pyg(root, dag=dag)

    assert data.graph_target_indices(0).tolist() == [1]
    tn_entity_indices = set(data.graph_lgan_tn_edges(0)[1].tolist())
    assert tn_entity_indices.issubset(set(data.graph_target_positions(0).tolist()))


def test_flat_horizon_encode_only_keeps_root_state_relations(small_blocks):
    space, domain, problem = small_blocks
    root = problem.get_initial_state()
    transitions = _first_distinct_changed_transitions(space, root, count=1)
    dag = _single_step_dag(root, transitions, candidate_ids=[101])
    data = FlatHorizonEncoder(
        domain,
        ignore_actions=False,
        root_policy="encode_only",
    ).encode_pyg(root, dag=dag)

    relation_names = set(data.flattened_relations.keys())
    assert not any("[state]" in name for name in relation_names)
    assert 0 not in data.graph_target_indices(0).tolist()


def test_flat_horizon_lgan_rejects_missing_candidate_rows(small_blocks):
    _space, domain, problem = small_blocks
    root = problem.get_initial_state()
    encoder = FlatHorizonEncoder(
        domain,
        ignore_actions=False,
        include_lgan_edges=True,
    )

    with pytest.raises(ValueError, match="requires surviving candidate state rows"):
        encoder.encode(root)


def test_flat_horizon_encode_rejects_explicit_actions_and_history(small_blocks):
    space, domain, problem = small_blocks
    root = problem.get_initial_state()
    transitions = _first_distinct_changed_transitions(space, root, count=1)
    action, _target = transitions[0]
    encoder = FlatHorizonEncoder(domain, ignore_actions=False)

    with pytest.raises(
        ValueError,
        match="FlatHorizonEncoder does not support explicit action payloads",
    ):
        encoder.encode(root, actions=[action])

    goals = list(problem.get_goal_condition().get_literals())
    with pytest.raises(
        ValueError,
        match="FlatHorizonEncoder does not support history_subgoals payloads",
    ):
        encoder.encode(root, goals=goals, history_subgoals=[(-1, goals[:1])])


def test_flat_horizon_batch_accepts_empty_unsupported_payloads(small_blocks):
    space, domain, problem = small_blocks
    root = problem.get_initial_state()
    transitions = _first_distinct_changed_transitions(space, root, count=1)
    dag = _single_step_dag(root, transitions)
    goals = list(problem.get_goal_condition().get_literals())
    encoder = FlatHorizonEncoder(domain, ignore_actions=False)

    encoding = encoder.encode_batch(
        [root],
        dags=[dag],
        goals=[goals],
        actions=[],
        history_subgoals=[],
    )

    assert encoding.num_graphs == 1
