from __future__ import annotations

import pytest

import mifrost
from mifrost.encoders import FlatRelationEncoder, HGraphEncoder

from .test_utils import adv_domain

_HISTORY_RELATION_SUFFIX = "[hist]"


def _satisfactions_from_derivations(derivations) -> set:
    mapping = {
        mifrost.GoalDerivation.satisfied: mifrost.GoalDerivation.satisfied,
        mifrost.GoalDerivation.unsatisfied: mifrost.GoalDerivation.unsatisfied,
        mifrost.GoalDerivation.added_satisfied: mifrost.GoalDerivation.added_satisfied,
        mifrost.GoalDerivation.added_unsatisfied: mifrost.GoalDerivation.added_unsatisfied,
    }
    return {mapping[derivation] for derivation in derivations if derivation in mapping}


def _domain_predicates(domain):
    adv = adv_domain(domain)
    predicates = []
    for accessor_name in (
        "get_static_predicates",
        "get_fluent_predicates",
        "get_derived_predicates",
    ):
        predicates.extend(list(getattr(adv, accessor_name)()))
    return predicates


def _relation_names_in_order(encoder) -> list[str]:
    return list(encoder.relation_dict.arity)


def _shared_relation_names(
    flat_encoder: FlatRelationEncoder,
    hgraph_encoder: HGraphEncoder,
) -> list[str]:
    hgraph_name_set = set(_relation_names_in_order(hgraph_encoder))
    return [
        name
        for name in _relation_names_in_order(flat_encoder)
        if name in hgraph_name_set
    ]


def _expected_flat_only_relation_names(domain) -> set[str]:
    formatter = mifrost.RelationFormatter
    adv = adv_domain(domain)
    expected = {formatter.format_action_schema(action) for action in adv.get_actions()}
    for predicate in _domain_predicates(domain):
        if predicate.get_arity() == 0:
            continue
        expected.add(
            formatter.format_predicate(
                predicate,
                polarity=True,
                suffix=_HISTORY_RELATION_SUFFIX,
            )
        )
        expected.add(
            formatter.format_predicate(
                predicate,
                polarity=False,
                suffix=_HISTORY_RELATION_SUFFIX,
            )
        )
    return expected


def _expected_hgraph_only_relation_names(
    domain,
    hgraph_encoder: HGraphEncoder,
) -> set[str]:
    formatter = mifrost.RelationFormatter
    expected = set()
    max_goal_level = int(hgraph_encoder.relation_dict.max_goal_level)
    goal_satisfactions = _satisfactions_from_derivations(
        hgraph_encoder.relation_dict.goal_derivations
    )
    for predicate in _domain_predicates(domain):
        if predicate.get_arity() != 0:
            continue
        expected.add(formatter.format_predicate(predicate))
        for goal_level in range(max_goal_level + 1):
            for polarity in (True, False):
                expected.add(
                    formatter.format_predicate(
                        predicate,
                        goal_level=goal_level,
                        polarity=polarity,
                    )
                )
                for satisfaction in goal_satisfactions:
                    expected.add(
                        formatter.format_predicate(
                            predicate,
                            goal_level=goal_level,
                            satisfaction=satisfaction,
                            polarity=polarity,
                        )
                    )
    return expected


def _flat_relation_counts_by_name(data) -> dict[str, int]:
    counts = data.relation_instance_counts_total().tolist()
    return {
        relation_name: int(count)
        for relation_name, count in zip(data.schema.names, counts, strict=True)
    }


def _hgraph_relation_counts_by_name(data, relation_names: list[str]) -> dict[str, int]:
    return {
        relation_name: (
            int(data[relation_name].num_nodes)
            if relation_name in data.node_types
            else 0
        )
        for relation_name in relation_names
    }


def test_flat_hgraph_shared_schema_order_and_arities_match(horizon_cases):
    _space, domain, _problem = horizon_cases
    hgraph_encoder = HGraphEncoder(domain)
    flat_encoder = FlatRelationEncoder(domain)

    flat_names = _relation_names_in_order(flat_encoder)
    hgraph_names = _relation_names_in_order(hgraph_encoder)
    shared_names = _shared_relation_names(flat_encoder, hgraph_encoder)
    hgraph_shared_names = [name for name in hgraph_names if name in set(flat_names)]

    assert shared_names == hgraph_shared_names
    assert [flat_encoder.relation_dict.arity[name] for name in shared_names] == [
        hgraph_encoder.relation_dict.arity[name] for name in shared_names
    ]


def test_flat_hgraph_schema_diffs_are_only_intentional(horizon_cases):
    _space, domain, _problem = horizon_cases
    hgraph_encoder = HGraphEncoder(domain)
    flat_encoder = FlatRelationEncoder(domain)

    hgraph_names = set(_relation_names_in_order(hgraph_encoder))
    flat_names = set(_relation_names_in_order(flat_encoder))
    flat_only = flat_names - hgraph_names
    hgraph_only = hgraph_names - flat_names

    expected_flat_only = _expected_flat_only_relation_names(domain)
    expected_hgraph_only = _expected_hgraph_only_relation_names(domain, hgraph_encoder)

    assert flat_only == expected_flat_only, (
        "Unexpected flat-only schema names.\n"
        f"expected={sorted(expected_flat_only)}\n"
        f"actual={sorted(flat_only)}"
    )
    assert hgraph_only == expected_hgraph_only, (
        "Unexpected hgraph-only schema names.\n"
        f"expected={sorted(expected_hgraph_only)}\n"
        f"actual={sorted(hgraph_only)}"
    )


def test_flat_hgraph_shared_relation_counts_match_on_single_encode(small_blocks):
    _space, domain, problem = small_blocks
    state = problem.get_initial_state()
    goals = list(problem.get_goal_condition().get_literals())
    hgraph_encoder = HGraphEncoder(domain)
    flat_encoder = FlatRelationEncoder(domain)

    hgraph_data = hgraph_encoder.encode_pyg(state, goals=goals)
    flat_data = flat_encoder.encode_pyg(state, goals=goals)
    shared_names = _shared_relation_names(flat_encoder, hgraph_encoder)
    flat_counts = _flat_relation_counts_by_name(flat_data)
    hgraph_counts = _hgraph_relation_counts_by_name(hgraph_data, shared_names)

    assert {name: flat_counts[name] for name in shared_names} == hgraph_counts
    assert {
        name: flat_counts[name] * flat_encoder.relation_dict.arity[name]
        for name in shared_names
    } == {
        name: hgraph_counts[name] * hgraph_encoder.relation_dict.arity[name]
        for name in shared_names
    }


def test_flat_hgraph_shared_relation_counts_match_on_batch_encode(small_blocks):
    space, domain, problem = small_blocks
    state = problem.get_initial_state()
    transitions = list(space.get_forward_transitions(state))
    if not transitions:
        pytest.skip("Fixture does not provide a successor for batch parity.")
    successor = transitions[0][1]
    goals = list(problem.get_goal_condition().get_literals())
    hgraph_encoder = HGraphEncoder(domain)
    flat_encoder = FlatRelationEncoder(domain)

    hgraph_batch = hgraph_encoder.encode_batch(
        [state, successor],
        goals=goals,
    ).as_pyg(as_batch=True)
    flat_batch = flat_encoder.encode_batch(
        [state, successor],
        goals=goals,
    ).as_pyg(as_batch=True)
    shared_names = _shared_relation_names(flat_encoder, hgraph_encoder)
    flat_counts = _flat_relation_counts_by_name(flat_batch)
    hgraph_counts = _hgraph_relation_counts_by_name(hgraph_batch, shared_names)

    assert {name: flat_counts[name] for name in shared_names} == hgraph_counts
    assert {
        name: flat_counts[name] * flat_encoder.relation_dict.arity[name]
        for name in shared_names
    } == {
        name: hgraph_counts[name] * hgraph_encoder.relation_dict.arity[name]
        for name in shared_names
    }
