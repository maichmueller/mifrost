from __future__ import annotations

import pytest

import mifrost


def test_hgraph_config_accepts_kwargs() -> None:
    config = mifrost.HGraphEncoderConfig(include_static=False, symbol_type_id="_sym_")
    assert config.include_static is False
    assert config.symbol_type_id == "_sym_"


def test_horizon_config_accepts_kwargs() -> None:
    config = mifrost.HorizonEncoderConfig(
        transition_mode=mifrost.HorizonEncoderMode.Delta,
        enable_parent_relation=False,
        target_symbol_prefix="T@",
    )
    assert config.transition_mode == mifrost.HorizonEncoderMode.Delta
    assert config.enable_parent_relation is False
    assert config.target_symbol_prefix == "T@"


def test_successor_config_accepts_kwargs() -> None:
    config = mifrost.SuccessorEncoderConfig(
        successor_mode=mifrost.SuccessorEncoderMode.Delta,
        successor_suffix="[next]",
        include_successor_goal_satisfaction=False,
    )
    assert config.successor_mode == mifrost.SuccessorEncoderMode.Delta
    assert config.successor_suffix == "[next]"
    assert config.include_successor_goal_satisfaction is False


def test_color_config_accepts_kwargs() -> None:
    config = mifrost.ColorEncoderConfig(
        edge_features=True,
        enable_global_predicate_nodes=False,
    )
    assert config.edge_features is True
    assert config.enable_global_predicate_nodes is False


def test_unknown_config_kwarg_raises() -> None:
    with pytest.raises(ValueError, match="Unknown HGraphEncoderConfig kwarg"):
        mifrost.HGraphEncoderConfig(does_not_exist=True)


def test_horizon_encoder_exposes_unified_config(small_blocks) -> None:
    _, domain, _ = small_blocks
    encoder = mifrost.HorizonEncoder(
        domain,
        transition_mode=mifrost.HorizonEncoderMode.Delta,
        exclude_root_candidate=False,
        include_static=False,
        symbol_type_id="_sym_",
    )
    assert encoder.config.transition_mode == mifrost.HorizonEncoderMode.Delta
    assert encoder.config.exclude_root_candidate is False
    assert encoder.config.include_static is False
    assert encoder.config.symbol_type_id == "_sym_"


def test_transition_encoder_exposes_unified_config(small_blocks) -> None:
    _, domain, _ = small_blocks
    encoder = mifrost.TransitionHGraphEncoder(
        domain,
        successor_suffix="[next]",
        include_lgan_edges=True,
        max_goal_level=1,
    )
    assert encoder.config.successor_suffix == "[next]"
    assert encoder.config.include_lgan_edges is True
    assert encoder.config.max_goal_level == 1


def test_hgraph_encoder_exposes_relation_dict(small_blocks) -> None:
    _, domain, _ = small_blocks
    encoder = mifrost.HGraphEncoder(
        domain,
        max_goal_level=1,
        support_literals=True,
        goal_satisfaction_derivations={
            mifrost.GoalSatisfaction.satisfied,
            mifrost.GoalSatisfaction.unsatisfied,
        },
    )
    relation_dict = encoder.relation_dict
    assert type(relation_dict).__name__ == "RelationDict"
    assert isinstance(relation_dict.arity, dict)
    assert relation_dict.max_goal_level == 1
    assert relation_dict.support_literals is True
    satisfactions = set(relation_dict.goal_satisfaction_derivations)
    assert mifrost.GoalSatisfaction.none in satisfactions
    assert mifrost.GoalSatisfaction.satisfied in satisfactions
    assert mifrost.GoalSatisfaction.unsatisfied in satisfactions


def test_derived_hgraph_encoders_expose_relation_dict(small_blocks) -> None:
    _, domain, _ = small_blocks
    horizon_encoder = mifrost.HorizonEncoder(
        domain,
        enable_parent_relation=True,
    )
    transition_encoder = mifrost.TransitionHGraphEncoder(domain)

    assert type(horizon_encoder.relation_dict).__name__ == "RelationDict"
    assert isinstance(horizon_encoder.relation_dict.arity, dict)
    assert horizon_encoder.parent_relation in horizon_encoder.relation_dict.arity
    assert type(transition_encoder.relation_dict).__name__ == "RelationDict"
    assert isinstance(transition_encoder.relation_dict.arity, dict)


def test_relation_dict_constructs_from_simple_mapping() -> None:
    relation_dict = mifrost.RelationDict({"custom_rel": 2, "other_rel": 1})
    assert relation_dict.arity["custom_rel"] == 2
    assert relation_dict.arity["other_rel"] == 1


def test_hgraph_encoder_update_relations_accepts_mapping_and_relation_dict(
    small_blocks,
) -> None:
    _, domain, _ = small_blocks
    encoder = mifrost.HGraphEncoder(domain)

    encoder.update_relations({"custom_rel": 2})
    assert "custom_rel" in encoder.relation_dict.arity
    assert encoder.relation_dict.arity["custom_rel"] == 2

    custom = mifrost.RelationDict(
        {"custom_rel_2": 3},
        2,
        True,
        {
            mifrost.GoalSatisfaction.satisfied,
            mifrost.GoalSatisfaction.unsatisfied,
        },
    )
    encoder.update_relations(custom)
    assert "custom_rel_2" in encoder.relation_dict.arity
    assert encoder.relation_dict.max_goal_level == 2
    assert encoder.relation_dict.support_literals is True


def test_horizon_encoder_update_relations_reapplies_parent_relation(
    small_blocks,
) -> None:
    _, domain, _ = small_blocks
    encoder = mifrost.HorizonEncoder(domain, enable_parent_relation=True)
    encoder.update_relations({"manual_rel": 1})
    assert "manual_rel" in encoder.relation_dict.arity
    assert encoder.parent_relation in encoder.relation_dict.arity
