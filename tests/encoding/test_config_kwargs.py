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
