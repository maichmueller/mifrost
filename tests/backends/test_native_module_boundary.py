from __future__ import annotations

import mifrost


def test_neutral_bindings_are_shared_with_compatibility_facade() -> None:
    neutral = mifrost._neutral_core
    compatibility = mifrost._core
    pymimir_adapter = mifrost._pymimir_adapter

    assert compatibility.BatchEncoding is neutral.BatchEncoding
    assert compatibility.BatchBuilder is neutral.BatchBuilder
    assert compatibility.FlatRelationEncoderConfig is neutral.FlatRelationEncoderConfig
    assert compatibility.GoalDerivation is neutral.GoalDerivation
    assert compatibility.TargetSource is neutral.TargetSource
    assert getattr(compatibility, "_set_batch_encoding_collate_spec") is getattr(
        neutral, "_set_batch_encoding_collate_spec"
    )
    assert compatibility.HGraphEncoderEngine is pymimir_adapter.HGraphEncoderEngine
    assert not hasattr(pymimir_adapter, "BatchEncoding")


def test_neutral_module_excludes_pymimir_encoder_bindings() -> None:
    neutral = mifrost._neutral_core

    assert hasattr(neutral, "SemanticFlatRelationEncoderEngine")
    assert not hasattr(neutral, "HGraphEncoderEngine")
    assert not hasattr(neutral, "FlatRelationEncoderEngine")
    assert not hasattr(neutral, "TransitionDAG")
