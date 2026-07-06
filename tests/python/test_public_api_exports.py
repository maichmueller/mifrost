from __future__ import annotations

import mifrost
import mifrost.encoders as encoders
from mifrost._encoder_public import (
    ENCODER_NAMESPACE_EXPORTS,
    TOP_LEVEL_ENCODER_EXPORTS,
)


def test_encoder_export_manifest_has_no_duplicate_names() -> None:
    assert len(ENCODER_NAMESPACE_EXPORTS) == len(set(ENCODER_NAMESPACE_EXPORTS))
    assert len(TOP_LEVEL_ENCODER_EXPORTS) == len(set(TOP_LEVEL_ENCODER_EXPORTS))


def test_encoder_namespace_exports_match_manifest() -> None:
    assert tuple(encoders.__all__) == ENCODER_NAMESPACE_EXPORTS


def test_top_level_encoder_exports_are_imported_when_available() -> None:
    assert tuple(mifrost._encoder_exports) == TOP_LEVEL_ENCODER_EXPORTS
    for name in TOP_LEVEL_ENCODER_EXPORTS:
        assert name in ENCODER_NAMESPACE_EXPORTS, name
        assert hasattr(mifrost, name), name
        assert name in mifrost.__all__, name


def test_rooted_flat_horizon_encoder_export_policy() -> None:
    assert hasattr(encoders, "FlatRootedHorizonEncoder")
    assert "FlatRootedHorizonEncoder" in encoders.__all__
    assert mifrost.FlatRootedHorizonEncoder is encoders.FlatRootedHorizonEncoder


def test_encoder_namespace_only_exports_are_not_top_level() -> None:
    assert "AtomStatus" in encoders.__all__
    assert "AtomStatus" not in TOP_LEVEL_ENCODER_EXPORTS
    assert "AtomStatus" not in mifrost.__all__
