from __future__ import annotations

import mifrost
import mifrost.encoders as encoders


def test_top_level_encoder_exports_are_imported_when_available() -> None:
    for name in mifrost._encoder_exports:
        assert hasattr(mifrost, name), name
        assert name in mifrost.__all__, name


def test_rooted_flat_horizon_encoder_export_policy() -> None:
    assert hasattr(encoders, "FlatRootedHorizonEncoder")
    assert "FlatRootedHorizonEncoder" in encoders.__all__
    assert mifrost.FlatRootedHorizonEncoder is encoders.FlatRootedHorizonEncoder
