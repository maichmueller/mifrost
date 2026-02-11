import torch

import mifrost


def test_map_view_wrapper_uses_nanobind_per_type_metadata():
    builder = mifrost.BatchBuilder()
    builder.set_schema_flag("edge_features", True)
    builder.add_node_features("atom", "x", torch.zeros(2, 3, dtype=torch.float32))

    flags = builder.schema_flags_view()
    dims = builder.node_feature_dims_view()

    # Public API target: one wrapper type, independent of concrete backend map type.
    assert type(flags).__name__ == "MapView"
    assert type(dims).__name__ == "MapView"
    assert isinstance(flags, mifrost.MapView[str, bool])
    assert isinstance(dims, mifrost.MapView[str, int])

    # Wrapper metadata is expected to be exposed as python types.
    assert flags.key_type is str
    assert flags.value_type is bool
    assert dims.key_type is str
    assert dims.value_type is int

    # Wrapper metadata should be directly sourced from backend class metadata
    # (individual key/value type metadata, not a pair lookup table).
    assert flags.key_type is flags._impl.__class__.key_type
    assert flags.value_type is flags._impl.__class__.value_type
    assert dims.key_type is dims._impl.__class__.key_type
    assert dims.value_type is dims._impl.__class__.value_type

    # Backend concrete map classes should surface per-type metadata.
    bool_cls = getattr(mifrost._core, "MapView[str,bool]")
    int_cls = getattr(mifrost._core, "MapView[str,int]")
    assert flags._impl.__class__ is bool_cls
    assert dims._impl.__class__ is int_cls
    assert bool_cls.key_type is str
    assert bool_cls.value_type is bool
    assert int_cls.key_type is str
    assert int_cls.value_type is int
