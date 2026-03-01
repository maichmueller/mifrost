"""Python bindings for the mifrost extension module."""

import sys as _sys

_in_stubgen = any("stubgen.py" in arg or "nanobind.stubgen" in arg for arg in _sys.argv)

if _in_stubgen:
    # nanobind stub generation imports `mifrost._core`; keep package init minimal
    # to avoid recursive extension imports during parent-package bootstrap.
    # Also bypass the editable meta-path finder so stubgen resolves `_core`
    # from the `-i` source path, not from a potentially stale installed wheel copy.
    _sys.meta_path = [
        finder
        for finder in _sys.meta_path
        if finder.__class__.__name__ != "ScikitBuildRedirectingFinder"
    ]
    __all__: list[str] = []
else:
    from collections.abc import Mapping as _ABCMapping

    from . import _core  # make _core cpp module explicitly available to re-export
    from .map_view import MapView, install_map_view_wrappers
    from .schema_keys import (
        BATCH_ATTR,
        EDGE_INDEX_ATTR_PREFIX,
        EDGE_INDEX_DST_COMPONENT,
        EDGE_INDEX_KEY_PREFIX,
        EDGE_INDEX_SRC_COMPONENT,
        EDGE_TYPE_SEPARATOR,
        PTR_ATTR,
        TYPE_ATTR_SEPARATOR,
        make_edge_type_key,
        make_type_attr_key,
    )

    install_map_view_wrappers(_core)
    _ABCMapping.register(_core.RelationDict)
    from ._core import *  # noqa: F401,F403
    from .graph_fields import CollateSpec, DType, GraphFieldSpec, Inc, Mode

    def _batch_encoding_from_payload(payload: bytes):
        return _core.BatchEncoding.loads(payload)

    def _normalize_collate_spec(
        collate_spec: _ABCMapping[str, CollateSpec | _ABCMapping[str, object]] | None,
    ) -> dict[str, dict[str, object]] | None:
        if collate_spec is None:
            return None
        if not isinstance(collate_spec, _ABCMapping):
            raise TypeError(
                f"batch_encodings collate_spec must be a mapping, got {type(collate_spec)!r}"
            )
        out: dict[str, dict[str, object]] = {}
        for key, spec in collate_spec.items():
            out[str(key)] = CollateSpec.from_spec(spec).to_core_dict()
        return out

    def batch_encodings(
        encodings,
        collate_spec: (
            _ABCMapping[str, CollateSpec | _ABCMapping[str, object]] | None
        ) = None,
    ) -> BatchEncoding:
        return _core.batch_encodings(encodings, _normalize_collate_spec(collate_spec))

    _encoder_exports = [
        "HGraphEncoder",
        "HGraphEncoderStream",
        "HGraphMutableEncoderStream",
        "ColorEncoder",
        "ColorEncoderStream",
        "HorizonEncoder",
        "HorizonEncoderStream",
        "TransitionHGraphEncoder",
        "TransitionEffectsHGraphEncoder",
        "TransitionHGraphEncoderStream",
        "TransitionEffectsHGraphEncoderStream",
        "ILGEncoder",
        "ILGEncoderStream",
        "EncoderBase",
        "StreamEncoderBase",
        "transition_dag_from_rustworkx",
        "encoding_to_tensors",
        "register_state_adapter",
        "unregister_state_adapter",
        "register_domain_adapter",
        "unregister_domain_adapter",
        "register_literal_adapter",
        "unregister_literal_adapter",
        "register_action_adapter",
        "unregister_action_adapter",
    ]

    _encoders_import_error: Exception | None = None
    try:
        from .encoders import (  # noqa: F401
            ColorEncoder,
            ColorEncoderStream,
            EncoderBase,
            HGraphEncoder,
            HGraphEncoderStream,
            HGraphMutableEncoderStream,
            HorizonEncoder,
            HorizonEncoderStream,
            ILGEncoder,
            ILGEncoderStream,
            StreamEncoderBase,
            TransitionEffectsHGraphEncoder,
            TransitionEffectsHGraphEncoderStream,
            TransitionHGraphEncoder,
            TransitionHGraphEncoderStream,
            encoding_to_tensors,
            transition_dag_from_rustworkx,
            register_action_adapter,
            register_domain_adapter,
            register_literal_adapter,
            register_state_adapter,
            unregister_action_adapter,
            unregister_domain_adapter,
            unregister_literal_adapter,
            unregister_state_adapter,
        )
    except Exception as e:  # pragma: no cover - exercised in minimal wheel tests
        # Keep `import mifrost` working for core-only consumers and for wheel
        # smoke tests. Encoder wrappers depend on optional heavy deps
        # (torch/torch_geometric).
        _encoders_import_error = e

        def __getattr__(name: str):
            if name in _encoder_exports:
                raise ModuleNotFoundError(
                    "mifrost encoder wrappers require optional dependencies. "
                    "Install with `pip install mifrost[test]` (for tests) or "
                    "`pip install mifrost[torch]` / `pip install mifrost[perf]`."
                ) from _encoders_import_error
            raise AttributeError(name)

    __all__ = [
        name
        for name in list(getattr(_core, "__all__", []))
        if not name.startswith("MapView[")
    ] + [
        "Mode",
        "DType",
        "Inc",
        "GraphFieldSpec",
        "CollateSpec",
        "MapView",
        "TYPE_ATTR_SEPARATOR",
        "EDGE_TYPE_SEPARATOR",
        "EDGE_INDEX_ATTR_PREFIX",
        "EDGE_INDEX_KEY_PREFIX",
        "EDGE_INDEX_SRC_COMPONENT",
        "EDGE_INDEX_DST_COMPONENT",
        "PTR_ATTR",
        "BATCH_ATTR",
        "make_type_attr_key",
        "make_edge_type_key",
    ]

    if _encoders_import_error is None:
        __all__ += _encoder_exports
