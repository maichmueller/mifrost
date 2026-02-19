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
    from collections.abc import Mapping as _Mapping

    from . import _core  # make _core cpp module explicitly available to re-export
    from .map_view import MapView, install_map_view_wrappers

    install_map_view_wrappers(_core)
    _Mapping.register(_core.RelationDict)
    from ._core import *  # noqa: F401,F403
    from .graph_fields import DType, GraphFieldSpec, Inc, Mode

    def _batch_encoding_from_payload(payload: bytes):
        return _core.BatchEncoding.loads(payload)

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
                    "`pip install mifrost[perf]` / `pip install torch torch-geometric`."
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
        "MapView",
    ]

    if _encoders_import_error is None:
        __all__ += _encoder_exports
