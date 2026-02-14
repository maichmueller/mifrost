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
    from . import _core  # make _core cpp module explicitly available to re-export
    from .map_view import MapView, install_map_view_wrappers

    install_map_view_wrappers(_core)
    from ._core import *  # noqa: F401,F403
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
    from .graph_fields import DType, GraphFieldSpec, Inc, Mode

    def _batch_encoding_from_payload(payload: bytes):
        return _core.BatchEncoding.loads(payload)

    __all__ = [
        name
        for name in list(getattr(_core, "__all__", []))
        if not name.startswith("MapView[")
    ] + [
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
        "Mode",
        "DType",
        "Inc",
        "GraphFieldSpec",
        "MapView",
    ]
