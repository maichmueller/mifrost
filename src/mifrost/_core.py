"""Compatibility facade for backend-neutral and Pymimir native bindings.

The historical :mod:`mifrost._core` import remains stable while the actual
extensions are independently loadable. Neutral installations expose only the
planner-free surface; installations with Pymimir additionally expose its
adapter bindings.
"""

from __future__ import annotations

import importlib

from . import _neutral_core
from ._neutral_core import *  # noqa: F401,F403


def _public_names(module: object) -> list[str]:
    return [name for name in dir(module) if not name.startswith("_")]


def _install_legacy_map_view_names() -> None:
    """Preserve bracketed runtime aliases without emitting invalid stub syntax."""
    for name in dir(_neutral_core):
        if not name.startswith("_MapView_"):
            continue
        native_type = getattr(_neutral_core, name)
        key_type = native_type.key_type
        value_type = native_type.value_type
        globals()[f"MapView[{key_type.__name__},{value_type.__name__}]"] = native_type


_install_legacy_map_view_names()

# These policy constants are planner-neutral even though the historical native
# binding exported them from the Pymimir adapter module.
DEFAULT_SYMBOL_TYPE_ID = "_symbol_"
DEFAULT_LGAN_TN_EDGE_POS = "_lgan_tn_"
DEFAULT_LGAN_NN_EDGE_POS = "_lgan_nn_"
DEFAULT_LGAN_RR_EDGE_POS = "_lgan_rr_"
DEFAULT_HISTORY_LINK_RELATION = "_history_link_"


_pymimir_adapter_error: ImportError | None = None
try:
    _pymimir_adapter = importlib.import_module(f"{__package__}._pymimir_adapter")
except ImportError as error:
    _pymimir_adapter_error = error
    _pymimir_adapter = None
    ColorEncoderConfig = _neutral_core.SemanticColorEncoderConfig
    HGraphEncoderConfig = _neutral_core.SemanticHGraphEncoderConfig
else:
    for _name in dir(_pymimir_adapter):
        if _name.startswith("__"):
            continue
        globals()[_name] = getattr(_pymimir_adapter, _name)


def _require_pymimir_adapter() -> object:
    if _pymimir_adapter is None:
        raise ModuleNotFoundError(
            "The native Pymimir adapter is unavailable. Install "
            "mifrost[pymimir], or rebuild with "
            "MIFROST_BUILD_BACKENDS=pymimir (or both)."
        ) from _pymimir_adapter_error
    return _pymimir_adapter


_set_batch_encoding_collate_spec = getattr(
    _neutral_core, "_set_batch_encoding_collate_spec"
)

__all__ = _public_names(_neutral_core)
__all__ += [
    "DEFAULT_SYMBOL_TYPE_ID",
    "DEFAULT_LGAN_TN_EDGE_POS",
    "DEFAULT_LGAN_NN_EDGE_POS",
    "DEFAULT_LGAN_RR_EDGE_POS",
    "DEFAULT_HISTORY_LINK_RELATION",
]
if _pymimir_adapter is not None:
    __all__ += _public_names(_pymimir_adapter)
else:
    __all__ += ["ColorEncoderConfig", "HGraphEncoderConfig"]
__all__ = list(dict.fromkeys(__all__))
