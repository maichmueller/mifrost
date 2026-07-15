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


try:
    _pymimir_adapter = importlib.import_module(f"{__package__}._pymimir_adapter")
except ModuleNotFoundError as error:
    if error.name != f"{__package__}._pymimir_adapter":
        raise
    _pymimir_adapter = None
else:
    for _name in dir(_pymimir_adapter):
        if _name.startswith("__"):
            continue
        globals()[_name] = getattr(_pymimir_adapter, _name)

_set_batch_encoding_collate_spec = getattr(
    _neutral_core, "_set_batch_encoding_collate_spec"
)

__all__ = _public_names(_neutral_core)
if _pymimir_adapter is not None:
    __all__ += _public_names(_pymimir_adapter)
__all__ = list(dict.fromkeys(__all__))
