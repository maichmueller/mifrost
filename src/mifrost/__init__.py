"""Python bindings for the mifrost extension module."""

import sys as _sys
from pathlib import Path as _Path
from typing import Any


def _package_root() -> _Path:
    return _Path(__file__).resolve().parent


def _native_package_root() -> _Path | None:
    core_module = globals().get("_core")
    core_file = getattr(core_module, "__file__", None)
    if core_file is None:
        return None
    return _Path(core_file).resolve().parent


def _package_roots() -> tuple[_Path, ...]:
    native_root = _native_package_root()
    source_root = _package_root()
    if native_root is None or native_root == source_root:
        return (source_root,)
    return (native_root, source_root)


def _existing_library_dir() -> _Path | None:
    for pkg_root in _package_roots():
        for libdir in ("lib", "lib64"):
            candidate = pkg_root / libdir
            if candidate.is_dir():
                return candidate
    return None


def _existing_cmake_dir() -> _Path | None:
    for pkg_root in _package_roots():
        for libdir in ("lib", "lib64"):
            candidate = pkg_root / libdir / "cmake" / "mifrost"
            if candidate.is_dir():
                return candidate

    # Editable builds keep native binaries in the source tree while installing
    # the generated CMake package. Locate that installed package without
    # assuming a particular site-packages spelling.
    for entry in _sys.path:
        candidate = _Path(entry) / "mifrost" / "lib" / "cmake" / "mifrost"
        if candidate.is_dir():
            return candidate
    return None


def get_include_dir() -> str:
    pkg_root = _package_root()
    for candidate_root in _package_roots():
        include_dir = candidate_root / "include"
        if include_dir.is_dir():
            return str(include_dir)

    source_include_dir = pkg_root.parent / "_core"
    if source_include_dir.is_dir():
        return str(source_include_dir)

    return str(include_dir)


def get_include() -> str:
    return get_include_dir()


def get_library_dir() -> str:
    library_dir = _existing_library_dir()
    if library_dir is not None:
        return str(library_dir)
    return str(_package_root() / "lib")


def get_cmake_dir() -> str:
    cmake_dir = _existing_cmake_dir()
    if cmake_dir is not None:
        return str(cmake_dir)
    return str(_package_root() / "lib" / "cmake" / "mifrost")


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
    # Editable installs can also extend this package's search path with the
    # installed extension directory. Stub generation must inspect the freshly
    # built source-tree module, not a stale installed binary.
    __path__ = [str(_package_root())]
    __all__: list[str] = []
else:
    from collections.abc import Mapping as _ABCMapping

    from . import _core

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
    if hasattr(_core, "RelationDict"):
        _ABCMapping.register(_core.RelationDict)
    from ._core import *  # noqa: F401,F403
    from ._encoder_public import (
        ENCODER_OPTIONAL_DEPENDENCY_MESSAGE,
        TOP_LEVEL_ENCODER_EXPORTS,
    )
    from .graph_fields import CollateSpec, DType, GraphFieldSpec, Inc, Mode

    def _batch_encoding_from_payload(payload: bytes):
        return _core.BatchEncoding.loads(payload)

    def _normalize_collate_spec(
        collate_spec: _ABCMapping[str, CollateSpec | _ABCMapping[str, Any]] | None,
    ) -> dict[str, dict[str, Any]] | None:
        if collate_spec is None:
            return None
        if not isinstance(collate_spec, _ABCMapping):
            raise TypeError(
                f"batch_encodings collate_spec must be a mapping, got {type(collate_spec)!r}"
            )
        out: dict[str, dict[str, Any]] = {}
        for key, spec in collate_spec.items():
            out[str(key)] = CollateSpec.from_spec(spec).to_core_dict()
        return out

    def batch_encodings(  # type: ignore[misc]
        encodings,
        collate_spec: (
            _ABCMapping[str, CollateSpec | _ABCMapping[str, Any]] | None
        ) = None,
        fast_path: bool = False,
    ) -> _core.BatchEncoding:
        return _core.batch_encodings(
            encodings,
            _normalize_collate_spec(collate_spec),
            fast_path,
        )

    _encoder_exports = list(TOP_LEVEL_ENCODER_EXPORTS)

    _encoders_import_error: Exception | None = None
    try:
        from . import encoders as _encoders

        for _name in _encoder_exports:
            globals()[_name] = getattr(_encoders, _name)
    except Exception as e:  # pragma: no cover - exercised in minimal wheel tests
        # Keep `import mifrost` working for core-only consumers and for wheel
        # smoke tests. Encoder wrappers depend on optional heavy deps
        # (torch/torch_geometric).
        _encoders_import_error = e
        for _name in _encoder_exports:
            globals().pop(_name, None)

        def __getattr__(name: str):
            if name in _encoder_exports:
                if getattr(_core, "_pymimir_adapter_error", None) is not None:
                    require_adapter = getattr(_core, "_require_pymimir_adapter")
                    require_adapter()
                raise ModuleNotFoundError(ENCODER_OPTIONAL_DEPENDENCY_MESSAGE) from (
                    _encoders_import_error
                )
            raise AttributeError(name)

    _native_exports = list(getattr(_core, "__all__", []))

    __all__ = list(
        dict.fromkeys(
            name for name in _native_exports if not name.startswith("MapView[")
        )
    ) + [
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
        "get_include",
        "get_include_dir",
        "get_library_dir",
        "get_cmake_dir",
    ]

    if _encoders_import_error is None:
        __all__ += _encoder_exports
