from __future__ import annotations

import importlib
import inspect

import pytest

import mifrost
import mifrost.encoders as encoders
import mifrost.encoders.common as encoder_common
from mifrost._encoder_public import (
    ENCODER_DIRECT_EXPORTS,
    ENCODER_EXPORT_MANIFEST,
    ENCODER_LAZY_EXPORTS,
    ENCODER_NAMESPACE_EXPORTS,
    TOP_LEVEL_ENCODER_EXPORTS,
    encoder_direct_export_names,
    encoder_lazy_imports,
    encoder_namespace_export_names,
    top_level_encoder_export_names,
)

#: Public helpers that sit in an exported module's ``__all__`` but are reached
#: through that module rather than through ``mifrost.encoders``.
MODULE_ONLY_EXPORTS = frozenset(
    {
        "flat_relation_data_from_pyg",
        "normalize_flat_relation_batch_metadata",
    }
)


def test_encoder_export_tables_are_derived_from_manifest() -> None:
    assert ENCODER_LAZY_EXPORTS == ENCODER_EXPORT_MANIFEST.lazy_imports
    assert ENCODER_DIRECT_EXPORTS == ENCODER_EXPORT_MANIFEST.direct_export_names
    assert ENCODER_NAMESPACE_EXPORTS == ENCODER_EXPORT_MANIFEST.namespace_export_names
    assert TOP_LEVEL_ENCODER_EXPORTS == ENCODER_EXPORT_MANIFEST.top_level_export_names

    assert encoder_lazy_imports() == ENCODER_LAZY_EXPORTS
    assert encoder_direct_export_names() == ENCODER_DIRECT_EXPORTS
    assert encoder_namespace_export_names() == ENCODER_NAMESPACE_EXPORTS
    assert top_level_encoder_export_names() == TOP_LEVEL_ENCODER_EXPORTS


def test_encoder_common_compatibility_wrappers_are_available() -> None:
    for name in (
        "_convert_batch_payload",
        "_encoding_dict_to_pyg",
        "to_pyg",
        "to_tensor_payload",
        "encoding_to_tensors",
    ):
        assert callable(getattr(encoder_common, name))


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


def test_exported_modules_publish_their_api_through_the_manifest() -> None:
    """A module's new public class or function needs a manifest entry.

    The manifest is otherwise only checked against itself, so a module that
    grows a public name without an entry would go unnoticed. Names that are
    deliberately module-only belong in ``MODULE_ONLY_EXPORTS`` instead.
    """
    pytest.importorskip("torch")
    namespace = set(ENCODER_NAMESPACE_EXPORTS)
    unpublished = []
    for module_name in sorted({module for module, _ in ENCODER_LAZY_EXPORTS.values()}):
        module = importlib.import_module(module_name, encoders.__name__)
        for name in getattr(module, "__all__", ()):
            value = getattr(module, name)
            if not (inspect.isclass(value) or inspect.isfunction(value)):
                continue
            if name not in namespace and name not in MODULE_ONLY_EXPORTS:
                unpublished.append(f"{module_name}.{name}")

    assert not unpublished, (
        "public names missing from the encoder export manifest "
        f"(src/mifrost/_encoder_public.py): {unpublished}"
    )
