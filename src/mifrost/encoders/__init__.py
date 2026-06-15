"""Public encoder API surface.

Exports concrete encoders, stream variants, and shared base/helpers.
"""

# ruff: noqa: F401

import sys as _sys
from importlib import import_module as _import_module

from .._encoder_public import ENCODER_LAZY_EXPORTS, ENCODER_NAMESPACE_EXPORTS

_in_stubgen = any("stubgen.py" in arg or "nanobind.stubgen" in arg for arg in _sys.argv)

_LAZY_EXPORTS = ENCODER_LAZY_EXPORTS

if not _in_stubgen:
    from .base import EncoderBase, StreamEncoderBase
    from ._rustworkx_dag import transition_dag_from_rustworkx
    from .common import _encoding_dict_to_pyg, _split_goals, encoding_to_tensors
    from ..graph_fields import CollateSpec
    from ..schema_keys import (
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
    from .types import (
        BatchEncodingInput,
        BatchEncodingLike,
        BatchParam,
        FlatEncoding,
        register_action_adapter,
        register_domain_adapter,
        register_literal_adapter,
        register_state_adapter,
        unregister_action_adapter,
        unregister_domain_adapter,
        unregister_literal_adapter,
        unregister_state_adapter,
    )


def __getattr__(name: str):
    target = _LAZY_EXPORTS.get(name)
    if target is not None:
        module_name, attr_name = target
        value = getattr(_import_module(module_name, __name__), attr_name)
        globals()[name] = value
        return value
    raise AttributeError(name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__) | set(_LAZY_EXPORTS))


if _in_stubgen:
    __all__ = list(_LAZY_EXPORTS)
else:
    __all__ = list(ENCODER_NAMESPACE_EXPORTS)
