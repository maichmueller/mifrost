"""Public encoder API surface.

Exports concrete encoders, stream variants, and shared base/helpers.
"""

# ruff: noqa: F401

import sys as _sys
from importlib import import_module as _import_module

from .._encoder_public import ENCODER_LAZY_EXPORTS, ENCODER_NAMESPACE_EXPORTS

_in_stubgen = any("stubgen.py" in arg or "nanobind.stubgen" in arg for arg in _sys.argv)

_LAZY_EXPORTS = ENCODER_LAZY_EXPORTS

_DIRECT_LAZY_EXPORTS = {
    "EncoderBase": (".base", "EncoderBase"),
    "StreamEncoderBase": (".base", "StreamEncoderBase"),
    "transition_dag_from_rustworkx": (
        "._rustworkx_dag",
        "transition_dag_from_rustworkx",
    ),
    "_split_goals": (".common", "_split_goals"),
    "_encoding_dict_to_pyg": (".conversion", "_encoding_dict_to_pyg"),
    "encoding_to_tensors": (".conversion", "encoding_to_tensors"),
    "to_pyg": (".conversion", "to_pyg"),
    "to_tensor_payload": (".conversion", "to_tensor_payload"),
    "CollateSpec": ("..graph_fields", "CollateSpec"),
    "TYPE_ATTR_SEPARATOR": ("..schema_keys", "TYPE_ATTR_SEPARATOR"),
    "EDGE_TYPE_SEPARATOR": ("..schema_keys", "EDGE_TYPE_SEPARATOR"),
    "EDGE_INDEX_ATTR_PREFIX": ("..schema_keys", "EDGE_INDEX_ATTR_PREFIX"),
    "EDGE_INDEX_KEY_PREFIX": ("..schema_keys", "EDGE_INDEX_KEY_PREFIX"),
    "EDGE_INDEX_SRC_COMPONENT": ("..schema_keys", "EDGE_INDEX_SRC_COMPONENT"),
    "EDGE_INDEX_DST_COMPONENT": ("..schema_keys", "EDGE_INDEX_DST_COMPONENT"),
    "PTR_ATTR": ("..schema_keys", "PTR_ATTR"),
    "BATCH_ATTR": ("..schema_keys", "BATCH_ATTR"),
    "make_edge_type_key": ("..schema_keys", "make_edge_type_key"),
    "make_type_attr_key": ("..schema_keys", "make_type_attr_key"),
    "BatchEncodingInput": (".types", "BatchEncodingInput"),
    "BatchEncodingLike": (".types", "BatchEncodingLike"),
    "BatchParam": (".types", "BatchParam"),
    "FlatEncoding": (".types", "FlatEncoding"),
    "register_action_adapter": (".types", "register_action_adapter"),
    "register_domain_adapter": (".types", "register_domain_adapter"),
    "register_literal_adapter": (".types", "register_literal_adapter"),
    "register_state_adapter": (".types", "register_state_adapter"),
    "unregister_action_adapter": (".types", "unregister_action_adapter"),
    "unregister_domain_adapter": (".types", "unregister_domain_adapter"),
    "unregister_literal_adapter": (".types", "unregister_literal_adapter"),
    "unregister_state_adapter": (".types", "unregister_state_adapter"),
}


def __getattr__(name: str):
    target = _LAZY_EXPORTS.get(name) or _DIRECT_LAZY_EXPORTS.get(name)
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
