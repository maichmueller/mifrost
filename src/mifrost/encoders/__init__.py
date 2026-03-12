"""Public encoder API surface.

Exports concrete encoders, stream variants, and shared base/helpers.
"""

from importlib import import_module as _import_module

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

# Keep all encoder wrappers behind one declarative lazy-export table. This keeps
# `mifrost.encoders` lightweight at import time while preserving the flat public API.
_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "HGraphEncoder": (".hgraph", "HGraphEncoder"),
    "HGraphEncoderStream": (".hgraph", "HGraphEncoderStream"),
    "HGraphMutableEncoderStream": (".hgraph", "HGraphMutableEncoderStream"),
    "HorizonEncoder": (".horizon", "HorizonEncoder"),
    "HorizonEncoderStream": (".horizon", "HorizonEncoderStream"),
    "ColorEncoder": (".color", "ColorEncoder"),
    "ColorEncoderStream": (".color", "ColorEncoderStream"),
    "FlatRelationEncoder": (".flat", "FlatRelationEncoder"),
    "FlatRelationEncoderStream": (".flat", "FlatRelationEncoderStream"),
    "FlatRelationMutableEncoderStream": (
        ".flat",
        "FlatRelationMutableEncoderStream",
    ),
    "FlatHorizonEncoder": (".flat_horizon", "FlatHorizonEncoder"),
    "FlatHorizonEncoderStream": (".flat_horizon", "FlatHorizonEncoderStream"),
    "FlatTransitionEncoder": (".flat_transition", "FlatTransitionEncoder"),
    "FlatTransitionEffectsEncoder": (
        ".flat_transition",
        "FlatTransitionEffectsEncoder",
    ),
    "FlatTransitionEncoderStream": (
        ".flat_transition",
        "FlatTransitionEncoderStream",
    ),
    "FlatTransitionEffectsEncoderStream": (
        ".flat_transition",
        "FlatTransitionEffectsEncoderStream",
    ),
    "FlatRelationData": (".flat_data", "FlatRelationData"),
    "FlatRelationSchema": (".flat_data", "FlatRelationSchema"),
    "TransitionHGraphEncoder": (
        ".transition",
        "TransitionHGraphEncoder",
    ),
    "TransitionEffectsHGraphEncoder": (
        ".transition",
        "TransitionEffectsHGraphEncoder",
    ),
    "TransitionHGraphEncoderStream": (
        ".transition",
        "TransitionHGraphEncoderStream",
    ),
    "TransitionEffectsHGraphEncoderStream": (
        ".transition",
        "TransitionEffectsHGraphEncoderStream",
    ),
    "ILGEncoder": (".ilg", "ILGEncoder"),
    "ILGEncoderStream": (".ilg", "ILGEncoderStream"),
    "AtomStatus": (".ilg", "AtomStatus"),
    "ExampleConstantEncoder": (
        ".custom_example",
        "ExampleConstantEncoder",
    ),
    "ExampleConstantStreamEncoder": (
        ".custom_example",
        "ExampleConstantStreamEncoder",
    ),
}


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


__all__ = [
    "HGraphEncoder",
    "HGraphEncoderStream",
    "HGraphMutableEncoderStream",
    "HorizonEncoder",
    "HorizonEncoderStream",
    "ColorEncoder",
    "ColorEncoderStream",
    "FlatRelationEncoder",
    "FlatRelationEncoderStream",
    "FlatRelationMutableEncoderStream",
    "FlatHorizonEncoder",
    "FlatHorizonEncoderStream",
    "FlatTransitionEncoder",
    "FlatTransitionEffectsEncoder",
    "FlatTransitionEncoderStream",
    "FlatTransitionEffectsEncoderStream",
    "FlatRelationData",
    "FlatRelationSchema",
    "TransitionHGraphEncoder",
    "TransitionEffectsHGraphEncoder",
    "TransitionHGraphEncoderStream",
    "TransitionEffectsHGraphEncoderStream",
    "ILGEncoder",
    "ILGEncoderStream",
    "AtomStatus",
    "ExampleConstantEncoder",
    "ExampleConstantStreamEncoder",
    "EncoderBase",
    "StreamEncoderBase",
    "transition_dag_from_rustworkx",
    "_encoding_dict_to_pyg",
    "_split_goals",
    "encoding_to_tensors",
    "CollateSpec",
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
    "BatchEncodingLike",
    "BatchEncodingInput",
    "BatchParam",
    "FlatEncoding",
    "register_state_adapter",
    "unregister_state_adapter",
    "register_domain_adapter",
    "unregister_domain_adapter",
    "register_literal_adapter",
    "unregister_literal_adapter",
    "register_action_adapter",
    "unregister_action_adapter",
]
