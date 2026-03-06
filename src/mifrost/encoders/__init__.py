"""Public encoder API surface.

Exports concrete encoders, stream variants, and shared base/helpers.
"""

import sys as _sys

_in_stubgen = any("stubgen.py" in arg or "nanobind.stubgen" in arg for arg in _sys.argv)

from .base import EncoderBase, StreamEncoderBase
from ._rustworkx_dag import transition_dag_from_rustworkx
from .common import _encoding_dict_to_pyg, _split_goals, encoding_to_tensors
from .hgraph import HGraphEncoder, HGraphEncoderStream, HGraphMutableEncoderStream
from .horizon import HorizonEncoder, HorizonEncoderStream
from .color import ColorEncoder, ColorEncoderStream
from .flat_data import FlatRelationData, FlatRelationSchema
from .transition import (
    TransitionEffectsHGraphEncoder,
    TransitionEffectsHGraphEncoderStream,
    TransitionHGraphEncoder,
    TransitionHGraphEncoderStream,
)
from .ilg import ILGEncoder, ILGEncoderStream, AtomStatus
from .custom_example import ExampleConstantEncoder, ExampleConstantStreamEncoder
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
    BatchParam,
    register_action_adapter,
    register_domain_adapter,
    register_literal_adapter,
    register_state_adapter,
    unregister_action_adapter,
    unregister_domain_adapter,
    unregister_literal_adapter,
    unregister_state_adapter,
)

if not _in_stubgen:
    from .flat import FlatRelationEncoder
    from .flat_horizon import FlatHorizonEncoder
    from .flat_transition import FlatTransitionEncoder, FlatTransitionEffectsEncoder


def __getattr__(name: str):
    if name == "FlatRelationEncoder":
        from .flat import FlatRelationEncoder as _FlatRelationEncoder

        globals()[name] = _FlatRelationEncoder
        return _FlatRelationEncoder
    if name == "FlatHorizonEncoder":
        from .flat_horizon import FlatHorizonEncoder as _FlatHorizonEncoder

        globals()[name] = _FlatHorizonEncoder
        return _FlatHorizonEncoder
    if name == "FlatTransitionEncoder":
        from .flat_transition import FlatTransitionEncoder as _FlatTransitionEncoder

        globals()[name] = _FlatTransitionEncoder
        return _FlatTransitionEncoder
    if name == "FlatTransitionEffectsEncoder":
        from .flat_transition import (
            FlatTransitionEffectsEncoder as _FlatTransitionEffectsEncoder,
        )

        globals()[name] = _FlatTransitionEffectsEncoder
        return _FlatTransitionEffectsEncoder
    raise AttributeError(name)


__all__ = [
    "HGraphEncoder",
    "HGraphEncoderStream",
    "HGraphMutableEncoderStream",
    "HorizonEncoder",
    "HorizonEncoderStream",
    "ColorEncoder",
    "ColorEncoderStream",
    "FlatRelationEncoder",
    "FlatHorizonEncoder",
    "FlatTransitionEncoder",
    "FlatTransitionEffectsEncoder",
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
    "BatchParam",
    "register_state_adapter",
    "unregister_state_adapter",
    "register_domain_adapter",
    "unregister_domain_adapter",
    "register_literal_adapter",
    "unregister_literal_adapter",
    "register_action_adapter",
    "unregister_action_adapter",
]
