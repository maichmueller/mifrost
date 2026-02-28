"""Public encoder API surface.

Exports concrete encoders, stream variants, and shared base/helpers.
"""

from .base import EncoderBase, StreamEncoderBase
from ._rustworkx_dag import transition_dag_from_rustworkx
from .common import _encoding_dict_to_pyg, _split_goals, encoding_to_tensors
from .hgraph import HGraphEncoder, HGraphEncoderStream, HGraphMutableEncoderStream
from .horizon import HorizonEncoder, HorizonEncoderStream
from .color import ColorEncoder, ColorEncoderStream
from .transition import (
    TransitionEffectsHGraphEncoder,
    TransitionEffectsHGraphEncoderStream,
    TransitionHGraphEncoder,
    TransitionHGraphEncoderStream,
)
from .ilg import ILGEncoder, ILGEncoderStream, AtomStatus
from .custom_example import ExampleConstantEncoder, ExampleConstantStreamEncoder
from ..graph_fields import CollateSpec
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

__all__ = [
    "HGraphEncoder",
    "HGraphEncoderStream",
    "HGraphMutableEncoderStream",
    "HorizonEncoder",
    "HorizonEncoderStream",
    "ColorEncoder",
    "ColorEncoderStream",
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
