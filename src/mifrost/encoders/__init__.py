"""Public encoder API surface.

Exports concrete encoders, stream variants, and shared base/helpers.
"""

from .base import EncoderBase, StreamEncoderBase
from .common import _parts_to_pyg, _split_goals, parts_to_tensors
from .hgraph import HGraphEncoder, HGraphEncoderStream
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
from .types import (
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
    "_parts_to_pyg",
    "_split_goals",
    "parts_to_tensors",
    "register_state_adapter",
    "unregister_state_adapter",
    "register_domain_adapter",
    "unregister_domain_adapter",
    "register_literal_adapter",
    "unregister_literal_adapter",
    "register_action_adapter",
    "unregister_action_adapter",
]
