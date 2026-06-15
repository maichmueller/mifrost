"""Shared encoder export policy for the package root and encoder namespace."""

ENCODER_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
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
    "FlatRootedHorizonEncoder": (
        ".flat_rooted_horizon",
        "FlatRootedHorizonEncoder",
    ),
    "FlatHorizonEncoderStream": (".flat_horizon", "FlatHorizonEncoderStream"),
    "FlatHorizonMutableEncoderStream": (
        ".flat_horizon",
        "FlatHorizonMutableEncoderStream",
    ),
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

ENCODER_DIRECT_EXPORTS = (
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
)

ENCODER_NAMESPACE_EXPORTS = (*tuple(ENCODER_LAZY_EXPORTS), *ENCODER_DIRECT_EXPORTS)

TOP_LEVEL_ENCODER_EXPORTS = (
    "HGraphEncoder",
    "HGraphEncoderStream",
    "HGraphMutableEncoderStream",
    "ColorEncoder",
    "ColorEncoderStream",
    "FlatRelationEncoder",
    "FlatRelationEncoderStream",
    "FlatRelationMutableEncoderStream",
    "FlatHorizonEncoder",
    "FlatRootedHorizonEncoder",
    "FlatHorizonEncoderStream",
    "FlatHorizonMutableEncoderStream",
    "FlatTransitionEncoder",
    "FlatTransitionEffectsEncoder",
    "FlatTransitionEncoderStream",
    "FlatTransitionEffectsEncoderStream",
    "FlatRelationData",
    "FlatRelationSchema",
    "HorizonEncoder",
    "HorizonEncoderStream",
    "TransitionHGraphEncoder",
    "TransitionEffectsHGraphEncoder",
    "TransitionHGraphEncoderStream",
    "TransitionEffectsHGraphEncoderStream",
    "ILGEncoder",
    "ILGEncoderStream",
    "EncoderBase",
    "StreamEncoderBase",
    "transition_dag_from_rustworkx",
    "encoding_to_tensors",
    "BatchEncodingLike",
    "BatchEncodingInput",
    "FlatEncoding",
    "register_state_adapter",
    "unregister_state_adapter",
    "register_domain_adapter",
    "unregister_domain_adapter",
    "register_literal_adapter",
    "unregister_literal_adapter",
    "register_action_adapter",
    "unregister_action_adapter",
)

_missing_top_level_exports = set(TOP_LEVEL_ENCODER_EXPORTS) - set(
    ENCODER_NAMESPACE_EXPORTS
)
if _missing_top_level_exports:
    missing = ", ".join(sorted(_missing_top_level_exports))
    raise RuntimeError(
        f"top-level encoder exports missing from encoder namespace: {missing}"
    )

__all__ = [
    "ENCODER_LAZY_EXPORTS",
    "ENCODER_DIRECT_EXPORTS",
    "ENCODER_NAMESPACE_EXPORTS",
    "TOP_LEVEL_ENCODER_EXPORTS",
]
